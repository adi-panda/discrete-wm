"""
Train the PatchVQ-VAE tokenizer on collected Atari frames.

Usage:
    python train_tokenizer.py --data data/atari_data.npz --steps 50000
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import imageio

from models.discrete_diffusion import PatchVQVAE
from hf_utils import push_checkpoint, ensure_repo


def train_tokenizer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data = np.load(args.data)
    frames = torch.from_numpy(data['frames'])  # [N, 64, 64, 3] uint8
    print(f"Loaded {len(frames)} frames")

    # Dataset
    dataset = TensorDataset(frames)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    # Model
    model = PatchVQVAE(
        patch_size=args.patch_size,
        num_channels=3,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"PatchVQ-VAE parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"Token grid: {64 // args.patch_size}x{64 // args.patch_size} = "
          f"{(64 // args.patch_size) ** 2} tokens per frame")
    print(f"Vocab size: {args.vocab_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # LR schedule
    def lr_lambda(step):
        if step < 1000:
            return step / 1000
        return 0.5 * (1 + math.cos(math.pi * (step - 1000) / max(1, args.steps - 1000)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    model.train()
    data_iter = iter(dataloader)
    start_time = time.time()
    running_loss = 0.0
    running_recon = 0.0
    codebook_usage = set()

    pbar = tqdm(range(args.steps), desc="Training tokenizer")

    for step in pbar:
        try:
            (batch_frames,) = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            (batch_frames,) = next(data_iter)

        batch_frames = batch_frames.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            recon, tokens, recon_loss, commitment_loss, codebook_loss = model(batch_frames)

        loss = recon_loss + 0.25 * commitment_loss + codebook_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_recon += recon_loss.item()
        codebook_usage.update(tokens.unique().cpu().tolist())

        if (step + 1) % 100 == 0:
            avg_loss = running_loss / 100
            avg_recon = running_recon / 100
            usage_pct = len(codebook_usage) / args.vocab_size * 100

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'recon': f'{avg_recon:.4f}',
                'usage': f'{usage_pct:.0f}%',
            })
            running_loss = 0.0
            running_recon = 0.0

        if (step + 1) % 5000 == 0:
            # Save sample reconstructions
            model.eval()
            with torch.no_grad():
                sample = batch_frames[:8]
                recon, tokens, _, _, _ = model(sample)
                recon_uint8 = ((recon.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
                orig_uint8 = sample.cpu().numpy()

                rows = []
                for i in range(min(8, len(sample))):
                    row = np.concatenate([orig_uint8[i], recon_uint8[i]], axis=1)
                    rows.append(row)
                grid = np.concatenate(rows, axis=0)
                imageio.imwrite(os.path.join(args.sample_dir, f'tokenizer_step_{step+1:06d}.png'), grid)

            model.train()

        if (step + 1) % 10000 == 0:
            step_path = os.path.join(args.ckpt_dir, f'tokenizer_step_{step+1:06d}.pt')
            torch.save({
                'model': model.state_dict(),
                'step': step + 1,
                'args': vars(args),
            }, step_path)
            if args.hf_push:
                push_checkpoint(step_path, exp_name=args.exp_name, repo_id=args.hf_repo)

    # Save final
    final_path = os.path.join(args.ckpt_dir, 'tokenizer_final.pt')
    torch.save({
        'model': model.state_dict(),
        'step': args.steps,
        'args': vars(args),
    }, final_path)
    if args.hf_push:
        # Final: block so script doesn't exit before upload finishes
        push_checkpoint(final_path, exp_name=args.exp_name, repo_id=args.hf_repo,
                        blocking=True)

    elapsed = time.time() - start_time
    print(f"\nTokenizer training complete: {args.steps} steps in {elapsed:.0f}s")
    print(f"Codebook utilization: {len(codebook_usage)}/{args.vocab_size} "
          f"({len(codebook_usage)/args.vocab_size*100:.1f}%)")
    print(f"Saved to {final_path}")

    # Compute reconstruction PSNR on a batch
    model.eval()
    with torch.no_grad():
        sample = frames[:256].to(device)
        recon, _, _, _, _ = model(sample)
        recon_uint8 = ((recon.clamp(-1, 1) + 1) / 2 * 255)
        orig_float = sample.float()
        mse = ((recon_uint8 - orig_float) ** 2).mean()
        psnr = 10 * math.log10(255.0 ** 2 / mse.item())
        print(f"Reconstruction PSNR: {psnr:.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/vast/adi/discrete_wm/data/atari_data.npz")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--ckpt-dir", default="/vast/adi/discrete_wm/checkpoints")
    parser.add_argument("--sample-dir", default="/vast/adi/discrete_wm/figures/tokenizer")

    # HuggingFace Hub
    parser.add_argument("--hf-repo", default="adipanda/discrete-wm")
    parser.add_argument("--exp-name", default="breakout-v1")
    parser.add_argument("--hf-push", action=argparse.BooleanOptionalAction, default=True,
                        help="Push checkpoints to HF hub after each save (use --no-hf-push to disable)")

    args = parser.parse_args()
    if args.hf_push:
        ensure_repo(args.hf_repo)
    train_tokenizer(args)
