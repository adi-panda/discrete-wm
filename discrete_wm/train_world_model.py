"""
Training script for the discrete diffusion world model.

Two-stage pipeline:
1. First train the VQ-VAE tokenizer (train_tokenizer.py)
2. Pre-tokenize the dataset (this script with --pretokenize)
3. Train the world model on tokenized data

Usage:
    python train_world_model.py --pretokenize --tokenizer-ckpt checkpoints/tokenizer_final.pt
    python train_world_model.py --data data/tokenized_data.npz --steps 200000
"""

import argparse
import math
import os
import time
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.discrete_diffusion import DiscreteWorldModel, PatchVQVAE
from utils import AtariFramePairDataset, AtariTokenizedDataset, cosine_mask_schedule
from hf_utils import push_checkpoint, ensure_repo


def pretokenize_dataset(args):
    """Pre-tokenize the entire dataset using the trained VQ-VAE."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    ckpt = torch.load(args.tokenizer_ckpt, map_location=device)
    tok_args = ckpt['args']
    tokenizer = PatchVQVAE(
        patch_size=tok_args['patch_size'],
        num_channels=3,
        vocab_size=tok_args['vocab_size'],
        embed_dim=tok_args['embed_dim'],
    ).to(device)
    tokenizer.load_state_dict(ckpt['model'])
    tokenizer.eval()

    # Load raw data
    data = np.load(args.raw_data)
    frames = torch.from_numpy(data['frames'])
    actions = data['actions']
    num_actions = int(data['num_actions'])

    print(f"Tokenizing {len(frames)} frames...")

    all_tokens = []
    batch_size = 256
    with torch.no_grad():
        for i in tqdm(range(0, len(frames), batch_size)):
            batch = frames[i:i+batch_size].to(device)
            tokens, _, _, _ = tokenizer.encode(batch)
            all_tokens.append(tokens.cpu().numpy())

    all_tokens = np.concatenate(all_tokens, axis=0)  # [N, num_patches]
    print(f"Tokenized: {all_tokens.shape}, vocab range: [{all_tokens.min()}, {all_tokens.max()}]")

    # Build pairs
    prev_tokens = all_tokens[:-1]  # [N-1, num_patches]
    next_tokens = all_tokens[1:]   # [N-1, num_patches]

    # Trim to match actions length
    n = min(len(prev_tokens), len(actions))
    prev_tokens = prev_tokens[:n]
    next_tokens = next_tokens[:n]
    actions_trimmed = actions[:n]

    save_path = os.path.join(os.path.dirname(args.raw_data), 'tokenized_data.npz')
    np.savez_compressed(
        save_path,
        prev_tokens=prev_tokens.astype(np.int16),
        next_tokens=next_tokens.astype(np.int16),
        actions=actions_trimmed,
        num_actions=num_actions,
    )
    print(f"Saved tokenized data to {save_path}")
    return save_path


def _save_and_push(state, local_path, args, remote_name=None):
    torch.save(state, local_path)
    if args.hf_push:
        push_checkpoint(
            local_path,
            exp_name=args.exp_name,
            repo_id=args.hf_repo,
            remote_name=remote_name,
        )


def save_sample_images(model, tokenizer, dataset, device, step, save_dir,
                       num_samples=8, num_steps=8):
    """Generate and save sample predictions."""
    import imageio

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    indices = list(range(0, min(num_samples * 10, len(dataset)), 10))[:num_samples]
    prev_list, act_list, gt_list = [], [], []
    for idx in indices:
        pt, a, nt = dataset[idx]
        prev_list.append(pt)
        act_list.append(a)
        gt_list.append(nt)

    prev_tokens = torch.stack(prev_list).to(device)
    actions = torch.stack(act_list).to(device)
    gt_tokens = torch.stack(gt_list).to(device)

    with torch.no_grad():
        pred_tokens = model.generate(prev_tokens, actions, num_steps=num_steps,
                                      temperature=0.9, device=device)

        # Decode all with tokenizer
        prev_frames = tokenizer.decode_tokens(prev_tokens)
        gt_frames = tokenizer.decode_tokens(gt_tokens)
        pred_frames = tokenizer.decode_tokens(pred_tokens)

        # Convert to uint8
        prev_uint8 = ((prev_frames.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
        gt_uint8 = ((gt_frames.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
        pred_uint8 = ((pred_frames.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()

    rows = []
    for i in range(min(num_samples, len(pred_uint8))):
        row = np.concatenate([prev_uint8[i], gt_uint8[i], pred_uint8[i]], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    imageio.imwrite(os.path.join(save_dir, f'samples_step_{step:06d}.png'), grid)

    model.train()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer (needed for decoding samples)
    tok_ckpt = torch.load(args.tokenizer_ckpt, map_location=device)
    tok_args = tok_ckpt['args']
    tokenizer = PatchVQVAE(
        patch_size=tok_args['patch_size'],
        num_channels=3,
        vocab_size=tok_args['vocab_size'],
        embed_dim=tok_args['embed_dim'],
    ).to(device)
    tokenizer.load_state_dict(tok_ckpt['model'])
    tokenizer.eval()

    # Dataset
    dataset = AtariTokenizedDataset(args.data)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )

    grid_size = int(math.sqrt(dataset.prev_tokens.shape[1]))

    # Model
    model = DiscreteWorldModel(
        vocab_size=tok_args['vocab_size'],
        grid_h=grid_size,
        grid_w=grid_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_actions=dataset.num_actions,
        dropout=args.dropout,
        cond_dim=args.d_model,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # LR schedule: warmup + cosine decay
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    best_loss = float('inf')

    # CSV logging
    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'loss', 'lr', 'time', 'mask_ratio_mean'])

    # Resume
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        global_step = ckpt['step']
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f"Resumed from step {global_step}")

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda')
    mask_token_id = tok_args['vocab_size']

    print(f"\nStarting training for {args.total_steps} steps...")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Model: {args.n_layers}L, {args.d_model}D, {args.n_heads}H")
    print(f"Vocab: {tok_args['vocab_size']} codes, grid: {grid_size}x{grid_size} = "
          f"{grid_size**2} tokens/frame")

    model.train()
    data_iter = iter(dataloader)
    start_time = time.time()
    running_loss = 0.0
    log_interval = 100

    pbar = tqdm(range(global_step, args.total_steps), desc="Training")

    for step_idx in pbar:
        try:
            prev_tokens, actions, next_tokens = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            prev_tokens, actions, next_tokens = next(data_iter)

        prev_tokens = prev_tokens.to(device).long()
        actions = actions.to(device)
        next_tokens = next_tokens.to(device).long()

        B, N = next_tokens.shape

        # Sample masking ratio
        t = torch.rand(B, device=device)
        mask_ratio = cosine_mask_schedule(t)

        # Create mask
        mask = torch.rand(B, N, device=device) < mask_ratio.unsqueeze(1)
        # Ensure at least one token is masked
        if not mask.any():
            mask[0, 0] = True

        # Masked version
        masked_next = next_tokens.clone()
        masked_next[mask] = mask_token_id

        # Forward
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(masked_next, prev_tokens, actions, mask_ratio)
            # Loss on masked positions only
            logits_masked = logits[mask]  # [num_masked, vocab_size]
            targets_masked = next_tokens[mask]  # [num_masked]
            loss = F.cross_entropy(logits_masked, targets_masked)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1
        loss_val = loss.item()
        running_loss += loss_val

        if global_step % log_interval == 0:
            avg_loss = running_loss / log_interval
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            sps = log_interval / (time.time() - start_time) if global_step == log_interval else global_step / elapsed

            csv_writer.writerow([global_step, avg_loss, lr, elapsed, mask_ratio.mean().item()])
            csv_file.flush()

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{lr:.2e}',
                'sps': f'{sps:.1f}',
            })
            running_loss = 0.0

        if global_step % args.sample_every == 0:
            save_sample_images(
                model, tokenizer, dataset, device, global_step, args.sample_dir,
                num_samples=8, num_steps=args.gen_steps
            )
            model.train()

        if global_step % args.save_every == 0:
            full_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': global_step,
                'best_loss': best_loss,
                'args': vars(args),
                'tokenizer_args': tok_args,
            }

            ckpt_path = os.path.join(args.ckpt_dir, f'model_step_{global_step:06d}.pt')
            _save_and_push(full_state, ckpt_path, args)

            latest_path = os.path.join(args.ckpt_dir, 'model_latest.pt')
            _save_and_push(full_state, latest_path, args)

            if loss_val < best_loss:
                best_loss = loss_val
                best_state = {
                    'model': model.state_dict(),
                    'step': global_step,
                    'loss': best_loss,
                    'args': vars(args),
                    'tokenizer_args': tok_args,
                }
                _save_and_push(
                    best_state,
                    os.path.join(args.ckpt_dir, 'model_best.pt'),
                    args,
                )

    csv_file.close()
    print(f"\nTraining complete. {global_step} steps in {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument("--pretokenize", action="store_true")
    parser.add_argument("--raw-data", default="/vast/adi/discrete_wm/data/atari_data.npz")
    parser.add_argument("--tokenizer-ckpt", default="/vast/adi/discrete_wm/checkpoints/tokenizer_final.pt")

    # Data
    parser.add_argument("--data", default="/vast/adi/discrete_wm/data/tokenized_data.npz")

    # Model
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Inference
    parser.add_argument("--gen-steps", type=int, default=8)

    # Checkpointing
    parser.add_argument("--ckpt-dir", default="/vast/adi/discrete_wm/checkpoints")
    parser.add_argument("--log-dir", default="/vast/adi/discrete_wm/logs")
    parser.add_argument("--sample-dir", default="/vast/adi/discrete_wm/figures/samples")
    parser.add_argument("--save-every", type=int, default=20000)
    parser.add_argument("--sample-every", type=int, default=10000)
    parser.add_argument("--resume", default=None)

    # HuggingFace Hub
    parser.add_argument("--hf-repo", default="adipanda/discrete-wm")
    parser.add_argument("--exp-name", default="breakout-v1")
    parser.add_argument("--hf-push", action=argparse.BooleanOptionalAction, default=True,
                        help="Push checkpoints to HF hub after each save (use --no-hf-push to disable)")

    args = parser.parse_args()

    if args.pretokenize:
        pretokenize_dataset(args)
    else:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.sample_dir, exist_ok=True)
        if args.hf_push:
            ensure_repo(args.hf_repo)
        train(args)
