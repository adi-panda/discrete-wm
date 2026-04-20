"""
Standalone DIAMOND denoiser training on static dataset.

Bypasses DIAMOND's full Hydra + Trainer pipeline to train just the denoiser
on our pre-collected ablation dataset. This avoids env creation, collector setup,
actor-critic training, and WandB dependencies.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'diamond', 'src'))

from models.diffusion import Denoiser, DenoiserConfig
from models.diffusion.inner_model import InnerModel, InnerModelConfig


class DiamondEpisodeDataset(Dataset):
    """Load DIAMOND-format episodes and return fixed-length segments."""

    def __init__(self, dataset_dir, seq_length=5, cache_in_ram=True):
        from pathlib import Path
        self.seq_length = seq_length
        self.episodes = []
        self.episode_lengths = []

        dataset_path = Path(dataset_dir)
        pt_files = sorted(dataset_path.rglob("*.pt"))
        pt_files = [f for f in pt_files if f.name != "info.pt"]

        for pt_file in tqdm(pt_files, desc="Loading episodes"):
            ep = torch.load(pt_file, map_location='cpu', weights_only=False)
            obs = ep['obs']
            if obs.dtype == torch.uint8:
                obs = obs.float().div(255).mul(2).sub(1)
            if obs.dim() == 4 and obs.shape[-1] == 3:
                obs = obs.permute(0, 3, 1, 2)
            act = ep['act'].long()
            self.episodes.append({'obs': obs, 'act': act})
            self.episode_lengths.append(len(obs))

        self.total_segments = sum(max(0, l - seq_length + 1) for l in self.episode_lengths)
        print(f"Loaded {len(self.episodes)} episodes, {self.total_segments} segments (seq_len={seq_length})")

    def __len__(self):
        return self.total_segments

    def __getitem__(self, idx):
        for i, ep in enumerate(self.episodes):
            max_start = self.episode_lengths[i] - self.seq_length
            if max_start < 0:
                continue
            if idx <= max_start:
                obs = ep['obs'][idx:idx + self.seq_length]
                act = ep['act'][idx:idx + self.seq_length - 1]
                if len(act) < self.seq_length - 1:
                    act = F.pad(act, (0, self.seq_length - 1 - len(act)))
                return obs, act
            idx -= max_start + 1
        return self.episodes[0]['obs'][:self.seq_length], self.episodes[0]['act'][:self.seq_length - 1]


def build_denoiser(num_actions=4):
    inner_cfg = InnerModelConfig(
        img_channels=3,
        num_steps_conditioning=4,
        cond_channels=256,
        depths=[2, 2, 2, 2],
        channels=[64, 64, 64, 64],
        attn_depths=[0, 0, 0, 0],
        num_actions=num_actions,
    )
    denoiser_cfg = DenoiserConfig(
        inner_model=inner_cfg,
        sigma_data=0.5,
        sigma_offset_noise=0.3,
    )
    return Denoiser(denoiser_cfg)


def sample_sigma(n, device, loc=-0.4, scale=1.2, sigma_min=2e-3, sigma_max=20):
    s = torch.randn(n, device=device) * scale + loc
    return s.exp().clip(sigma_min, sigma_max)


def add_dims(input, n):
    return input.reshape(input.shape + (1,) * (n - input.ndim))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="/vast/adi/discrete_wm/data/ablation_v1/diamond_dataset/train")
    parser.add_argument("--ckpt-dir", default="/vast/adi/discrete_wm/checkpoints/ablation_v1")
    parser.add_argument("--log-dir", default="/vast/adi/discrete_wm/logs/ablation_v1")
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-steps-conditioning", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=20000)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seq_length = args.num_steps_conditioning + 1 + 1
    dataset = DiamondEpisodeDataset(args.train_dir, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)

    denoiser = build_denoiser(num_actions=4).to(device)
    num_params = sum(p.numel() for p in denoiser.parameters())
    print(f"DIAMOND denoiser: {num_params:,} params ({num_params / 1e6:.1f}M)")

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_lambda(step):
        warmup = 100
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, args.total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    import csv
    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, 'diamond_training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'loss', 'lr', 'time'])

    os.makedirs(args.ckpt_dir, exist_ok=True)

    denoiser.train()
    data_iter = iter(dataloader)
    global_step = 0
    running_loss = 0.0
    start_time = time.time()
    log_interval = 100

    pbar = tqdm(range(args.total_steps), desc="Training DIAMOND denoiser")

    for step_idx in pbar:
        try:
            obs_seq, act_seq = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            obs_seq, act_seq = next(data_iter)

        obs_seq = obs_seq.to(device)
        act_seq = act_seq.to(device)

        n = args.num_steps_conditioning
        obs = obs_seq[:, :n]
        next_obs = obs_seq[:, n]
        act = act_seq[:, :n]

        b, t, c, h, w = obs.shape
        obs_flat = obs.reshape(b, t * c, h, w)

        sigma = sample_sigma(b, device)
        noisy = denoiser.apply_noise(next_obs, sigma, denoiser.cfg.sigma_offset_noise)

        cs = denoiser.compute_conditioners(sigma)
        model_output = denoiser.compute_model_output(noisy, obs_flat, act, cs)

        target = (next_obs - cs.c_skip * noisy) / cs.c_out
        loss = F.mse_loss(model_output, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        global_step += 1
        loss_val = loss.item()
        running_loss += loss_val

        if global_step % log_interval == 0:
            avg_loss = running_loss / log_interval
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            csv_writer.writerow([global_step, avg_loss, lr, elapsed])
            csv_file.flush()
            pbar.set_postfix(loss=f"{avg_loss:.6f}", lr=f"{lr:.2e}")
            running_loss = 0.0

        if global_step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f'diamond_step_{global_step:06d}.pt')
            torch.save(denoiser.state_dict(), ckpt_path)
            print(f"\nSaved checkpoint: {ckpt_path}")

    final_path = os.path.join(args.ckpt_dir, 'diamond_final.pt')
    torch.save(denoiser.state_dict(), final_path)
    print(f"\nSaved final checkpoint: {final_path}")

    csv_file.close()
    elapsed = time.time() - start_time
    print(f"\nDIAMOND training complete. {global_step} steps in {elapsed:.0f}s ({elapsed/3600:.1f}h)")


if __name__ == "__main__":
    main()
