"""
Train an Inverse Dynamics Model (IDM) on ground-truth Atari frames.

Input: (prev_frame, next_frame) concatenated → 6 channels
Output: action prediction (4 classes for Breakout)

Used as an unbiased controllability metric for comparing world models.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class IDM(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_actions),
        )

    def forward(self, prev_frame, next_frame):
        x = torch.cat([prev_frame, next_frame], dim=1)
        x = self.conv(x)
        return self.head(x)


class FramePairDataset(Dataset):
    def __init__(self, frames, actions, indices):
        self.frames = frames
        self.actions = actions
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        prev = torch.from_numpy(self.frames[i].copy()).float().permute(2, 0, 1) / 255.0
        nxt = torch.from_numpy(self.frames[i + 1].copy()).float().permute(2, 0, 1) / 255.0
        action = torch.tensor(self.actions[i], dtype=torch.long)
        return prev, nxt, action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/vast/adi/discrete_wm/data/ablation_v1/atari_data.npz")
    parser.add_argument("--split-path", default="/vast/adi/discrete_wm/data/ablation_v1/train_test_split.json")
    parser.add_argument("--ckpt-path", default="/vast/adi/discrete_wm/checkpoints/ablation_v1/idm.pt")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = np.load(args.data)
    frames = data['frames']
    actions = data['actions']
    episode_ends = data['episode_ends']
    num_actions = int(data['num_actions'])

    with open(args.split_path) as f:
        split = json.load(f)

    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    def get_valid_indices(ep_list):
        indices = []
        for ep_idx in ep_list:
            start = episode_starts[ep_idx]
            end = episode_ends[ep_idx]
            for i in range(start, end - 1):
                indices.append(i)
        return indices

    train_indices = get_valid_indices(split['train_episodes'])
    test_indices = get_valid_indices(split['test_episodes'])

    train_ds = FramePairDataset(frames, actions, train_indices)
    test_ds = FramePairDataset(frames, actions, test_indices)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)

    model = IDM(num_actions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"IDM params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train: {len(train_ds)} pairs, Test: {len(test_ds)} pairs")

    model.train()
    data_iter = iter(train_dl)
    pbar = tqdm(range(args.steps), desc="Training IDM")

    for step in pbar:
        try:
            prev, nxt, act = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            prev, nxt, act = next(data_iter)

        prev, nxt, act = prev.to(device), nxt.to(device), act.to(device)
        logits = model(prev, nxt)
        loss = F.cross_entropy(logits, act)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 500 == 0:
            acc = (logits.argmax(dim=-1) == act).float().mean().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for prev, nxt, act in test_dl:
            prev, nxt, act = prev.to(device), nxt.to(device), act.to(device)
            logits = model(prev, nxt)
            preds = logits.argmax(dim=-1)
            correct += (preds == act).sum().item()
            total += len(act)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(act.cpu().tolist())

    test_acc = correct / total
    print(f"\nTest accuracy: {test_acc:.4f} ({correct}/{total})")

    from sklearn.metrics import classification_report
    print(classification_report(all_labels, all_preds,
                                target_names=['NOOP', 'FIRE', 'RIGHT', 'LEFT']))

    os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'num_actions': num_actions,
        'test_acc': test_acc,
    }, args.ckpt_path)
    print(f"Saved IDM to {args.ckpt_path}")


if __name__ == "__main__":
    main()
