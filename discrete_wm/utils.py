"""Utility functions for the discrete diffusion world model."""

import math
import torch
import numpy as np
from torch.utils.data import Dataset


class AtariFramePairDataset(Dataset):
    """Dataset of (prev_frame, action, next_frame) triples from collected Atari data."""

    def __init__(self, data_path):
        data = np.load(data_path)
        self.frames = data['frames']  # [N, 64, 64, 3] uint8
        self.actions = data['actions']  # [N-1] int64
        self.num_actions = int(data['num_actions'])

        # Build valid indices for consecutive frame pairs
        self.valid_indices = list(range(len(self.actions)))
        print(f"Dataset: {len(self.valid_indices)} valid frame pairs, "
              f"{self.num_actions} actions")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        prev_frame = torch.from_numpy(self.frames[i].copy())      # [64, 64, 3] uint8
        next_frame = torch.from_numpy(self.frames[i + 1].copy())  # [64, 64, 3] uint8
        action = torch.tensor(self.actions[i], dtype=torch.long)
        return prev_frame, action, next_frame


class AtariTokenizedDataset(Dataset):
    """Pre-tokenized dataset for faster training."""

    def __init__(self, tokens_path):
        data = np.load(tokens_path)
        self.prev_tokens = torch.from_numpy(data['prev_tokens'].astype(np.int64))  # [N, num_patches]
        self.next_tokens = torch.from_numpy(data['next_tokens'].astype(np.int64))  # [N, num_patches]
        self.actions = torch.from_numpy(data['actions'].astype(np.int64))           # [N]
        self.num_actions = int(data['num_actions'])
        print(f"Tokenized dataset: {len(self)} pairs, {self.num_actions} actions, "
              f"{self.prev_tokens.shape[1]} tokens/frame")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.prev_tokens[idx], self.actions[idx], self.next_tokens[idx]


def cosine_mask_schedule(t):
    """Cosine masking schedule. t in [0, 1] -> mask ratio in [0, 1]."""
    return torch.cos(t * math.pi / 2)
