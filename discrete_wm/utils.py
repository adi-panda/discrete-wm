"""Utility functions for the discrete diffusion world model."""

import json
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


class AtariMultiFrameTokenizedDataset(Dataset):
    """Tokenized dataset returning C previous frames as context.

    Handles episode boundaries: pads missing frames by repeating earliest available.
    """

    def __init__(self, tokens_path, split_path=None, split='train', context_frames=4):
        data = np.load(tokens_path)
        all_tokens = torch.from_numpy(data['all_tokens'].astype(np.int64))  # [N_frames, num_patches]
        all_actions = torch.from_numpy(data['actions'].astype(np.int64))    # [N_transitions]
        episode_ends = data['episode_ends'].astype(np.int64)                # [E] frame indices
        self.num_actions = int(data['num_actions'])
        self.context_frames = context_frames

        if split_path is not None:
            with open(split_path) as f:
                split_info = json.load(f)
            ep_indices = split_info[f'{split}_episodes']
        else:
            ep_indices = list(range(len(episode_ends)))

        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        self.valid_indices = []
        for ep_idx in ep_indices:
            start = episode_starts[ep_idx]
            end = episode_ends[ep_idx]
            for frame_idx in range(start, end - 1):
                self.valid_indices.append(frame_idx)

        self.all_tokens = all_tokens
        self.all_actions = all_actions
        self.episode_starts = episode_starts
        self.episode_ends = episode_ends

        ep_membership = np.zeros(len(all_tokens), dtype=np.int64)
        for i, (s, e) in enumerate(zip(episode_starts, episode_ends)):
            ep_membership[s:e] = i
        self.ep_membership = ep_membership

        print(f"MultiFrame tokenized dataset ({split}): {len(self)} pairs, "
              f"{self.num_actions} actions, {context_frames} context frames, "
              f"{all_tokens.shape[1]} tokens/frame")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        frame_idx = self.valid_indices[idx]
        ep_idx = self.ep_membership[frame_idx]
        ep_start = self.episode_starts[ep_idx]

        prev_frames = []
        for c in range(self.context_frames - 1, -1, -1):
            src_idx = frame_idx - c
            if src_idx < ep_start:
                src_idx = ep_start
            prev_frames.append(self.all_tokens[src_idx])

        prev_tokens = torch.stack(prev_frames, dim=0)  # [C, N]
        next_tokens = self.all_tokens[frame_idx + 1]    # [N]
        action = self.all_actions[frame_idx]

        return prev_tokens, action, next_tokens


def cosine_mask_schedule(t):
    """Cosine masking schedule. t in [0, 1] -> mask ratio in [0, 1]."""
    return torch.cos(t * math.pi / 2)
