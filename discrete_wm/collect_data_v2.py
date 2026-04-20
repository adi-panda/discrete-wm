"""
Collect Atari Breakout gameplay data using DIAMOND's pretrained agent.

Saves in dual format:
1. Our format: atari_data.npz with {frames, actions, episode_ends, num_actions}
2. DIAMOND format: .pt files per episode under diamond_dataset/{train,test}/

Also saves train_test_split.json with episode indices.
"""

import argparse
import json
import os
import sys

import cv2
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm import tqdm

gym.register_envs(ale_py)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'diamond', 'src'))
from models.actor_critic import ActorCritic, ActorCriticConfig
from envs.atari_preprocessing import AtariPreprocessing
from utils import extract_state_dict


def build_actor_critic(num_actions=4):
    cfg = ActorCriticConfig(
        lstm_dim=512,
        img_channels=3,
        img_size=64,
        channels=[32, 32, 64, 64],
        down=[1, 1, 1, 1],
        num_actions=num_actions,
    )
    return ActorCritic(cfg)


def load_diamond_agent(device):
    path = hf_hub_download(repo_id="eloialonso/diamond", filename="atari_100k/models/Breakout.pt")
    full_sd = torch.load(path, map_location=device, weights_only=False)
    ac_sd = extract_state_dict(full_sd, "actor_critic")
    ac = build_actor_critic(num_actions=4).to(device)
    ac.load_state_dict(ac_sd)
    ac.eval()
    return ac


def make_env():
    env = gym.make("BreakoutNoFrameskip-v4", frameskip=1, render_mode="rgb_array")
    env = AtariPreprocessing(env=env, noop_max=30, frame_skip=4, screen_size=64)
    return env


def obs_to_tensor(obs, device):
    t = torch.from_numpy(obs).float().div(255).mul(2).sub(1)
    t = t.permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def collect(num_target_transitions=100000, epsilon=0.05, seed=42, device='cuda'):
    env = make_env()
    ac = load_diamond_agent(device)

    all_episodes = []
    current_obs_list = []
    current_act_list = []
    current_rew_list = []
    current_end_list = []
    current_trunc_list = []

    total_transitions = 0
    episode_count = 0

    obs, info = env.reset(seed=seed)
    current_obs_list.append(obs.copy())

    hx = torch.zeros(1, 512, device=device)
    cx = torch.zeros(1, 512, device=device)

    pbar = tqdm(total=num_target_transitions, desc="Collecting transitions")

    while total_transitions < num_target_transitions:
        obs_t = obs_to_tensor(obs, device)

        with torch.no_grad():
            output = ac.predict_act_value(obs_t, (hx, cx))
            logits = output.logits_act
            hx, cx = output.hx_cx

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = logits.argmax(dim=-1).item()

        next_obs, reward, terminated, truncated, info = env.step(action)

        current_act_list.append(action)
        current_rew_list.append(reward)
        current_end_list.append(terminated)
        current_trunc_list.append(truncated)
        current_obs_list.append(next_obs.copy())

        total_transitions += 1
        pbar.update(1)

        obs = next_obs

        if terminated or truncated:
            episode = {
                'obs': np.array(current_obs_list, dtype=np.uint8),
                'act': np.array(current_act_list, dtype=np.int64),
                'rew': np.array(current_rew_list, dtype=np.float32),
                'end': np.array(current_end_list, dtype=np.uint8),
                'trunc': np.array(current_trunc_list, dtype=np.uint8),
            }
            all_episodes.append(episode)
            episode_count += 1

            current_obs_list = []
            current_act_list = []
            current_rew_list = []
            current_end_list = []
            current_trunc_list = []

            obs, info = env.reset()
            current_obs_list.append(obs.copy())
            hx = torch.zeros(1, 512, device=device)
            cx = torch.zeros(1, 512, device=device)

    if len(current_act_list) > 0:
        episode = {
            'obs': np.array(current_obs_list, dtype=np.uint8),
            'act': np.array(current_act_list, dtype=np.int64),
            'rew': np.array(current_rew_list, dtype=np.float32),
            'end': np.array(current_end_list, dtype=np.uint8),
            'trunc': np.array(current_trunc_list, dtype=np.uint8),
        }
        all_episodes.append(episode)
        episode_count += 1

    pbar.close()
    env.close()

    print(f"\nCollected {total_transitions} transitions across {episode_count} episodes")
    return all_episodes


def save_our_format(episodes, save_path):
    all_frames = []
    all_actions = []
    episode_ends = []

    frame_idx = 0
    for ep in episodes:
        all_frames.append(ep['obs'])
        all_actions.append(ep['act'])
        frame_idx += len(ep['obs'])
        episode_ends.append(frame_idx)

    frames = np.concatenate(all_frames, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    episode_ends = np.array(episode_ends, dtype=np.int64)

    np.savez_compressed(
        save_path,
        frames=frames,
        actions=actions,
        episode_ends=episode_ends,
        num_actions=4,
    )
    print(f"Saved our format: {save_path}")
    print(f"  Frames: {frames.shape}, Actions: {actions.shape}, Episodes: {len(episode_ends)}")
    return frames, actions, episode_ends


def save_diamond_format(episodes, train_indices, test_indices, base_dir):
    for split_name, indices in [('train', train_indices), ('test', test_indices)]:
        split_dir = os.path.join(base_dir, split_name)
        for i, ep_idx in enumerate(indices):
            ep = episodes[ep_idx]
            obs_float = torch.from_numpy(ep['obs']).float().div(255).mul(2).sub(1)
            obs_float = obs_float.permute(0, 3, 1, 2)
            act = torch.from_numpy(ep['act']).long()
            rew = torch.from_numpy(ep['rew']).float()
            end = torch.from_numpy(ep['end'])
            trunc = torch.from_numpy(ep['trunc'])

            ep_dict = {
                'obs': obs_float.add(1).div(2).mul(255).byte(),
                'act': act,
                'rew': rew,
                'end': end,
                'trunc': trunc,
                'info': {},
            }

            n = 3
            powers = np.arange(n)
            subfolders = np.floor((i % 10 ** (1 + powers)) / 10**powers) * 10**powers
            subfolders = [int(x) for x in subfolders[::-1]]
            subfolders = "/".join([f"{x:0{n - j}d}" for j, x in enumerate(subfolders)])
            ep_path = os.path.join(split_dir, subfolders, f"{i}.pt")
            os.makedirs(os.path.dirname(ep_path), exist_ok=True)
            torch.save(ep_dict, ep_path)

    print(f"Saved DIAMOND format: {len(train_indices)} train, {len(test_indices)} test episodes")


def save_diamond_info(episodes, indices, directory):
    num_episodes = len(indices)
    lengths = []
    counter_rew = {}
    counter_end = {}

    for i, ep_idx in enumerate(indices):
        ep = episodes[ep_idx]
        ep_len = len(ep['obs'])
        lengths.append(ep_len)
        for r in ep['rew']:
            r_sign = int(np.sign(r))
            counter_rew[r_sign] = counter_rew.get(r_sign, 0) + 1
        for e in ep['end']:
            counter_end[int(e)] = counter_end.get(int(e), 0) + 1

    num_steps = sum(lengths)
    start_idx = np.cumsum([0] + lengths[:-1])

    state_dict = {
        'is_static': True,
        'num_episodes': num_episodes,
        'num_steps': num_steps,
        'start_idx': np.array(start_idx, dtype=np.int64),
        'lengths': np.array(lengths, dtype=np.int64),
        'counter_rew': dict(counter_rew),
        'counter_end': dict(counter_end),
    }
    info_path = os.path.join(directory, 'info.pt')
    torch.save(state_dict, info_path)
    print(f"Saved DIAMOND info.pt: {directory} ({num_episodes} eps, {num_steps} steps)")


def make_sanity_gifs(episodes, save_dir, num_episodes=5):
    import imageio
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_episodes, len(episodes))):
        ep = episodes[i]
        frames = ep['obs']
        gif_path = os.path.join(save_dir, f'episode_{i:03d}.gif')
        imageio.mimsave(gif_path, frames, fps=15, loop=0)
    print(f"Saved {min(num_episodes, len(episodes))} sanity GIFs to {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-transitions", type=int, default=100000)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="/vast/adi/discrete_wm/data/ablation_v1")
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Phase 1: Data Collection with DIAMOND Agent")
    print("=" * 60)

    episodes = collect(
        num_target_transitions=args.num_transitions,
        epsilon=args.epsilon,
        seed=args.seed,
        device=device,
    )

    num_episodes = len(episodes)
    np.random.seed(args.seed)
    perm = np.random.permutation(num_episodes)
    n_test = max(1, int(num_episodes * args.test_fraction))
    test_indices = sorted(perm[:n_test].tolist())
    train_indices = sorted(perm[n_test:].tolist())

    split = {
        'train_episodes': train_indices,
        'test_episodes': test_indices,
        'num_total_episodes': num_episodes,
        'seed': args.seed,
    }
    split_path = os.path.join(args.save_dir, 'train_test_split.json')
    with open(split_path, 'w') as f:
        json.dump(split, f, indent=2)
    print(f"\nSaved train/test split: {len(train_indices)} train, {len(test_indices)} test -> {split_path}")

    frames, actions, episode_ends = save_our_format(
        episodes, os.path.join(args.save_dir, 'atari_data.npz')
    )

    diamond_dir = os.path.join(args.save_dir, 'diamond_dataset')
    save_diamond_format(episodes, train_indices, test_indices, diamond_dir)
    save_diamond_info(episodes, train_indices, os.path.join(diamond_dir, 'train'))
    save_diamond_info(episodes, test_indices, os.path.join(diamond_dir, 'test'))

    action_hist = np.bincount(actions, minlength=4)
    print(f"\nAction histogram: NOOP={action_hist[0]}, FIRE={action_hist[1]}, RIGHT={action_hist[2]}, LEFT={action_hist[3]}")

    ep_lengths = [len(ep['act']) for ep in episodes]
    ep_rewards = [ep['rew'].sum() for ep in episodes]
    print(f"Episode lengths: mean={np.mean(ep_lengths):.1f}, min={min(ep_lengths)}, max={max(ep_lengths)}")
    print(f"Episode rewards: mean={np.mean(ep_rewards):.1f}, min={min(ep_rewards):.0f}, max={max(ep_rewards):.0f}")

    if len(frames) > 1:
        diffs = np.abs(frames[1:].astype(float) - frames[:-1].astype(float)).mean()
        print(f"Mean consecutive frame diff: {diffs:.2f} (on [0,255] scale)")

    sanity_dir = os.path.join(os.path.dirname(args.save_dir), '..', 'figures', 'ablation_v1', 'collection_sanity')
    sanity_dir = '/vast/adi/discrete_wm/figures/ablation_v1/collection_sanity'
    make_sanity_gifs(episodes, sanity_dir, num_episodes=5)

    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
