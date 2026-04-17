"""Collect Atari gameplay data for training the discrete diffusion world model."""

import argparse
import os
import numpy as np
import gymnasium as gym
import ale_py
import cv2
from tqdm import tqdm

# Register ALE environments
gym.register_envs(ale_py)


def collect_data(env_name="ALE/Breakout-v5", num_frames=100000, save_dir="data", seed=42):
    """Collect (prev_frame, action, next_frame) triples from random policy."""
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make(env_name, render_mode="rgb_array", frameskip=1)
    env.action_space.seed(seed)

    frames = []
    actions = []

    obs, info = env.reset(seed=seed)
    # Resize to 64x64 RGB
    obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
    frames.append(obs)

    frame_count = 0
    episodes = 0
    pbar = tqdm(total=num_frames, desc="Collecting frames")

    while frame_count < num_frames:
        # Frame skip of 4 (standard Atari)
        action = env.action_space.sample()

        total_reward = 0
        last_two = []
        for skip in range(4):
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            last_two.append(next_obs)
            if len(last_two) > 2:
                last_two.pop(0)
            if terminated or truncated:
                break

        # Max pool over last 2 frames (standard Atari preprocessing)
        if len(last_two) == 2:
            next_obs = np.maximum(last_two[0], last_two[1])
        else:
            next_obs = last_two[-1]

        next_obs = cv2.resize(next_obs, (64, 64), interpolation=cv2.INTER_AREA)

        frames.append(next_obs)
        actions.append(action)
        frame_count += 1
        pbar.update(1)

        if terminated or truncated:
            obs, info = env.reset()
            obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
            frames.append(obs)
            # Add a dummy action for the reset boundary (won't be used as prev_frame)
            actions.append(0)
            frame_count += 1
            pbar.update(1)
            episodes += 1

    pbar.close()
    env.close()

    frames = np.array(frames, dtype=np.uint8)
    actions = np.array(actions, dtype=np.int64)

    print(f"Collected {len(frames)} frames across {episodes} episodes")
    print(f"Action space: {env.action_space.n} actions")
    print(f"Frame shape: {frames[0].shape}")

    # Save
    np.savez_compressed(
        os.path.join(save_dir, f"atari_data.npz"),
        frames=frames,
        actions=actions,
        num_actions=env.action_space.n,
    )
    print(f"Saved to {save_dir}/atari_data.npz")

    # Save some sample frames
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    for i in range(min(20, len(frames))):
        cv2.imwrite(os.path.join(sample_dir, f"frame_{i:04d}.png"),
                    cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
    print(f"Saved {min(20, len(frames))} sample frames to {sample_dir}/")

    return frames, actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ALE/Breakout-v5")
    parser.add_argument("--num-frames", type=int, default=100000)
    parser.add_argument("--save-dir", default="/vast/adi/discrete_wm/data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    collect_data(args.env, args.num_frames, args.save_dir, args.seed)
