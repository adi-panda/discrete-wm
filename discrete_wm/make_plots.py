"""Generate analysis plots for the discrete diffusion world model."""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(csv_path, save_dir):
    """Plot training loss curve."""
    steps, losses, lrs = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            losses.append(float(row['loss']))
            lrs.append(float(row['lr']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(steps, losses, 'b-', linewidth=0.8)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Training Loss (Discrete Diffusion World Model)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # LR curve
    ax2.plot(steps, lrs, 'r-', linewidth=0.8)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training_loss.png")


def plot_quality_vs_steps(eval_results, save_dir):
    """Plot PSNR/SSIM vs denoising steps."""
    quality = eval_results.get('quality', {})
    if not quality:
        return

    gen_steps = sorted([int(k) for k in quality.keys()])
    psnr_means = [quality[str(s)]['psnr_mean'] for s in gen_steps]
    psnr_stds = [quality[str(s)]['psnr_std'] for s in gen_steps]
    ssim_means = [quality[str(s)]['ssim_mean'] for s in gen_steps]
    ssim_stds = [quality[str(s)]['ssim_std'] for s in gen_steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.errorbar(gen_steps, psnr_means, yerr=psnr_stds, marker='o', capsize=5,
                 linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Denoising Steps')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR vs Denoising Steps')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(gen_steps)
    ax1.set_xticklabels([str(s) for s in gen_steps])
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(gen_steps, ssim_means, yerr=ssim_stds, marker='s', capsize=5,
                 linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Denoising Steps')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM vs Denoising Steps')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(gen_steps)
    ax2.set_xticklabels([str(s) for s in gen_steps])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'quality_vs_steps.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved quality_vs_steps.png")


def plot_speed_vs_steps(eval_results, save_dir):
    """Plot FPS vs denoising steps."""
    speed = eval_results.get('speed', {})
    if not speed:
        return

    gen_steps = sorted([int(k) for k in speed.keys()])
    fps = [speed[str(s)]['fps'] for s in gen_steps]
    ms = [speed[str(s)]['ms_per_frame'] for s in gen_steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar([str(s) for s in gen_steps], fps, color='coral', edgecolor='black')
    ax1.set_xlabel('Denoising Steps')
    ax1.set_ylabel('Frames per Second')
    ax1.set_title('Inference Speed vs Denoising Steps')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar([str(s) for s in gen_steps], ms, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Denoising Steps')
    ax2.set_ylabel('ms per Frame')
    ax2.set_title('Latency vs Denoising Steps')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'speed_vs_steps.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved speed_vs_steps.png")


def plot_long_horizon(eval_results, save_dir):
    """Plot PSNR degradation over autoregressive steps."""
    psnr_by_step = eval_results.get('long_horizon_psnr', [])
    if not psnr_by_step:
        return

    steps = list(range(1, len(psnr_by_step) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, psnr_by_step, 'b-o', linewidth=1.5, markersize=4)
    ax.set_xlabel('Autoregressive Step')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Long-Horizon Quality Degradation')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=psnr_by_step[0], color='gray', linestyle='--', alpha=0.5,
               label=f'Step 1: {psnr_by_step[0]:.1f} dB')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'long_horizon.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved long_horizon.png")


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # Plot training loss if CSV exists
    if os.path.exists(args.csv_path):
        plot_training_loss(args.csv_path, args.save_dir)

    # Plot eval results if JSON exists
    if os.path.exists(args.eval_path):
        with open(args.eval_path) as f:
            eval_results = json.load(f)

        plot_quality_vs_steps(eval_results, args.save_dir)
        plot_speed_vs_steps(eval_results, args.save_dir)
        plot_long_horizon(eval_results, args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="/vast/adi/discrete_wm/logs/training_log.csv")
    parser.add_argument("--eval-path", default="/vast/adi/discrete_wm/figures/eval/eval_results.json")
    parser.add_argument("--save-dir", default="/vast/adi/discrete_wm/figures/plots")
    args = parser.parse_args()
    main(args)
