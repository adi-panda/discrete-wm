"""
Evaluation script for the discrete diffusion world model with VQ-VAE tokenizer.

Computes: PSNR, SSIM, LPIPS, inference speed, action controllability.
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import imageio

from models.discrete_diffusion import DiscreteWorldModel, PatchVQVAE
from utils import AtariTokenizedDataset


def compute_psnr_batch(pred, target):
    """PSNR between uint8 image batches. pred, target: [B, H, W, C] float"""
    mse = ((pred.float() - target.float()) ** 2).mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(255.0 ** 2 / (mse + 1e-8))
    return psnr


def compute_ssim_batch(pred, target):
    """SSIM for images. pred, target: [B, C, H, W] float in [0, 255]."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    pred = pred.float()
    target = target.float()

    # Use 8x8 average pooling as approximation
    k = 8
    mu_p = F.avg_pool2d(pred, k, stride=k)
    mu_t = F.avg_pool2d(target, k, stride=k)

    sigma_p_sq = F.avg_pool2d(pred ** 2, k, stride=k) - mu_p ** 2
    sigma_t_sq = F.avg_pool2d(target ** 2, k, stride=k) - mu_t ** 2
    sigma_pt = F.avg_pool2d(pred * target, k, stride=k) - mu_p * mu_t

    ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
           ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p_sq + sigma_t_sq + C2))

    return ssim.mean(dim=(1, 2, 3))


def decode_tokens_to_frames(tokenizer, tokens, device):
    """Decode VQ-VAE tokens to uint8 frames [B, H, W, C]."""
    with torch.no_grad():
        frames = tokenizer.decode_tokens(tokens.to(device))
        frames_uint8 = ((frames.clamp(-1, 1) + 1) / 2 * 255).byte()
    return frames_uint8


@torch.no_grad()
def evaluate_quality(model, tokenizer, dataset, device, num_samples=1000,
                     gen_steps_list=[4, 8, 16, 32], temperature=0.9):
    """Evaluate prediction quality at different denoising step counts."""
    model.eval()
    results = {}

    for gen_steps in gen_steps_list:
        psnr_list = []
        ssim_list = []

        batch_size = 64
        indices = list(range(0, min(num_samples * 5, len(dataset)), 5))[:num_samples]

        for i in tqdm(range(0, len(indices), batch_size), desc=f"Eval (steps={gen_steps})"):
            batch_indices = indices[i:i + batch_size]
            prev_list, act_list, gt_list = [], [], []
            for idx in batch_indices:
                pt, a, nt = dataset[idx]
                prev_list.append(pt)
                act_list.append(a)
                gt_list.append(nt)

            prev_tokens = torch.stack(prev_list).to(device).long()
            actions = torch.stack(act_list).to(device)
            gt_tokens = torch.stack(gt_list).to(device).long()

            # Generate
            pred_tokens = model.generate(
                prev_tokens, actions, num_steps=gen_steps,
                temperature=temperature, device=device
            )

            # Decode to images
            pred_frames = decode_tokens_to_frames(tokenizer, pred_tokens, device)
            gt_frames = decode_tokens_to_frames(tokenizer, gt_tokens, device)

            # PSNR
            psnr = compute_psnr_batch(pred_frames, gt_frames)
            psnr_list.extend(psnr.cpu().tolist())

            # SSIM (BCHW format)
            pred_bchw = pred_frames.permute(0, 3, 1, 2).float()
            gt_bchw = gt_frames.permute(0, 3, 1, 2).float()
            ssim = compute_ssim_batch(pred_bchw, gt_bchw)
            ssim_list.extend(ssim.cpu().tolist())

        results[gen_steps] = {
            'psnr_mean': float(np.mean(psnr_list)),
            'psnr_std': float(np.std(psnr_list)),
            'ssim_mean': float(np.mean(ssim_list)),
            'ssim_std': float(np.std(ssim_list)),
        }

        print(f"  Steps={gen_steps}: PSNR={results[gen_steps]['psnr_mean']:.2f}±"
              f"{results[gen_steps]['psnr_std']:.2f}, "
              f"SSIM={results[gen_steps]['ssim_mean']:.4f}±"
              f"{results[gen_steps]['ssim_std']:.4f}")

    return results


@torch.no_grad()
def evaluate_lpips(model, tokenizer, dataset, device, num_samples=500,
                   gen_steps=8, temperature=0.9):
    """Evaluate LPIPS."""
    import lpips
    model.eval()
    loss_fn = lpips.LPIPS(net='alex').to(device)

    lpips_list = []
    batch_size = 32
    indices = list(range(0, min(num_samples * 5, len(dataset)), 5))[:num_samples]

    for i in tqdm(range(0, len(indices), batch_size), desc="LPIPS"):
        batch_indices = indices[i:i + batch_size]
        prev_list, act_list, gt_list = [], [], []
        for idx in batch_indices:
            pt, a, nt = dataset[idx]
            prev_list.append(pt)
            act_list.append(a)
            gt_list.append(nt)

        prev_tokens = torch.stack(prev_list).to(device).long()
        actions = torch.stack(act_list).to(device)
        gt_tokens = torch.stack(gt_list).to(device).long()

        pred_tokens = model.generate(
            prev_tokens, actions, num_steps=gen_steps,
            temperature=temperature, device=device
        )

        pred_frames = decode_tokens_to_frames(tokenizer, pred_tokens, device)
        gt_frames = decode_tokens_to_frames(tokenizer, gt_tokens, device)

        # LPIPS expects [-1, 1], BCHW
        pred_lpips = pred_frames.float().permute(0, 3, 1, 2) / 127.5 - 1.0
        gt_lpips = gt_frames.float().permute(0, 3, 1, 2) / 127.5 - 1.0

        lp = loss_fn(pred_lpips, gt_lpips)
        lpips_list.extend(lp.squeeze().cpu().tolist())

    result = {'lpips_mean': float(np.mean(lpips_list)), 'lpips_std': float(np.std(lpips_list))}
    print(f"LPIPS: {result['lpips_mean']:.4f}±{result['lpips_std']:.4f}")
    return result


@torch.no_grad()
def evaluate_speed(model, dataset, device, gen_steps_list=[4, 8, 16, 32]):
    """Measure inference speed."""
    model.eval()
    pt, a, nt = dataset[0]
    prev_tokens = pt.unsqueeze(0).to(device).long()
    actions = a.unsqueeze(0).to(device)

    results = {}
    for gen_steps in gen_steps_list:
        # Warmup
        for _ in range(5):
            model.generate(prev_tokens, actions, num_steps=gen_steps, device=device)

        torch.cuda.synchronize()
        start = time.time()
        n = 100
        for _ in range(n):
            model.generate(prev_tokens, actions, num_steps=gen_steps, device=device)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        fps = n / elapsed
        ms = elapsed / n * 1000
        results[gen_steps] = {'fps': fps, 'ms_per_frame': ms}
        print(f"  Steps={gen_steps}: {fps:.1f} FPS ({ms:.1f} ms/frame)")

    return results


@torch.no_grad()
def evaluate_long_horizon(model, tokenizer, dataset, device, num_rollouts=10,
                          rollout_length=50, gen_steps=8, temperature=0.9, save_dir=None):
    """Autoregressive rollout quality."""
    model.eval()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    all_psnr = [[] for _ in range(rollout_length)]

    for r in tqdm(range(num_rollouts), desc="Long-horizon"):
        start_idx = r * 100
        if start_idx + rollout_length >= len(dataset):
            start_idx = 0

        pt, _, _ = dataset[start_idx]
        current = pt.unsqueeze(0).to(device).long()

        pred_frames_list = []
        gt_frames_list = []

        for step in range(rollout_length):
            idx = min(start_idx + step, len(dataset) - 1)
            _, action, gt_next = dataset[idx]
            action = action.unsqueeze(0).to(device)

            pred = model.generate(current, action, num_steps=gen_steps,
                                   temperature=temperature, device=device)

            pred_frame = decode_tokens_to_frames(tokenizer, pred, device)
            gt_frame = decode_tokens_to_frames(tokenizer, gt_next.unsqueeze(0).to(device).long(), device)

            psnr = compute_psnr_batch(pred_frame, gt_frame).item()
            all_psnr[step].append(psnr)

            pred_frames_list.append(pred_frame[0].cpu().numpy())
            gt_frames_list.append(gt_frame[0].cpu().numpy())

            current = pred  # Autoregressive

        if save_dir and r < 3:
            imageio.mimsave(os.path.join(save_dir, f'rollout_{r:02d}.gif'),
                           pred_frames_list, fps=10)
            comparison = [np.concatenate([g, p], axis=1)
                         for g, p in zip(gt_frames_list, pred_frames_list)]
            imageio.mimsave(os.path.join(save_dir, f'rollout_{r:02d}_cmp.gif'),
                           comparison, fps=10)

    psnr_by_step = [float(np.mean(x)) if x else 0.0 for x in all_psnr]
    print(f"Long-horizon PSNR: step1={psnr_by_step[0]:.2f}, "
          f"step10={psnr_by_step[min(9,len(psnr_by_step)-1)]:.2f}, "
          f"step50={psnr_by_step[-1]:.2f}")
    return psnr_by_step


@torch.no_grad()
def evaluate_action_ctrl(model, tokenizer, dataset, device, num_samples=50,
                         gen_steps=8, save_dir=None):
    """Test action controllability."""
    model.eval()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    n_actions = model.action_embed[0].num_embeddings
    diffs = []

    for i in tqdm(range(num_samples), desc="Action ctrl"):
        pt, _, _ = dataset[i * 10 % len(dataset)]
        prev = pt.unsqueeze(0).to(device).long()

        frames = []
        for a in range(n_actions):
            action = torch.tensor([a], device=device)
            pred = model.generate(prev, action, num_steps=gen_steps, temperature=0.9, device=device)
            frame = decode_tokens_to_frames(tokenizer, pred, device)[0].float()
            frames.append(frame)

        for a1 in range(n_actions):
            for a2 in range(a1 + 1, n_actions):
                diff = (frames[a1] - frames[a2]).abs().mean().item()
                diffs.append(diff)

        if save_dir and i < 5:
            prev_frame = decode_tokens_to_frames(tokenizer, prev, device)[0].cpu().numpy()
            row = [prev_frame] + [f.cpu().numpy().astype(np.uint8) for f in frames]
            grid = np.concatenate(row, axis=1)
            imageio.imwrite(os.path.join(save_dir, f'action_ctrl_{i:02d}.png'), grid)

    mean_diff = float(np.mean(diffs))
    print(f"Action controllability: mean pairwise diff = {mean_diff:.2f}")
    return mean_diff


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
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

    # Load dataset
    dataset = AtariTokenizedDataset(args.data)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_args = ckpt.get('args', {})
    grid_size = int(math.sqrt(dataset.prev_tokens.shape[1]))

    model = DiscreteWorldModel(
        vocab_size=tok_args['vocab_size'],
        grid_h=grid_size,
        grid_w=grid_size,
        d_model=model_args.get('d_model', 512),
        n_layers=model_args.get('n_layers', 8),
        n_heads=model_args.get('n_heads', 8),
        n_actions=dataset.num_actions,
        cond_dim=model_args.get('d_model', 512),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params / 1e6:.1f}M params, step {ckpt.get('step', '?')}")

    results = {}

    # 1. Quality
    print("\n=== Quality Metrics ===")
    results['quality'] = evaluate_quality(
        model, tokenizer, dataset, device,
        num_samples=args.num_samples,
        gen_steps_list=[4, 8, 16, 32],
    )

    # 2. LPIPS
    print("\n=== LPIPS ===")
    results['lpips'] = evaluate_lpips(
        model, tokenizer, dataset, device,
        num_samples=min(args.num_samples, 500),
    )

    # 3. Speed
    print("\n=== Inference Speed ===")
    results['speed'] = evaluate_speed(model, dataset, device)

    # 4. Long-horizon
    print("\n=== Long-horizon ===")
    results['long_horizon_psnr'] = evaluate_long_horizon(
        model, tokenizer, dataset, device,
        num_rollouts=10, rollout_length=50,
        save_dir=os.path.join(args.output_dir, 'rollouts'),
    )

    # 5. Action controllability
    print("\n=== Action Controllability ===")
    results['action_ctrl'] = evaluate_action_ctrl(
        model, tokenizer, dataset, device,
        num_samples=50,
        save_dir=os.path.join(args.output_dir, 'action_ctrl'),
    )

    # Save
    import json
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to {args.output_dir}/eval_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-ckpt", default="/vast/adi/discrete_wm/checkpoints/tokenizer_final.pt")
    parser.add_argument("--data", default="/vast/adi/discrete_wm/data/tokenized_data.npz")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default="/vast/adi/discrete_wm/figures/eval")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
