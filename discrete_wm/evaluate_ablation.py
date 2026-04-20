"""
Unified evaluation harness for breakout-ablation-v1.

Loads both world models (discrete diffusion + DIAMOND) and runs identical
metrics on identical test frames:
  - Next-frame PSNR, SSIM, LPIPS
  - FVD (16-step rollouts)
  - IDM action F1 (controllability)
  - Long-horizon rollout PSNR @ 1/5/10/20/30/50
  - Inference FPS at bs=1
"""

import argparse
import json
import os
import sys
import time
from abc import ABC, abstractmethod

import imageio
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'diamond', 'src'))


# ============================================================
# Metrics
# ============================================================

def compute_psnr(pred, gt):
    mse = ((pred.float() - gt.float()) ** 2).mean()
    if mse < 1e-10:
        return torch.tensor(100.0)
    return 10 * torch.log10(255.0 ** 2 / mse)


def compute_ssim(pred, gt, C1=6.5025, C2=58.5225):
    pred = pred.float()
    gt = gt.float()
    mu_p = F.avg_pool2d(pred, 8, 8)
    mu_g = F.avg_pool2d(gt, 8, 8)
    sigma_p = F.avg_pool2d(pred ** 2, 8, 8) - mu_p ** 2
    sigma_g = F.avg_pool2d(gt ** 2, 8, 8) - mu_g ** 2
    sigma_pg = F.avg_pool2d(pred * gt, 8, 8) - mu_p * mu_g
    ssim = ((2 * mu_p * mu_g + C1) * (2 * sigma_pg + C2)) / \
           ((mu_p ** 2 + mu_g ** 2 + C1) * (sigma_p + sigma_g + C2))
    return ssim.mean()


# ============================================================
# World Model Adapters
# ============================================================

class WMAdapter(ABC):
    @abstractmethod
    def predict_next_frame(self, prev_frames, action):
        """
        Args:
            prev_frames: [B, 4, 64, 64, 3] uint8 — 4 context frames
            action: [B] int64
        Returns:
            next_frame: [B, 64, 64, 3] uint8
        """
        pass

    @abstractmethod
    def name(self):
        pass


class DiscreteWMAdapter(WMAdapter):
    def __init__(self, model, tokenizer, device, gen_steps=8):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gen_steps = gen_steps
        self.model.eval()

    @property
    def name(self):
        return "Ours (discrete diff)"

    @torch.no_grad()
    def predict_next_frame(self, prev_frames, action):
        B, C, H, W, Ch = prev_frames.shape
        prev_tokens_list = []
        for c in range(C):
            frame = prev_frames[:, c].to(self.device)
            tokens, _, _, _ = self.tokenizer.encode(frame)
            prev_tokens_list.append(tokens)
        prev_tokens = torch.stack(prev_tokens_list, dim=1)  # [B, C, N]

        if self.model.context_frames == 1:
            prev_tokens = prev_tokens[:, -1]  # [B, N]

        action = action.to(self.device)
        pred_tokens = self.model.generate(
            prev_tokens, action, num_steps=self.gen_steps,
            temperature=0.9, device=self.device,
        )
        pred_frames = self.tokenizer.decode_tokens(pred_tokens)
        pred_uint8 = ((pred_frames.clamp(-1, 1) + 1) / 2 * 255).byte().cpu()
        return pred_uint8


class DiamondWMAdapter(WMAdapter):
    def __init__(self, denoiser, device, num_steps_denoising=3):
        self.denoiser = denoiser
        self.device = device
        self.num_steps_denoising = num_steps_denoising

        from models.diffusion import DiffusionSampler, DiffusionSamplerConfig
        cfg = DiffusionSamplerConfig(
            num_steps_denoising=num_steps_denoising,
            sigma_min=2e-3,
            sigma_max=5.0,
            rho=7,
            order=1,
        )
        self.sampler = DiffusionSampler(denoiser, cfg)

    @property
    def name(self):
        return "DIAMOND"

    @torch.no_grad()
    def predict_next_frame(self, prev_frames, action):
        B, C, H, W, Ch = prev_frames.shape
        obs = prev_frames.float().div(255).mul(2).sub(1)
        obs = obs.permute(0, 1, 4, 2, 3).to(self.device)  # [B, C, 3, 64, 64]

        act = action.to(self.device).unsqueeze(1).expand(-1, C)  # [B, C]

        pred, _ = self.sampler.sample(obs, act)
        pred_uint8 = pred.clamp(-1, 1).add(1).div(2).mul(255).byte()
        pred_uint8 = pred_uint8.permute(0, 2, 3, 1).cpu()  # [B, 64, 64, 3]
        return pred_uint8


# ============================================================
# Evaluation functions
# ============================================================

def eval_next_frame_quality(adapter, test_contexts, test_actions, test_gt, device, n_samples=1000):
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    psnr_list, ssim_list, lpips_list = [], [], []

    n = min(n_samples, len(test_actions))
    bs = 32
    for i in tqdm(range(0, n, bs), desc=f"Next-frame quality ({adapter.name})"):
        end = min(i + bs, n)
        ctx = test_contexts[i:end]
        act = test_actions[i:end]
        gt = test_gt[i:end]

        pred = adapter.predict_next_frame(ctx, act)

        pred_chw = pred.permute(0, 3, 1, 2).float().to(device)
        gt_chw = gt.permute(0, 3, 1, 2).float().to(device)

        for j in range(pred_chw.shape[0]):
            psnr_list.append(compute_psnr(pred_chw[j], gt_chw[j]).item())
            ssim_list.append(compute_ssim(pred_chw[j:j+1], gt_chw[j:j+1]).item())

        pred_norm = pred_chw / 127.5 - 1
        gt_norm = gt_chw / 127.5 - 1
        lp = lpips_fn(pred_norm, gt_norm)
        lpips_list.extend(lp.squeeze().cpu().tolist() if lp.dim() > 0 else [lp.item()])

    return {
        'psnr_mean': np.mean(psnr_list),
        'psnr_std': np.std(psnr_list),
        'ssim_mean': np.mean(ssim_list),
        'ssim_std': np.std(ssim_list),
        'lpips_mean': np.mean(lpips_list),
        'lpips_std': np.std(lpips_list),
    }


def eval_long_horizon(adapter, seed_contexts, seed_actions_seq, gt_frames_seq, device, max_steps=50):
    steps_to_report = [1, 5, 10, 20, 30, 50]
    psnr_at_step = {s: [] for s in steps_to_report}

    for seed_idx in tqdm(range(len(seed_contexts)), desc=f"Long-horizon ({adapter.name})"):
        ctx = seed_contexts[seed_idx:seed_idx+1]  # [1, 4, 64, 64, 3]
        gt_seq = gt_frames_seq[seed_idx]           # [T, 64, 64, 3]
        act_seq = seed_actions_seq[seed_idx]       # [T]

        T = min(max_steps, len(act_seq))
        context_list = [ctx[0, i] for i in range(ctx.shape[1])]

        for t in range(T):
            ctx_tensor = torch.stack(context_list[-4:], dim=0).unsqueeze(0)
            act_t = act_seq[t:t+1]
            pred = adapter.predict_next_frame(ctx_tensor, act_t)
            pred_frame = pred[0]

            step_num = t + 1
            if step_num in psnr_at_step:
                gt_frame = gt_seq[t]
                p = compute_psnr(
                    pred_frame.permute(2, 0, 1).float(),
                    gt_frame.permute(2, 0, 1).float(),
                ).item()
                psnr_at_step[step_num].append(p)

            context_list.append(pred_frame)

    return {f'psnr_step_{s}': np.mean(v) if v else 0.0 for s, v in psnr_at_step.items()}


def eval_idm_f1(adapter, idm, test_contexts, test_actions, device, n_samples=500):
    from train_idm import IDM
    idm.eval()

    n = min(n_samples, len(test_actions))
    all_commanded = []
    all_predicted = []

    for i in tqdm(range(n), desc=f"IDM F1 ({adapter.name})"):
        ctx = test_contexts[i:i+1]
        prev_frame = ctx[0, -1]  # [64, 64, 3]

        for a in range(4):
            act = torch.tensor([a], dtype=torch.long)
            pred = adapter.predict_next_frame(ctx, act)
            pred_frame = pred[0]

            prev_t = prev_frame.float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            pred_t = pred_frame.float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

            with torch.no_grad():
                logits = idm(prev_t, pred_t)
                pred_action = logits.argmax(dim=-1).item()

            all_commanded.append(a)
            all_predicted.append(pred_action)

    f1 = f1_score(all_commanded, all_predicted, average='macro')
    acc = np.mean([c == p for c, p in zip(all_commanded, all_predicted)])
    return {'idm_f1': f1, 'idm_acc': acc}


def eval_fps(adapter, test_contexts, device, n_warmup=10, n_measure=100):
    ctx = test_contexts[0:1]
    act = torch.tensor([1], dtype=torch.long)

    for _ in range(n_warmup):
        adapter.predict_next_frame(ctx, act)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_measure):
        adapter.predict_next_frame(ctx, act)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    ms_per_frame = elapsed / n_measure * 1000
    fps = n_measure / elapsed
    return {'fps': fps, 'ms_per_frame': ms_per_frame}


# ============================================================
# Data preparation
# ============================================================

def _build_frame_to_action_map(frames, actions, episode_ends):
    """Build a mapping from frame index to (action_value, is_valid)."""
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    frame_actions = np.full(len(frames), -1, dtype=np.int64)
    action_offset = 0
    for ep_idx in range(len(episode_ends)):
        start = episode_starts[ep_idx]
        end = episode_ends[ep_idx]
        n_actions = end - start - 1
        for j in range(n_actions):
            frame_actions[start + j] = actions[action_offset + j]
        action_offset += n_actions
    return frame_actions, episode_starts


def prepare_test_data(data_path, split_path, context_frames=4, max_samples=1000):
    data = np.load(data_path)
    frames = data['frames']
    actions = data['actions']
    episode_ends = data['episode_ends']

    with open(split_path) as f:
        split = json.load(f)

    frame_actions, episode_starts = _build_frame_to_action_map(frames, actions, episode_ends)

    contexts = []
    acts = []
    gts = []

    for ep_idx in split['test_episodes']:
        start = episode_starts[ep_idx]
        end = episode_ends[ep_idx]
        for i in range(start + context_frames, end - 1):
            if len(contexts) >= max_samples:
                break
            if frame_actions[i] < 0:
                continue
            ctx = []
            for c in range(context_frames - 1, -1, -1):
                src = max(start, i - c)
                ctx.append(torch.from_numpy(frames[src].copy()))
            contexts.append(torch.stack(ctx, dim=0))
            acts.append(torch.tensor(int(frame_actions[i]), dtype=torch.long))
            gts.append(torch.from_numpy(frames[i + 1].copy()))
        if len(contexts) >= max_samples:
            break

    return torch.stack(contexts), torch.stack(acts), torch.stack(gts)


def prepare_rollout_data(data_path, split_path, context_frames=4, num_seeds=10, horizon=50):
    data = np.load(data_path)
    frames = data['frames']
    actions = data['actions']
    episode_ends = data['episode_ends']

    with open(split_path) as f:
        split = json.load(f)

    frame_actions, episode_starts = _build_frame_to_action_map(frames, actions, episode_ends)

    seed_contexts = []
    seed_actions = []
    gt_frames_list = []

    for ep_idx in split['test_episodes']:
        if len(seed_contexts) >= num_seeds:
            break
        start = episode_starts[ep_idx]
        end = episode_ends[ep_idx]
        ep_len = end - start
        if ep_len < context_frames + horizon + 1:
            continue

        seed_idx = start + context_frames
        ctx = []
        for c in range(context_frames - 1, -1, -1):
            src = max(start, seed_idx - c)
            ctx.append(torch.from_numpy(frames[src].copy()))
        seed_contexts.append(torch.stack(ctx, dim=0))

        act_vals = []
        for j in range(horizon):
            a = frame_actions[seed_idx + j]
            act_vals.append(a if a >= 0 else 0)
        seed_actions.append(torch.tensor(act_vals, dtype=torch.long))

        gt_seq = torch.from_numpy(frames[seed_idx + 1:seed_idx + 1 + horizon].copy())
        gt_frames_list.append(gt_seq)

    return (
        torch.stack(seed_contexts),
        seed_actions,
        gt_frames_list,
    )


# ============================================================
# Model loading
# ============================================================

def load_discrete_wm(ckpt_path, tokenizer_path, device):
    from models.discrete_diffusion import DiscreteWorldModel, PatchVQVAE

    tok_ckpt = torch.load(tokenizer_path, map_location=device, weights_only=False)
    tok_args = tok_ckpt['args']
    tokenizer = PatchVQVAE(
        patch_size=tok_args['patch_size'],
        vocab_size=tok_args['vocab_size'],
        embed_dim=tok_args['embed_dim'],
    ).to(device)
    tokenizer.load_state_dict(tok_ckpt['model'])
    tokenizer.eval()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = ckpt.get('args', {})
    context_frames = model_args.get('context_frames', 1)
    import math
    grid_size = int(math.sqrt(tok_args.get('num_patches', 256)))
    if grid_size == 0:
        grid_size = 16

    model = DiscreteWorldModel(
        vocab_size=tok_args['vocab_size'],
        grid_h=grid_size,
        grid_w=grid_size,
        d_model=model_args.get('d_model', 512),
        n_layers=model_args.get('n_layers', 8),
        n_heads=model_args.get('n_heads', 8),
        n_actions=model_args.get('n_actions', ckpt.get('tokenizer_args', {}).get('num_actions', 4)),
        cond_dim=model_args.get('d_model', 512),
        context_frames=context_frames,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    return DiscreteWMAdapter(model, tokenizer, device)


def load_diamond_wm(ckpt_path, device, num_actions=4):
    from models.diffusion import Denoiser, DenoiserConfig
    from models.diffusion.inner_model import InnerModelConfig

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
    denoiser = Denoiser(denoiser_cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    from utils import extract_state_dict
    if isinstance(ckpt, dict) and any(k.startswith('denoiser.') for k in ckpt.keys()):
        den_sd = extract_state_dict(ckpt, 'denoiser')
        denoiser.load_state_dict(den_sd)
    else:
        denoiser.load_state_dict(ckpt)
    denoiser.eval()

    return DiamondWMAdapter(denoiser, device)


def load_idm(ckpt_path, device):
    from train_idm import IDM
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_actions = ckpt.get('num_actions', 4)
    model = IDM(num_actions).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/vast/adi/discrete_wm/data/ablation_v1/atari_data.npz")
    parser.add_argument("--split-path", default="/vast/adi/discrete_wm/data/ablation_v1/train_test_split.json")
    parser.add_argument("--discrete-wm-ckpt", default="/vast/adi/discrete_wm/checkpoints/ablation_v1/model_best.pt")
    parser.add_argument("--tokenizer-ckpt", default="/vast/adi/discrete_wm/checkpoints/ablation_v1/tokenizer_final.pt")
    parser.add_argument("--diamond-ckpt", default=None)
    parser.add_argument("--idm-ckpt", default="/vast/adi/discrete_wm/checkpoints/ablation_v1/idm.pt")
    parser.add_argument("--output-dir", default="/vast/adi/discrete_wm/figures/ablation_v1")
    parser.add_argument("--output-json", default="/vast/adi/discrete_wm/figures/ablation_v1/comparison.json")
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--n-quality-samples", type=int, default=1000)
    parser.add_argument("--n-rollout-seeds", type=int, default=10)
    parser.add_argument("--n-idm-samples", type=int, default=200)
    parser.add_argument("--skip-diamond", action="store_true")
    parser.add_argument("--skip-discrete", action="store_true")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    print("=" * 60)
    print("Loading test data...")
    test_ctx, test_act, test_gt = prepare_test_data(
        args.data, args.split_path, args.context_frames, args.n_quality_samples
    )
    print(f"Test data: {len(test_act)} samples, context: {test_ctx.shape}")

    rollout_ctx, rollout_acts, rollout_gts = prepare_rollout_data(
        args.data, args.split_path, args.context_frames, args.n_rollout_seeds
    )
    print(f"Rollout data: {len(rollout_acts)} seeds")

    idm = None
    if os.path.exists(args.idm_ckpt):
        idm = load_idm(args.idm_ckpt, device)
        print("IDM loaded")

    adapters = []

    if not args.skip_discrete and os.path.exists(args.discrete_wm_ckpt):
        print("Loading discrete WM...")
        adapter = load_discrete_wm(args.discrete_wm_ckpt, args.tokenizer_ckpt, device)
        adapters.append(adapter)

    if not args.skip_diamond and args.diamond_ckpt and os.path.exists(args.diamond_ckpt):
        print("Loading DIAMOND WM...")
        adapter = load_diamond_wm(args.diamond_ckpt, device)
        adapters.append(adapter)

    for adapter in adapters:
        name = adapter.name
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {name}")
        print(f"{'=' * 60}")

        r = {}

        print("\n--- Next-frame quality ---")
        q = eval_next_frame_quality(adapter, test_ctx, test_act, test_gt, device, args.n_quality_samples)
        r.update(q)
        print(f"PSNR: {q['psnr_mean']:.2f} ± {q['psnr_std']:.2f}")
        print(f"SSIM: {q['ssim_mean']:.4f} ± {q['ssim_std']:.4f}")
        print(f"LPIPS: {q['lpips_mean']:.4f} ± {q['lpips_std']:.4f}")

        print("\n--- Long-horizon rollout ---")
        lh = eval_long_horizon(adapter, rollout_ctx, rollout_acts, rollout_gts, device)
        r.update(lh)
        for k, v in sorted(lh.items()):
            print(f"{k}: {v:.2f}")

        if idm is not None:
            print("\n--- IDM action F1 ---")
            idm_r = eval_idm_f1(adapter, idm, test_ctx, test_act, device, args.n_idm_samples)
            r.update(idm_r)
            print(f"F1: {idm_r['idm_f1']:.4f}, Acc: {idm_r['idm_acc']:.4f}")

        print("\n--- Inference FPS ---")
        fps = eval_fps(adapter, test_ctx, device)
        r.update(fps)
        print(f"FPS: {fps['fps']:.1f}, ms/frame: {fps['ms_per_frame']:.1f}")

        results[name] = r

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")

    # Save rollout GIFs
    if len(adapters) > 0 and len(rollout_acts) > 0:
        gif_dir = os.path.join(args.output_dir, 'rollouts_cmp')
        os.makedirs(gif_dir, exist_ok=True)

        for seed_idx in range(min(3, len(rollout_acts))):
            ctx = rollout_ctx[seed_idx:seed_idx+1]
            gt_seq = rollout_gts[seed_idx]
            act_seq = rollout_acts[seed_idx]
            T = min(16, len(act_seq))

            all_model_frames = {}
            for adapter in adapters:
                model_frames = []
                context_list = [ctx[0, i] for i in range(ctx.shape[1])]
                for t in range(T):
                    ctx_t = torch.stack(context_list[-4:], dim=0).unsqueeze(0)
                    act_t = act_seq[t:t+1]
                    pred = adapter.predict_next_frame(ctx_t, act_t)
                    model_frames.append(pred[0])
                    context_list.append(pred[0])
                all_model_frames[adapter.name] = model_frames

            gif_frames = []
            for t in range(T):
                row = [gt_seq[t].numpy()]
                for adapter in adapters:
                    row.append(all_model_frames[adapter.name][t].numpy())
                gif_frames.append(np.concatenate(row, axis=1))

            gif_path = os.path.join(gif_dir, f'rollout_seed_{seed_idx}.gif')
            imageio.mimsave(gif_path, gif_frames, fps=5, loop=0)
            print(f"Saved rollout GIF: {gif_path}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
