# Ablation: Discrete Diffusion vs DIAMOND on Breakout

Experiment: **breakout-ablation-v1** (2026-04-20)

Head-to-head comparison of a masked discrete diffusion world model (ours) against DIAMOND's continuous diffusion denoiser, trained and evaluated on identical data.

## Ablation Table

| Metric | DIAMOND (continuous) | Ours (discrete diff) | Winner |
|---|---|---|---|
| Next-frame PSNR (dB) | **50.41** ± 6.38 | 49.10 ± 3.37 | DIAMOND (+1.3 dB) |
| Next-frame SSIM | 0.9957 ± 0.0127 | **0.9989** ± 0.0019 | Ours |
| Next-frame LPIPS ↓ | 0.0010 ± 0.0028 | **0.0003** ± 0.0006 | Ours (3.3× better) |
| FVD (16-step rollouts) ↓ | 0.91 | **0.41** | Ours (2.2× better) |
| IDM action F1 | **0.724** | 0.716 | Tie |
| Rollout PSNR @ step 1 | **51.09** | 48.22 | DIAMOND (+2.9 dB) |
| Rollout PSNR @ step 5 | 41.22 | **47.87** | Ours (+6.7 dB) |
| Rollout PSNR @ step 10 | 35.85 | **47.98** | Ours (+12.1 dB) |
| Rollout PSNR @ step 20 | 32.30 | **46.05** | Ours (+13.8 dB) |
| Rollout PSNR @ step 30 | 29.61 | **44.72** | Ours (+15.1 dB) |
| Rollout PSNR @ step 50 | 26.30 | **43.13** | Ours (+16.8 dB) |
| Inference FPS (best, H100 bs=1) | **34.0** (3 steps) | 34.2 (4 steps) | Tie |
| Params | **4.4M** | 48.5M | DIAMOND (11× smaller) |

## Per-metric Analysis

**Next-frame PSNR**: DIAMOND edges ahead by +1.3 dB on single-step prediction, though with much higher variance (σ=6.38 vs 3.37). Continuous diffusion's pixel-level MSE optimization pays off for one-step fidelity.

**Perceptual quality (SSIM/LPIPS)**: Our discrete model wins decisively — 3.3× better LPIPS and tighter SSIM. The VQ-VAE tokenizer enforces structural coherence that pixel-level noise removal does not.

**Long-horizon stability**: The standout result. By step 10, our model leads by +12.1 dB; by step 50, the gap is +16.8 dB. DIAMOND's continuous denoiser accumulates drift rapidly under autoregressive rollout. Our discrete token space acts as a natural error-correcting bottleneck — predictions snap to valid codebook entries, preventing the compounding continuous drift.

**FVD**: Our model produces 2.2× better video-level Fréchet distance over 16-frame rollouts, consistent with the long-horizon PSNR advantage.

**Action controllability (IDM F1)**: Roughly tied at 0.72/0.72. Both models respond similarly to action conditioning — the action signal is equally encoded through AdaLN (ours) and AdaGroupNorm (DIAMOND).

**Speed**: Nearly identical at best operating points (34 FPS each). Our model requires 4 discrete unmasking steps; DIAMOND requires 3 continuous denoising steps. The per-step cost differs (our Transformer forward pass vs their U-Net) but washes out.

**Efficiency**: DIAMOND is 11× smaller (4.4M vs 48.5M params). Our Transformer backbone with 1024-token cross-attention context is parameter-heavy. A convolutional VQ-VAE encoder and smaller Transformer could close this gap.

## Inference FPS Sweep

| Model | Steps | FPS | ms/frame |
|---|---|---|---|
| Ours | 4 | 34.2 | 29.3 |
| Ours | 8 | 19.5 | 51.2 |
| Ours | 16 | 10.0 | 100.1 |
| DIAMOND | 3 | 34.0 | 29.4 |
| DIAMOND | 5 | 25.1 | 39.9 |
| DIAMOND | 10 | 9.8 | 102.5 |

## Setup Parity Checklist

| Aspect | Status |
|---|---|
| Training data | Same 100k frames from DIAMOND's pretrained Breakout agent (ε=0.05) |
| Train/test split | Same 76/8 episode split (`train_test_split.json`) |
| Training steps | Both 100k optimizer steps |
| Context frames | Both 4 previous frames |
| Eval harness | Single script, identical test frames, identical action sequences |
| IDM | Same model (89.9% test acc) applied to both |
| Hardware | Single NVIDIA H100 80GB, same machine |

## Caveats

1. **DIAMOND trained from scratch on static data.** DIAMOND's published results use an iterative collect-train loop where the world model improves alongside the RL agent. Our static-dataset training likely underperforms their full pipeline. The comparison is fair for WM architecture but not for DIAMOND's full system.

2. **Single game.** Breakout is relatively simple. The long-horizon stability advantage may shrink or grow on more visually complex games (e.g., MsPacman, Montezuma's Revenge).

3. **Parameter asymmetry.** Our model is 11× larger. A smaller Transformer or convolutional architecture for the discrete diffusion model could change the efficiency story.

4. **FVD proxy.** FVD is computed using InceptionV3 frame features (not I3D video features). This is a per-frame Fréchet distance averaged over rollouts, not true FVD. Values should be compared relatively, not against published FVD benchmarks.

5. **100k steps may undertrain both models.** Our discrete WM loss was 1.25e-5 (well converged). DIAMOND's loss was 2.7e-5 (still dropping slightly). More training could narrow the next-frame gap.

6. **Action conditioning approximation.** For DIAMOND evaluation, we repeat the current action across all 4 conditioning slots (the harness does not track per-frame action history). This slightly disadvantages DIAMOND, which conditions on action sequences.

## Key Takeaway

Discrete diffusion over VQ-VAE tokens provides a **natural error-correcting mechanism** for autoregressive world model rollouts. While continuous diffusion (DIAMOND) produces slightly sharper single-step predictions, it suffers compounding drift under multi-step rollout. Our discrete model maintains 43 dB PSNR at 50 steps where DIAMOND drops to 26 dB — a difference visible to the naked eye. This stability advantage comes at the cost of 11× more parameters, suggesting the next step is architectural efficiency: smaller Transformers, convolutional tokenizers, or hybrid continuous-discrete approaches.

---

Full results: [`comparison.json`](/vast/adi/discrete_wm/figures/ablation_v1/comparison.json)
Rollout GIFs: [`figures/ablation_v1/rollouts_cmp/`](/vast/adi/discrete_wm/figures/ablation_v1/rollouts_cmp/)
