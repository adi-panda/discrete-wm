# Discrete Diffusion World Model for Atari

Masked discrete diffusion (MDLM-style) world model for Atari frames, compared against DIAMOND (continuous diffusion baseline). Currently trained on Breakout — experiment name `breakout-v1`.

Full quantitative results: [`results.md`](results.md). Phase-by-phase log: [`research_log.md`](research_log.md). **Ablation vs DIAMOND**: [`comparison.md`](comparison.md) — head-to-head on identical data with identical eval harness.

---

## Approach

Replace DIAMOND's continuous EDM diffusion over pixels with **masked discrete diffusion** over VQ-VAE tokens. Two stages:

1. **PatchVQ-VAE tokenizer** — 4×4×3 patch MLP encoder/decoder, 512-code × 256-dim codebook, 16×16=256 tokens per 64×64 frame, dead-code reset. 0.4M params.
2. **Discrete diffusion transformer** — bidirectional Transformer, 8 layers × 512 dim × 8 heads, 48.1M params. Cosine masking schedule, CE on masked positions only. Action + timestep conditioning via AdaLN (DiT-style); previous frame via cross-attention every layer. Inference is iterative confidence-based unmasking in T steps.

## Dataset

- **100,001 Breakout frames**, random policy via gymnasium `ALE/Breakout-v5`
- Standard Atari preproc: frame skip 4, max-pool last 2, resize to 64×64 RGB
- 4 actions (NOOP, FIRE, RIGHT, LEFT), roughly uniform
- Raw: `/vast/adi/discrete_wm/data/atari_data.npz` · Pre-tokenized: `tokenized_data.npz`

## Training

200k steps, bs=128, lr=1e-4 cosine + 1k warmup, AdamW (β=0.9/0.95, wd=0.01), grad-clip 1.0, bf16. ~5h 44min on a single H100 80GB. Loss 3.96 → 0.092 (500) → 0.003 (20k) → **0.0010** (200k). Effectively converged by 100k; the 100k→200k window mostly improved long-horizon stability.

## Headline results (1000 test samples, @ 200k)

| Denoising steps | PSNR (dB) | SSIM | LPIPS | FPS (H100) |
|---|---|---|---|---|
| 4  | 119.1 | 0.9990 | — | **49.4** |
| 8  | 118.3 | 0.9990 | **0.0004** | 24.3 |
| 16 | 119.1 | 0.9989 | — | 12.1 |
| 32 | 119.1 | 0.9991 | — | 6.0 |

- **Step count is nearly free** — quality is flat from 4→32 steps (unlike continuous diffusion). Ship 4 steps.
- **Real-time**: 49 FPS @ 4 steps ≈ 3.3× Atari real-time.
- **Long-horizon autoregressive**: 119 dB @ step 1 → 90 @ 10 → 72 @ 20 → 41 @ 50. Graceful degradation.
- **Action controllability**: mean pairwise pixel diff 0.19/255 across different actions from the same start frame. Differentiates, subtly (expected for Breakout single-step).
- **Tokenizer**: 55.58 dB recon PSNR, 100% codebook utilization.

## Baseline (DIAMOND)

Cloned to `/root/research/diamond/` and read for architecture reference — U-Net w/ AdaGroupNorm, [64,64,64,64] depth, EDM continuous diffusion, 3 denoising steps default, 4-frame history, MSE loss, RL-return evaluation. Not retrained; compared against published numbers.

## Planned next directions

1. Train an RL agent **inside** the world model (DIAMOND-style validation of the WM for policy learning)
2. **URSA-style metric path** — structured transitions through codebook embedding space instead of uniform masking
3. **AR-DF temporal tube masking** (Lumos-1) — same spatial mask across frames to prevent temporal shortcut learning
4. Multi-game generalization (Pong, MsPacman, Space Invaders)
5. Convolutional VQ-VAE instead of patch-MLP
6. Multi-frame conditioning (4 prev frames like DIAMOND) instead of 1

---

## Where things live

| What | Where |
|---|---|
| Source code | `/root/research/discrete_wm/` |
| uv project root | `/root/research/` (`pyproject.toml`, `uv.lock`) |
| Dataset (raw frames, tokenized data) | `/vast/adi/discrete_wm/data/` |
| Local checkpoint cache | `/vast/adi/discrete_wm/checkpoints/` |
| Training logs (`training_log.csv`, stdout) | `/vast/adi/discrete_wm/logs/` |
| Figures (samples, eval, plots) | `/vast/adi/discrete_wm/figures/` |
| Remote checkpoints | [`adipanda/discrete-wm`](https://huggingface.co/adipanda/discrete-wm) on HuggingFace Hub |
| DIAMOND baseline code | `/root/research/diamond/` |
| Agent scaffolding | `/root/discrete-world-model/` (launch scripts, instructions) |

Every checkpoint save during training is also pushed to HF under `<repo>/<exp_name>/<filename>` in a background thread (non-blocking, fault-tolerant).

Current HF contents:

```
adipanda/discrete-wm/
└── breakout-v1/
    ├── model_latest.pt      # 577 MB  (step 200k)
    ├── model_best.pt        # 192 MB
    └── tokenizer_final.pt   #  2.2 MB
```

---

## Running the autonomous research agent

The agent is driven by Claude Code in a headless loop, with its task defined by `instructions.md`.

**Launch scripts** (`/root/discrete-world-model/launch_agent.sh`, `launch_research.sh` — identical) do:

```bash
IS_SANDBOX=1 stdbuf -oL \
  claude --dangerously-skip-permissions \
         --model claude-opus-4-6 \
         --max-turns 200 \
         --output-format stream-json \
         --verbose \
         -p "$(cat instructions.md)"
```

**To kick it off fresh:**

```bash
cd /root/discrete-world-model
chmod +x launch_agent.sh
nohup ./launch_agent.sh &
disown
tail -f research_agent_output.log
```

- PID is written to `/root/discrete-world-model/agent.pid` — kill with `kill $(cat /root/discrete-world-model/agent.pid)`.
- Full stream-json trace goes to `research_agent_output.log`.
- Phase-by-phase progress log the agent appends to itself: `research_log.md`.

---

## Manual workflow (no agent)

All scripts default to the `/vast/adi/discrete_wm/` paths and push checkpoints to `adipanda/discrete-wm/breakout-v1/` on HF. Override with `--exp-name`, `--hf-repo`, or `--no-hf-push`.

```bash
cd /root/research/discrete_wm

# 1. Collect Breakout frames (→ /vast/adi/discrete_wm/data/atari_data.npz)
python collect_data.py --num-frames 100000

# 2. Train the VQ-VAE tokenizer
python train_tokenizer.py --steps 50000

# 3. Pre-tokenize the dataset
python train_world_model.py --pretokenize

# 4. Train the discrete diffusion world model
python train_world_model.py --total-steps 200000

# 5. Evaluate (PSNR / SSIM / LPIPS / speed / long-horizon / action controllability)
python evaluate.py --checkpoint /vast/adi/discrete_wm/checkpoints/model_best.pt

# 6. Plots
python make_plots.py

# 7. Interactive play
python play_interactive.py --mode terminal
```

Resume training from a saved checkpoint:

```bash
python train_world_model.py --resume /vast/adi/discrete_wm/checkpoints/model_latest.pt
```

Run without HF upload (local only):

```bash
python train_world_model.py --no-hf-push
```

Start a fresh experiment (separate HF subfolder):

```bash
python train_world_model.py --exp-name breakout-v2
```

---

## HuggingFace auth

Scripts use `huggingface_hub`. Authenticate once:

```bash
hf auth login      # or: huggingface-cli login
```

Upload helper lives in `discrete_wm/hf_utils.py` — background-thread pushes that fail gracefully if the network hiccups.

---

## Key files

| File | Purpose |
|---|---|
| `discrete_wm/collect_data.py` | Run random policy on Breakout, save frames + actions |
| `discrete_wm/train_tokenizer.py` | Train the PatchVQ-VAE tokenizer |
| `discrete_wm/train_world_model.py` | Pretokenize + train discrete diffusion world model |
| `discrete_wm/evaluate.py` | Compute quality / speed / long-horizon metrics |
| `discrete_wm/make_plots.py` | Training curves + eval plots |
| `discrete_wm/play_interactive.py` | Drive the model from keyboard / terminal / notebook |
| `discrete_wm/interactive_play.ipynb` | Notebook front-end for interactive play |
| `discrete_wm/hf_utils.py` | HF Hub upload helpers |
| `discrete_wm/models/discrete_diffusion.py` | `PatchVQVAE` + `DiscreteWorldModel` classes |
| `discrete_wm/utils.py` | Datasets + cosine mask schedule |
| `instructions.md` | Agent task spec (phases, budget, success criteria) |
| `research_log.md` | Agent's append-only progress log |
| `results.md` | Preliminary quantitative results + plots |
