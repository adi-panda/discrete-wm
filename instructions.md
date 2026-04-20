# Autonomous Research Agent: Discrete Diffusion vs DIAMOND — Clean Ablation on Breakout

---

## ⚡ RESUME MODE — READ FIRST ⚡

This is a **resume** of an earlier run. State as of 2026-04-20 ~17:40 UTC:

- Phases 0, 1, 2 (our WM), 4 (IDM) are **DONE**. Check `research_log.md` for details.
- Phase 3 (DIAMOND) **stalled at step 20k of 100k** because the agent invoked training with a trailing `| head -5` pipe that sent SIGPIPE to the python process. The python process lingered at 0% GPU util until killed.
- Phases 5 (unified eval) and 6 (write-up) **have not run**.

**Your job, in order:**
1. **Verify the GPU is free.** `nvidia-smi` should show low utilization; if any stale `python train_diamond_standalone.py` process is still alive, kill it.
2. **Restart DIAMOND training to 100k steps** using `discrete_wm/train_diamond_standalone.py` (already exists). **CRITICAL: do NOT pipe stdout through `head`, `tee | head`, or any other pipe that can close early.** Run it as a backgrounded job with stdout/stderr redirected to a log file, e.g.:
   ```bash
   cd /root/research/discrete_wm
   nohup python train_diamond_standalone.py \
     --train-dir /vast/adi/discrete_wm/data/ablation_v1/diamond_dataset/train \
     --ckpt-dir /vast/adi/discrete_wm/checkpoints/ablation_v1 \
     --log-dir /vast/adi/discrete_wm/logs/ablation_v1 \
     --total-steps 100000 \
     --batch-size 32 \
     > /vast/adi/discrete_wm/logs/ablation_v1/diamond_stdout.log 2>&1 &
   echo $! > /vast/adi/discrete_wm/logs/ablation_v1/diamond.pid
   ```
   Check progress by `tail`ing the log file or reading the CSV. Expected ~1.6h to finish.
3. **If DIAMOND has a `--resume` flag**, use it to resume from `diamond_step_020000.pt`. Otherwise start fresh — the 20k checkpoint is undertrained either way.
4. **While DIAMOND trains, write `discrete_wm/evaluate_ablation.py`** per Phase 5 so it's ready the moment DIAMOND finishes.
5. **Run the eval harness** against both models (our `model_best.pt` and the final DIAMOND checkpoint).
6. **Write `/root/research/comparison.md`** per Phase 6.

If DIAMOND stalls or crashes a second time, fall back to Option B: load DIAMOND's pretrained `eloialonso/diamond` Breakout checkpoint, note the data mismatch caveat clearly in `comparison.md`, and ship the eval anyway.

**Do not redo Phases 0–2 or 4.** Append everything to `research_log.md` under a new section `## Resume: 2026-04-20`.

---

## Goal

Run a **head-to-head ablation** between our masked discrete diffusion world model and **DIAMOND** (continuous diffusion baseline), trained on **identical data** and evaluated with an **identical harness**. Single H100 80GB. Time budget: ~10 hours overnight. Log everything to `research_log.md`.

This experiment replaces the earlier `breakout-v1` run. That run proved the discrete diffusion mechanism fits Atari frames but had two fatal flaws for a paper comparison:
- **Random policy data** → ball barely moves → "copy previous frame" is near-optimal → inflated PSNR, trivial action controllability.
- **Single-frame context** → model can't infer velocity → action conditioning is underpowered.
- **No actual DIAMOND run** → comparisons were apples-to-oranges against paper numbers.

This run fixes all three. Experiment name: **`breakout-ablation-v1`**. Do not overwrite `breakout-v1` data or checkpoints.

---

## CRITICAL RULES

1. **Always append progress to `/root/research/research_log.md`** after each phase. Include timestamps, metrics, what worked, what failed.
2. **Never delete data, checkpoints, or logs.** Only append/create. `breakout-v1` stays untouched.
3. **New paths everywhere.** Data under `/vast/adi/discrete_wm/data/ablation_v1/`, checkpoints under `/vast/adi/discrete_wm/checkpoints/ablation_v1/`, figures under `/vast/adi/discrete_wm/figures/ablation_v1/`, HF subfolder `adipanda/discrete-wm/ablation-v1/`.
4. **If a step fails 3 times, log the error, skip, move on.** Do not loop.
5. **Mixed precision (bf16).** If OOM, reduce batch size before model size.
6. **Commit working code to git after each phase** so nothing is lost.
7. **Same train/test split for both models.** Split by episode, hold out last ~10% of episodes for the test set. Write the split indices to disk in Phase 1 and load them everywhere downstream.

---

## Target deliverable

By morning, a single `comparison.md` with an ablation table:

| Metric | DIAMOND | Ours (discrete diff) |
|---|---|---|
| Next-frame PSNR | ... | ... |
| Next-frame SSIM | ... | ... |
| Next-frame LPIPS | ... | ... |
| FVD (16-step rollouts) | ... | ... |
| IDM action F1 | ... | ... |
| Long-horizon PSNR @ step 10 / 20 / 50 | ... | ... |
| Inference FPS (H100, bs=1) | ... | ... |
| Params | ~8M | ~48M |

Same data, same held-out frames, same IDM, same sampling length. Only the WM differs.

---

## Phase 0: Setup (~20 min)

### 0.1 Verify env
```bash
cd /root/research
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"
```
Should print H100 and ~80 GB.

### 0.2 Confirm DIAMOND clone
`/root/research/diamond/` already exists. Verify:
```bash
ls /root/research/diamond/src/        # expect: main.py, trainer.py, agent.py, models/, data/
ls /root/research/diamond/config/      # expect: trainer.yaml, agent/, env/
```

If missing files, re-clone: `git clone https://github.com/eloialonso/diamond.git /root/research/diamond`. Install in editable mode:
```bash
cd /root/research/diamond && pip install -e . || true
```
(OK if install fails — we can run scripts directly via `python src/main.py`.)

### 0.3 Pre-create output dirs
```bash
mkdir -p /vast/adi/discrete_wm/data/ablation_v1
mkdir -p /vast/adi/discrete_wm/data/ablation_v1/diamond_dataset/{train,test}
mkdir -p /vast/adi/discrete_wm/checkpoints/ablation_v1
mkdir -p /vast/adi/discrete_wm/figures/ablation_v1
mkdir -p /vast/adi/discrete_wm/logs/ablation_v1
```

Log env + paths to `research_log.md`.

---

## Phase 1: Shared Data Collection (~45 min)

**Single source of truth.** Both models see the same frames. Policy is DIAMOND's pretrained Breakout agent (not random) so actions actually change the game state.

### 1.1 Download DIAMOND's pretrained Breakout agent

HF repo: `eloialonso/diamond`, file: `atari_100k/models/BreakoutNoFrameskip-v4.pt`.

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="eloialonso/diamond",
                       filename="atari_100k/models/BreakoutNoFrameskip-v4.pt")
```

Load into DIAMOND's `Agent` class (`diamond/src/agent.py`). You want the actor-critic head to drive action selection; the WM portion of the checkpoint is not used for collection.

### 1.2 Write a collector script: `discrete_wm/collect_data_v2.py`

- Env: `BreakoutNoFrameskip-v4` via DIAMOND's `atari_preprocessing.py` pipeline — **use theirs verbatim** so frames match DIAMOND's training distribution (64×64 RGB uint8, frame-skip 4, max-pool over last 2 raw frames, `cv2.INTER_AREA` resize, no frame stacking).
- Episodes: run DIAMOND's agent until you have ≥100k total transitions across ≥400 episodes. Use epsilon 0.05 for mild exploration diversity.
- Save **both formats** in a single pass:
  - **Our format**: `ablation_v1/atari_data.npz` with `{frames: [N,64,64,3] uint8, actions: [N] int64, episode_ends: [E] int64}`.
  - **DIAMOND format**: one `.pt` file per episode under `ablation_v1/diamond_dataset/train/0/0/{i}.pt` (and `/test/...` for the 10% held-out). Each episode dict: `{'obs': [T,64,64,3] uint8, 'act': [T] int64, 'rew': [T] float32, 'end': [T] bool, 'trunc': [T] bool}`. See `diamond/src/data/episode.py` and `diamond/src/data/dataset.py` for the exact schema.
  - Also save a `train_test_split.json` with episode index lists so all downstream steps use the identical split.

### 1.3 Sanity checks before moving on

- Action distribution should be non-uniform (FIRE and LEFT/RIGHT dominate for a trained policy — good).
- Mean pixel diff between consecutive frames >> our old random-policy number. If it's still tiny, something is broken.
- Render 5 episodes as GIFs to `figures/ablation_v1/collection_sanity/`.

Log counts, action histogram, mean frame-to-frame diff.

---

## Phase 2: Our Discrete Diffusion WM with 4-Frame Context (~4 hours)

### 2.1 Reuse or retrain the VQ-VAE tokenizer

The `breakout-v1` tokenizer hit 55 dB recon PSNR with 100% codebook util on random-policy frames. Test it on the new agent-collected frames:

```python
# Load /vast/adi/discrete_wm/checkpoints/tokenizer_final.pt
# Encode+decode 1000 sample frames from ablation_v1 data, compute PSNR
```

- If ≥ 45 dB: **reuse it**, copy to `checkpoints/ablation_v1/tokenizer_final.pt`.
- If < 45 dB: retrain 50k steps on the new data (~20 min).

### 2.2 Architectural change: 4-frame context

Current `DiscreteWorldModel` in `discrete_wm/models/discrete_diffusion.py` cross-attends to tokens of **one** previous frame. Extend to **4** (DIAMOND's `num_steps_conditioning`).

Minimum change: concatenate the 4 previous frames' token sequences (4 × 256 = 1024 tokens) as the cross-attention key/value source. Add a learned frame-position embedding (0..3) so the model can distinguish t-1, t-2, t-3, t-4. Do not change the query side (still 256 tokens of the current frame).

Edge case: for the first 3 frames of every episode, pad missing past frames with a copy of the earliest available frame (document this choice in the log).

### 2.3 Pre-tokenize

Run `train_world_model.py --pretokenize` over the new data to produce `ablation_v1/tokenized_data.npz` with shape-matched (prev_tokens[4,256], action, next_tokens[256]) tuples.

### 2.4 Train

```bash
python train_world_model.py \
  --exp-name breakout-ablation-v1 \
  --data /vast/adi/discrete_wm/data/ablation_v1/tokenized_data.npz \
  --tokenizer-ckpt /vast/adi/discrete_wm/checkpoints/ablation_v1/tokenizer_final.pt \
  --total-steps 100000 \
  --batch-size 64 \
  --context-frames 4
```

- 100k steps (not 200k) to fit in overnight budget alongside DIAMOND training.
- bs=64 (down from 128) — 4× context quadruples memory for the cross-attention KV cache.
- Everything else unchanged: lr 1e-4 cosine, AdamW β=(0.9, 0.95), wd 0.01, grad clip 1.0, bf16.
- Save checkpoint every 20k steps, push to `adipanda/discrete-wm/ablation-v1/` via existing `hf_utils.py`.
- Generate sample predictions every 20k steps to `figures/ablation_v1/samples/`.

Log loss curve and sample PSNR at each checkpoint.

---

## Phase 3: DIAMOND WM on the Same Data (~3 hours)

### 3.1 Configure static-dataset, WM-only training

Use DIAMOND's `static_dataset.path` mechanism (`diamond/config/trainer.yaml:50-52`) to bypass live collection. Disable the RL agent loop so we train **only the denoiser** — we don't care about their reward/end/actor-critic heads for this comparison.

Create a config override (either via CLI or a new YAML under `diamond/config/`) that sets:

```yaml
static_dataset:
  path: /vast/adi/discrete_wm/data/ablation_v1/diamond_dataset
  ignore_sample_weights: true

training:
  model_free: false         # no model-free RL data collection
  should_collect_train_in_env: false
  should_collect_test_in_env: false
  num_epochs: 100           # tune to hit ~100k denoiser steps total
  num_steps_first_epoch: 10000
  num_steps_per_epoch: 1000

agent:
  denoiser:
    # keep default [64,64,64,64] channels, num_steps_conditioning=4, attention_depths=[0,0,0,0]
  rew_end_model: null       # skip if possible
  actor_critic: null        # skip if possible
```

If `rew_end_model`/`actor_critic` cannot be disabled cleanly, train them too but ignore their outputs — only the denoiser is evaluated.

Run:
```bash
cd /root/research/diamond
python src/main.py \
  env.train.id=BreakoutNoFrameskip-v4 \
  static_dataset.path=/vast/adi/discrete_wm/data/ablation_v1/diamond_dataset \
  training.model_free=false \
  hydra.run.dir=/vast/adi/discrete_wm/checkpoints/ablation_v1/diamond_run
```

Target: ~100k denoiser optimizer steps to roughly match our compute. Log denoising MSE every 1k steps. Save final checkpoint to `checkpoints/ablation_v1/diamond_final.pt`.

### 3.2 If DIAMOND's training pipeline balks

Common issues and fallbacks:
- Episode format mismatch → re-verify schema against `diamond/src/data/episode.py:10-15`.
- Hydra config errors → drop the YAML and override everything via CLI.
- Reward/end model requires reward data → our collector already saves rewards, so this should be fine.
- If after 3 serious attempts you cannot get static-dataset training working, **fall back to loading DIAMOND's pretrained `BreakoutNoFrameskip-v4.pt` checkpoint and note the data mismatch in the log.** This is Option B from the plan and weaker, but still better than no DIAMOND baseline.

---

## Phase 4: Inverse Dynamics Model for Controllability (~20 min)

Train a small IDM on **ground-truth** frames so it's unbiased between the two WMs.

`discrete_wm/train_idm.py`:
- Input: `(prev_frame, next_frame)` concatenated along channels → 6 channels.
- Architecture: 4-conv CNN (64 → 128 → 256 → 256, stride 2 each) → global avg pool → linear → softmax over 4 actions.
- Data: ground-truth `(prev, next, action)` tuples from `ablation_v1/atari_data.npz` (same train split, same held-out test set).
- Train 20k steps, bs=256, lr 3e-4, AdamW.
- Target test accuracy ≥ 85%. If lower, the controllability metric is noisy — log this clearly.
- Save to `checkpoints/ablation_v1/idm.pt`.

---

## Phase 5: Unified Evaluation Harness (~1 hour)

Write **one script**: `discrete_wm/evaluate_ablation.py`. It loads both models and runs identical metrics on identical test frames.

### 5.1 Model loading abstraction

Two thin adapters with the same interface:

```python
class WMAdapter:
    def predict_next_frame(self, prev_frames_rgb_uint8, action) -> next_frame_rgb_uint8:
        ...

class DiscreteWMAdapter(WMAdapter):
    # tokenize prev_frames → run our generate() → decode tokens → uint8
    ...

class DiamondWMAdapter(WMAdapter):
    # call their denoiser.denoise() with 3 sampling steps → [-1,1] → uint8
    ...
```

Both take the **same** 4-frame RGB uint8 context and an action, return a 64×64 RGB uint8 next-frame prediction.

### 5.2 Metrics (run each on both adapters over the identical held-out test set)

1. **Next-frame quality** — PSNR, SSIM, LPIPS over 1000 held-out (prev_frames, action, gt_next) tuples.
2. **FVD** — 16-frame rollouts, 50 rollouts, compute Fréchet Video Distance against ground-truth rollouts. Use `torchmetrics` or `fvd.py` from a standard implementation — do not hand-roll.
3. **IDM action F1** — for N test contexts, generate next frame under each of the 4 actions → feed (prev, gen_next) into IDM → check F1 of predicted action vs commanded action. This is the MineWorld-standard controllability metric.
4. **Long-horizon rollout PSNR** — from the same 10 seed contexts, autoregressively generate 50 frames, PSNR vs ground-truth trajectory at steps 1/5/10/20/30/50.
5. **Inference FPS** — 100 single-frame generates at bs=1, report mean ms/frame and FPS. For our model sweep 4/8/16 denoising steps; for DIAMOND sweep 3/5/10.

Save `comparison.json` with every number, and side-by-side rollout GIFs to `figures/ablation_v1/rollouts_cmp/` (columns: GT | DIAMOND | Ours).

### 5.3 Same-seed parity

Critical for FVD and rollout metrics: use identical starting frames and identical action sequences for both models. Seed torch/np before every generate call.

---

## Phase 6: Write-up (~30 min)

Generate `/root/research/comparison.md` with:

- The ablation table (format above).
- 1 sentence per metric on who wins and by how much.
- A "Setup parity checklist" — data source, num training steps, train/test split, context length, eval harness — to pre-empt reviewer "but the comparison wasn't fair" complaints.
- Honest caveats: we trained DIAMOND from scratch for only ~100k steps (they use full iterative pipeline); single game only; random-ish data not a full competent-agent distribution.
- Update `README.md` with a one-line pointer to this file.

Commit everything.

---

## Success Criteria

1. ✅ ≥100k frames collected via pretrained DIAMOND agent, saved in both formats with a logged train/test split.
2. ✅ Our discrete WM trained ≥100k steps with 4-frame context on the new data.
3. ✅ DIAMOND WM trained ≥100k denoiser steps on the identical static dataset (or fallback to their pretrained checkpoint with clear logging).
4. ✅ IDM trained with ≥85% test acc.
5. ✅ `evaluate_ablation.py` runs both models through identical PSNR / SSIM / LPIPS / FVD / IDM-F1 / long-horizon / FPS measurements.
6. ✅ `comparison.md` written with the ablation table and honest caveats.

Partial results are fine. If DIAMOND training stalls, prioritize getting the eval harness working against their pretrained checkpoint (Option B fallback). The eval harness itself is the most valuable artifact — it will be reused for every future experiment.

---

## File Structure (updated)

```
/root/research/
├── research_log.md           # append-only progress log
├── comparison.md             # Phase 6 deliverable
├── diamond/                  # upstream DIAMOND, used in Phase 3
├── discrete_wm/
│   ├── collect_data_v2.py    # NEW: DIAMOND-agent-driven collection, dual-format save
│   ├── train_tokenizer.py    # existing
│   ├── train_world_model.py  # UPDATED: --context-frames flag
│   ├── train_idm.py          # NEW: Phase 4
│   ├── evaluate_ablation.py  # NEW: unified harness, both adapters
│   ├── make_plots.py         # existing
│   ├── hf_utils.py           # existing
│   ├── models/discrete_diffusion.py  # UPDATED: 4-frame cross-attention context
│   └── utils.py              # UPDATED: dataset returns 4 prev frames, not 1
└── instructions.md           # this file

/vast/adi/discrete_wm/
├── data/ablation_v1/
│   ├── atari_data.npz
│   ├── tokenized_data.npz
│   ├── train_test_split.json
│   └── diamond_dataset/{train,test}/0/0/*.pt
├── checkpoints/ablation_v1/
│   ├── tokenizer_final.pt
│   ├── discrete_wm_*.pt
│   ├── diamond_run/...        # hydra output dir
│   ├── diamond_final.pt
│   └── idm.pt
├── figures/ablation_v1/
│   ├── collection_sanity/
│   ├── samples/
│   ├── rollouts_cmp/
│   └── plots/
└── logs/ablation_v1/

HuggingFace: adipanda/discrete-wm/ablation-v1/...
```

---

## Debugging tips

- If our WM's loss doesn't drop below ~0.5: verify masking actually masks, AdaLN is getting a non-trivial action embedding, and the 4-frame context is actually being attended to (not zeroed).
- If DIAMOND training crashes on dataset load: check that episodes are `.pt` dicts, obs is uint8, and the `/train/0/0/` hierarchy matches `diamond/src/data/dataset.py:122-128` exactly.
- If FVD is implausibly bad: double-check frame dtype and range — most FVD implementations expect float in [0,1] or uint8 in [0,255], not [-1,1].
- If IDM action F1 is ~25% (random): IDM itself may not be converged, or generated frames may be too similar across actions. Check both.
- GPU OOM during ours: reduce bs 64 → 32, then `d_model` 512 → 384.
- GPU OOM during DIAMOND: their default bs=32 is already small; reduce `num_steps_conditioning` from 4 → 2 as last resort (and match this on our side for fairness).
