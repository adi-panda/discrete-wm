# Autonomous Research Agent: Discrete Diffusion World Models for Atari

## Goal

Build and evaluate a **discrete diffusion world model** for Atari game environments, comparing it against DIAMOND (continuous diffusion world model). You are running on a single H100 80GB GPU. Work autonomously through the phases below. Log everything to `research_log.md`.

---

## CRITICAL RULES

1. **Always append progress to `~/research/research_log.md`** after completing each phase or sub-step. Include timestamps, what you did, what worked, what failed, and any metrics.
2. **Never delete data, checkpoints, or logs.** Only append/create.
3. **If a step fails 3 times, log the error, skip it, and move to the next step.** Don't get stuck in loops.
4. **Keep GPU memory in mind.** You have 80GB VRAM. Use mixed precision (bf16/fp16). If OOM, reduce batch size first, then model size.
5. **Commit working code to git regularly** so nothing is lost.
6. **Time budget: ~10 hours.** Prioritize getting *something* trained and evaluated over perfection.

---

## Phase 0: Environment Setup (~30 min)

### 0.1 Create workspace
```bash
mkdir -p ~/research && cd ~/research
git init
```

### 0.2 Install core dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install einops wandb accelerate tqdm gymnasium[atari] gymnasium[accept-rom-license] ale-py opencv-python-headless imageio pillow scipy
pip install autorom
AutoROM --accept-license
```

### 0.3 Clone DIAMOND as baseline
```bash
git clone https://github.com/eloialonso/diamond.git ~/research/diamond
cd ~/research/diamond
pip install -e .
```
- Read DIAMOND's README and understand the training pipeline
- Verify you can run a short training on Breakout (just 1000 steps to test the pipeline)
- If DIAMOND's repo structure has changed, adapt accordingly. The key file is usually `src/` or `diamond/`

### 0.4 Verify GPU
```bash
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
```

Log the environment setup results.

---

## Phase 1: Understand DIAMOND's Architecture (~30 min)

Before building anything new, study DIAMOND's codebase:

1. **Find and read** the world model class — the part that takes (observation_history, action) → next_observation
2. **Identify** the diffusion process — noise schedule, denoising steps, loss function
3. **Identify** how actions are conditioned — are they embedded and added? Cross-attention? AdaLN?
4. **Identify** the data pipeline — how are Atari frames preprocessed, stored, loaded
5. **Identify** training loop — batch size, learning rate, number of steps for convergence
6. **Note** the evaluation metrics they use — PSNR, SSIM, LPIPS, FVD, or game-specific

Write a summary to `research_log.md` documenting:
- Model architecture (what backbone? U-Net? DiT?)
- How conditioning works
- Frame size and preprocessing
- Default hyperparameters
- How many training steps they report needing

---

## Phase 2: Build the Discrete Diffusion World Model (~3 hours)

This is the core research contribution. You're building an action-conditioned discrete diffusion model for Atari frames.

### 2.1 Design the visual tokenizer

Option A (preferred, faster): Use a **simple learned VQ-VAE** on Atari frames.
- Atari frames are small (typically 64x64 or 84x84 grayscale or 64x64 RGB after preprocessing)
- Build a lightweight VQ-VAE: encoder (4 conv layers with stride 2) → quantize → decoder
- Codebook size: 512 or 1024 tokens
- Spatial downsampling: 4x or 8x, so a 64x64 frame becomes 16x16 or 8x8 token grid = 256 or 64 tokens per frame
- Train this FIRST on collected Atari frames, ~50k steps should suffice
- Save the tokenizer checkpoint

Option B (even faster): Use a **pixel-level discrete representation** — quantize pixel values directly into 256 bins (or fewer, like 64 bins). No learned tokenizer needed. Simpler but might be worse quality.

**Start with Option B for speed, then try Option A if time permits.**

### 2.2 Build the discrete diffusion backbone

Implement a **masked discrete diffusion model** (MDLM-style) for Atari frames:

```python
# Core idea:
# 1. Tokenize frame into discrete tokens (e.g., 64x64 frame → 16x16 grid of VQ codes)
# 2. Forward process: randomly mask tokens with [MASK] token (masking ratio from schedule)
# 3. Model predicts original token IDs for masked positions
# 4. Loss: cross-entropy on masked positions only
# 5. Inference: start from all-masked, iteratively unmask in T steps

class DiscreteWorldModel(nn.Module):
    def __init__(self, vocab_size, num_tokens_per_frame, d_model, n_layers, n_heads, n_actions):
        # Transformer (bidirectional within frame)
        # Action conditioning via learned embedding added to all token positions
        # Previous frame tokens as prefix context (causal across frames)
        pass

    def forward(self, masked_tokens, mask, action_embed, prev_frame_tokens):
        # Returns logits over vocab for each position
        pass
```

Architecture specifics:
- **Backbone**: Small transformer (6-8 layers, 256-512 dim, 4-8 heads)
- **Action conditioning**: Embed action as a vector, add to every token embedding (simplest) OR use AdaLN (better, like DiT)
- **Context**: Concatenate previous frame's tokens as causal prefix. Current frame tokens attend to each other bidirectionally and to previous frame's tokens, but NOT vice versa.
- **Positional encoding**: 2D learned positional encoding for spatial positions within frame
- **Masking schedule**: Cosine schedule for masking ratio during training (sample t ~ U(0,1), mask ratio = cos(t * π/2))

### 2.3 Implement training loop

```python
# Training loop pseudocode:
for batch in dataloader:
    prev_frames, actions, next_frames = batch  # consecutive frame pairs + action

    # Tokenize
    prev_tokens = tokenize(prev_frames)  # [B, N]
    next_tokens = tokenize(next_frames)  # [B, N]

    # Sample masking ratio
    t = torch.rand(B)
    mask_ratio = torch.cos(t * math.pi / 2)

    # Create masked version of next_tokens
    mask = torch.rand(B, N) < mask_ratio.unsqueeze(1)
    masked_next = next_tokens.clone()
    masked_next[mask] = MASK_TOKEN_ID

    # Embed action
    action_embed = action_embedder(actions)

    # Forward
    logits = model(masked_next, mask, action_embed, prev_tokens)

    # Loss: CE only on masked positions
    loss = F.cross_entropy(logits[mask], next_tokens[mask])

    loss.backward()
    optimizer.step()
```

### 2.4 Implement inference (iterative unmasking)

```python
def generate_next_frame(model, prev_tokens, action, num_steps=8):
    B, N = prev_tokens.shape
    tokens = torch.full((B, N), MASK_TOKEN_ID)  # start fully masked

    for step in range(num_steps):
        # Get predictions for all positions
        logits = model(tokens, mask=None, action_embed=embed(action), prev_tokens=prev_tokens)

        # For masked positions, sample from predicted distribution
        probs = F.softmax(logits, dim=-1)

        # Confidence-based unmasking: unmask the most confident predictions
        confidence = probs.max(dim=-1).values  # [B, N]
        still_masked = (tokens == MASK_TOKEN_ID)

        # How many to unmask this step
        n_to_unmask = int(N * (1 - (step + 1) / num_steps) ... ) # schedule

        # Unmask top-confidence positions
        # ... (implementation detail)

    return tokens  # fully unmasked discrete tokens → decode to image
```

### 2.5 Data collection

Collect Atari gameplay data:
```python
import gymnasium as gym

# Collect random-policy gameplay for training data
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
# Collect ~100k frames with actions
# Store as (prev_frame, action, next_frame) triples
# Save to disk as .npz or similar
```

Also try: use a pretrained agent (like a simple DQN) for better coverage of the game state space. But random policy is fine as a starting point — DIAMOND also works with random policy data.

Focus on **one game first**: Breakout is the standard choice. Pong is even simpler if you need a faster iteration cycle.

---

## Phase 3: Training (~3-4 hours)

### 3.1 Train the VQ-VAE tokenizer (if using Option A)
- 50-100k steps on collected frames
- Verify reconstructions look reasonable (save sample images)
- Measure reconstruction PSNR

### 3.2 Train the discrete diffusion world model
- Start small: 4 layers, 256 dim, batch size 64
- Train for 100k-200k steps
- Log loss curve to wandb or to a CSV file
- Save checkpoints every 20k steps
- Every 50k steps, generate sample predictions and save as images:
  - Given: prev_frame + action
  - Model generates: predicted next_frame
  - Compare with: actual next_frame

### 3.3 Hyperparameter notes
- Learning rate: 1e-4 with cosine decay
- Optimizer: AdamW, weight decay 0.01
- Batch size: start at 64, increase if GPU memory allows
- Gradient clipping: 1.0
- Denoising steps at inference: try 4, 8, 16, 32

---

## Phase 4: Evaluation (~1 hour)

### 4.1 Visual quality metrics
Compare your model vs DIAMOND on:
- **PSNR** (peak signal-to-noise ratio) of predicted vs actual next frames
- **SSIM** (structural similarity)
- **LPIPS** (learned perceptual similarity) — `pip install lpips`
- Compute over 1000+ frame predictions

### 4.2 Action controllability
- Generate rollouts with the same starting frame but different actions
- Verify visually that different actions produce different outcomes
- If possible, train a simple action classifier on real frames, then test if it can correctly identify the action from generated frame pairs

### 4.3 Long-horizon rollout quality
- Start from a real frame, autoregressively generate 50-100 frames
- Compute FVD (Fréchet Video Distance) or visual quality over time
- Compare degradation rate: does discrete diffusion maintain quality longer than continuous diffusion?

### 4.4 Inference speed
- Measure frames per second at different denoising step counts (4, 8, 16, 32)
- Compare with DIAMOND's inference speed
- Report wall-clock time per frame on H100

### 4.5 Ablation: number of denoising steps
- Generate the same test frames with 2, 4, 8, 16, 32 steps
- Plot quality (PSNR/SSIM) vs steps
- Find the sweet spot for quality/speed tradeoff

---

## Phase 5: Analysis and Write-up (~1 hour)

### 5.1 Generate comparison figures
Save side-by-side images:
- Column 1: Previous frame (input)
- Column 2: Action taken
- Column 3: Ground truth next frame
- Column 4: DIAMOND prediction (if you got it running)
- Column 5: Your discrete diffusion prediction

### 5.2 Generate plots
- Training loss curves
- PSNR/SSIM vs training steps
- PSNR/SSIM vs denoising steps
- FPS vs denoising steps
- Long-horizon quality degradation over autoregressive steps

### 5.3 Write preliminary results
Create `results.md` with:
- Table of quantitative metrics (your model vs DIAMOND)
- Key figures
- Preliminary conclusions
- What worked, what didn't
- Suggested next steps (e.g., scale up, try Minecraft, try URSA metric path)

---

## Phase 6: If Time Remains — Improvements

Try these in priority order:

1. **URSA-style metric path**: Instead of uniform categorical noise, use URSA's linearized metric path where tokens transition through "nearby" codes in the codebook embedding space. This should improve sample quality.

2. **Multiple Atari games**: Train on Pong, MsPacman, or other games and compare generalization.

3. **Learned VQ-VAE tokenizer**: If you started with pixel quantization (Option B), try training a proper VQ-VAE for better token representations.

4. **AR-DF from Lumos-1**: Implement temporal tube masking — when training on sequences of 2+ frames, apply the same spatial mask pattern across frames to prevent shortcut learning.

5. **Scale up**: Bigger model (8+ layers, 512+ dim), more data, longer training.

---

## File Structure

```
~/research/
├── research_log.md          # Running log of everything
├── diamond/                  # Cloned DIAMOND baseline
├── discrete_wm/             # Your code
│   ├── data/                # Atari frame data
│   ├── models/
│   │   ├── vqvae.py         # Visual tokenizer
│   │   ├── discrete_diffusion.py  # Core discrete diffusion model
│   │   └── world_model.py   # Action-conditioned wrapper
│   ├── train_tokenizer.py
│   ├── train_world_model.py
│   ├── evaluate.py
│   ├── collect_data.py
│   └── utils.py
├── checkpoints/              # Saved models
├── figures/                  # Generated comparison images and plots
└── results.md               # Final results summary
```

---

## Debugging Tips

- If training loss doesn't decrease after 5k steps: check that masking is working correctly, check that action conditioning is being applied, try a higher learning rate
- If generated frames are blurry/noisy: increase denoising steps, check tokenizer quality, try temperature < 1.0 during sampling
- If GPU OOM: reduce batch size → reduce model dim → reduce sequence length → use gradient checkpointing
- If Atari env crashes: make sure `gymnasium[atari]` and `ale-py` are installed, run `AutoROM --accept-license`
- If DIAMOND won't install: it's OK, just run your model standalone and compare against their reported numbers from the paper

---

## Success Criteria

By morning, I want to see in `research_log.md`:
1. ✅ Data collected (≥50k frame-action pairs)
2. ✅ Discrete diffusion model implemented and training
3. ✅ At least 100k training steps completed
4. ✅ Sample generated frames saved (even if quality is poor)
5. ✅ Basic metrics computed (PSNR, SSIM on ≥100 test frames)
6. ✅ Inference speed measured
7. ✅ Preliminary results written up

Even partial results are valuable. The point is to validate the approach and identify the right hyperparameters for a full-scale run later.