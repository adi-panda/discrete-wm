"""
Discrete Diffusion World Model for Atari.

Uses masked discrete diffusion (MDLM-style) with patch-based tokenization:
- Frames (64x64 RGB) are split into 4x4 patches → 16x16 = 256 tokens
- Each patch is quantized to one of `vocab_size` discrete codes
- Forward process: randomly mask tokens with [MASK]
- Model predicts original token IDs for masked positions
- Inference: start from all-masked, iteratively unmask

Architecture: Bidirectional Transformer with AdaLN conditioning (DiT-style)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --- Patch Tokenizer (lightweight VQ-VAE) ---

class PatchVQVAE(nn.Module):
    """
    Simple VQ-VAE that converts 4x4x3 patches into discrete tokens.
    Encoder: linear projection of flattened patch → codebook lookup
    Decoder: codebook embedding → linear projection → 4x4x3 patch
    """
    def __init__(self, patch_size=4, num_channels=3, vocab_size=512, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.patch_dim = patch_size * patch_size * num_channels  # 48

        # Encoder: deeper for better representations
        self.encoder = nn.Sequential(
            nn.Linear(self.patch_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Codebook
        self.codebook = nn.Embedding(vocab_size, embed_dim)
        # Better initialization: normal with reasonable scale
        nn.init.normal_(self.codebook.weight, std=0.1)

        # Decoder: deeper for better reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.patch_dim),
        )

        # Track codebook usage for reset
        self.register_buffer('code_count', torch.zeros(vocab_size))
        self.register_buffer('code_avg', torch.zeros(vocab_size, embed_dim))

    def encode(self, frames):
        """
        frames: [B, H, W, C] uint8 or float [0, 255]
        Returns: tokens [B, N], embeddings [B, N, D]
        """
        if frames.dtype == torch.uint8:
            frames = frames.float()
        frames = frames / 255.0 * 2.0 - 1.0  # Normalize to [-1, 1]

        B, H, W, C = frames.shape
        ps = self.patch_size

        # Extract patches: [B, H/ps, W/ps, ps*ps*C]
        patches = rearrange(frames, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)',
                           p1=ps, p2=ps)  # [B, N, 48]

        # Encode
        z_e = self.encoder(patches)  # [B, N, D]

        # Quantize: find nearest codebook entry
        # z_e: [B, N, D], codebook: [V, D]
        dists = torch.cdist(z_e, self.codebook.weight.unsqueeze(0).expand(B, -1, -1))  # [B, N, V]
        tokens = dists.argmin(dim=-1)  # [B, N]

        # Get quantized embeddings
        z_q = self.codebook(tokens)  # [B, N, D]

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        return tokens, z_q_st, z_e, z_q

    def decode(self, z_q):
        """
        z_q: [B, N, D]
        Returns: frames [B, H, W, C] in [-1, 1]
        """
        patches = self.decoder(z_q)  # [B, N, patch_dim]
        h = w = int(math.sqrt(patches.shape[1]))
        ps = self.patch_size
        frames = rearrange(patches, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
                          h=h, w=w, p1=ps, p2=ps)
        return frames

    def decode_tokens(self, tokens):
        """Decode from discrete token IDs."""
        z_q = self.codebook(tokens)
        return self.decode(z_q)

    def reset_dead_codes(self, z_e, tokens):
        """Reset unused codebook entries to random encoder outputs."""
        if not self.training:
            return
        with torch.no_grad():
            # Update usage counts
            flat_tokens = tokens.reshape(-1)
            for idx in flat_tokens.unique():
                self.code_count[idx] += (flat_tokens == idx).sum().float()

            # Find dead codes (used < 1 time on average)
            dead_mask = self.code_count < 1.0
            n_dead = dead_mask.sum().item()
            if n_dead > 0 and n_dead < self.vocab_size:
                # Replace dead codes with random encoder outputs
                flat_z = z_e.detach().reshape(-1, self.embed_dim)
                rand_idx = torch.randint(0, len(flat_z), (n_dead,), device=z_e.device)
                new_codes = (flat_z[rand_idx] + torch.randn_like(flat_z[rand_idx]) * 0.01).float()
                self.codebook.weight.data[dead_mask] = new_codes

            # Decay counts so codes need to keep being used
            self.code_count *= 0.99

    def forward(self, frames):
        """Full encode-decode for training the tokenizer."""
        tokens, z_q_st, z_e, z_q = self.encode(frames)
        recon = self.decode(z_q_st)

        # Reset dead codebook entries
        self.reset_dead_codes(z_e, tokens)

        # Losses
        recon_loss = F.mse_loss(recon, frames.float() / 255.0 * 2.0 - 1.0)
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z_e.detach())

        return recon, tokens, recon_loss, commitment_loss, codebook_loss


# --- Positional Encoding ---

class LearnedPositionalEncoding2D(nn.Module):
    """Learned 2D positional encoding for spatial token grids."""
    def __init__(self, d_model, max_h=16, max_w=16):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, d_model // 2)
        self.col_embed = nn.Embedding(max_w, d_model // 2)

    def forward(self, h, w, device):
        rows = torch.arange(h, device=device)
        cols = torch.arange(w, device=device)
        row_emb = self.row_embed(rows).unsqueeze(1).expand(-1, w, -1)
        col_emb = self.col_embed(cols).unsqueeze(0).expand(h, -1, -1)
        pos = torch.cat([row_emb, col_emb], dim=-1)
        return pos.reshape(h * w, -1)


# --- AdaLN (Adaptive Layer Norm, DiT-style) ---

class AdaLN(nn.Module):
    """Adaptive Layer Normalization conditioned on external signal."""
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, d_model * 2)

    def forward(self, x, cond):
        scale_shift = self.proj(cond).unsqueeze(1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


# --- Transformer Block with AdaLN ---

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, cond_dim, dropout=0.0):
        super().__init__()
        self.adaln1 = AdaLN(d_model, cond_dim)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.adaln_cross = AdaLN(d_model, cond_dim)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.adaln2 = AdaLN(d_model, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond, context):
        # Self-attention
        h = self.adaln1(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h

        # Cross-attention to previous frame
        h = self.adaln_cross(x, cond)
        h, _ = self.cross_attn(h, context, context)
        x = x + h

        # FFN
        h = self.adaln2(x, cond)
        h = self.ffn(h)
        x = x + h

        return x


# --- Main Discrete Diffusion Model ---

class DiscreteWorldModel(nn.Module):
    """
    Masked discrete diffusion model for Atari frame prediction.

    Given previous frame tokens + action, predicts next frame tokens.
    Uses cosine masking schedule and iterative unmasking for generation.

    Token grid: 16x16 = 256 tokens per frame (from 4x4 patches of 64x64 frames).
    Supports multi-frame context via cross-attention (context_frames=1 or 4).
    """
    def __init__(
        self,
        vocab_size=512,
        grid_h=16,
        grid_w=16,
        d_model=512,
        n_layers=8,
        n_heads=8,
        n_actions=4,
        dropout=0.0,
        cond_dim=512,
        context_frames=1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_tokens = grid_h * grid_w  # 256
        self.d_model = d_model
        self.mask_token_id = vocab_size
        self.context_frames = context_frames

        self.token_embed = nn.Embedding(vocab_size + 1, d_model)
        self.pos_enc = LearnedPositionalEncoding2D(d_model, grid_h, grid_w)

        if context_frames > 1:
            self.frame_pos_embed = nn.Embedding(context_frames, d_model)

        self.action_embed = nn.Sequential(
            nn.Embedding(n_actions, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.prev_frame_proj = nn.Linear(d_model, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, cond_dim, dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def _build_context(self, prev_tokens, device):
        """Build cross-attention context from previous frame tokens.

        Args:
            prev_tokens: [B, N] for single frame, [B, C, N] for multi-frame
        Returns:
            context: [B, C*N, D]
        """
        pos = self.pos_enc(self.grid_h, self.grid_w, device)

        if self.context_frames == 1:
            if prev_tokens.dim() == 3:
                prev_tokens = prev_tokens[:, 0]
            context = self.token_embed(prev_tokens)
            context = self.prev_frame_proj(context + pos.unsqueeze(0))
            return context

        B = prev_tokens.shape[0]
        C = self.context_frames
        N = self.n_tokens

        if prev_tokens.dim() == 2:
            prev_tokens = prev_tokens.unsqueeze(1).expand(-1, C, -1)

        all_contexts = []
        for f in range(C):
            emb = self.token_embed(prev_tokens[:, f])
            emb = emb + pos.unsqueeze(0) + self.frame_pos_embed.weight[f].unsqueeze(0).unsqueeze(0)
            all_contexts.append(emb)
        context = torch.cat(all_contexts, dim=1)  # [B, C*N, D]
        context = self.prev_frame_proj(context)
        return context

    def forward(self, masked_tokens, prev_tokens, action, mask_ratio):
        """
        Args:
            masked_tokens: [B, N]
            prev_tokens: [B, N] or [B, C, N] for multi-frame context
            action: [B]
            mask_ratio: [B]
        Returns:
            logits: [B, N, vocab_size]
        """
        B = masked_tokens.shape[0]
        device = masked_tokens.device

        x = self.token_embed(masked_tokens)
        pos = self.pos_enc(self.grid_h, self.grid_w, device)
        x = x + pos.unsqueeze(0)

        context = self._build_context(prev_tokens, device)

        act_emb = self.action_embed(action)
        if mask_ratio.dim() == 1:
            mask_ratio = mask_ratio.unsqueeze(-1)
        time_emb = self.time_embed(mask_ratio.float())
        cond = self.cond_proj(act_emb + time_emb)

        for layer in self.layers:
            x = layer(x, cond, context)

        x = self.output_norm(x)
        logits = self.output_head(x)

        return logits

    @torch.no_grad()
    def generate(self, prev_tokens, action, num_steps=8, temperature=1.0, device='cuda'):
        """
        Args:
            prev_tokens: [B, N] or [B, C, N]
            action: [B]
        Returns:
            tokens: [B, N]
        """
        if prev_tokens.dim() == 3:
            B = prev_tokens.shape[0]
            N = prev_tokens.shape[2]
        else:
            B, N = prev_tokens.shape

        tokens = torch.full((B, N), self.mask_token_id, device=device, dtype=torch.long)
        is_masked = torch.ones(B, N, dtype=torch.bool, device=device)

        for step in range(num_steps):
            ratio = 1.0 - (step + 1) / num_steps
            mask_ratio_tensor = torch.full((B,), max(ratio, 0.0), device=device)

            logits = self.forward(tokens, prev_tokens, action, mask_ratio_tensor)

            probs = F.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(B, N)
            confidence = probs.max(dim=-1).values

            confidence = confidence.masked_fill(~is_masked, -float('inf'))

            if step < num_steps - 1:
                n_masked = is_masked.sum(dim=1).float()
                target_masked = max(ratio, 0.0) * N
                n_to_unmask = (n_masked - target_masked).clamp(min=1).long()

                for b in range(B):
                    masked_idx = is_masked[b].nonzero(as_tuple=True)[0]
                    if len(masked_idx) == 0:
                        continue
                    n_unmask = min(n_to_unmask[b].item(), len(masked_idx))
                    conf = confidence[b, masked_idx]
                    _, top_k = conf.topk(n_unmask)
                    unmask_pos = masked_idx[top_k]
                    tokens[b, unmask_pos] = sampled[b, unmask_pos]
                    is_masked[b, unmask_pos] = False
            else:
                tokens = torch.where(is_masked, sampled, tokens)
                is_masked.fill_(False)

        return tokens


# --- Tokenization utilities ---

def tokenize_frames_direct(frames, num_bins=64):
    """
    Direct pixel quantization (Option B - no VQ-VAE needed).
    Quantizes each pixel channel independently into bins.

    frames: [B, H, W, C] uint8 [0, 255]
    Returns: tokens [B, H*W, C] in [0, num_bins-1]
    """
    if frames.dtype == torch.uint8:
        frames = frames.float()
    if frames.dim() == 4 and frames.shape[1] == 3:
        frames = frames.permute(0, 2, 3, 1)
    B, H, W, C = frames.shape
    tokens = (frames / 256.0 * num_bins).long().clamp(0, num_bins - 1)
    return tokens.reshape(B, H * W, C)


def detokenize_frames_direct(tokens, num_bins=64, h=64, w=64):
    """Reverse of tokenize_frames_direct."""
    frames = ((tokens.float() + 0.5) / num_bins * 256.0).clamp(0, 255).byte()
    B = frames.shape[0]
    return frames.reshape(B, h, w, -1)
