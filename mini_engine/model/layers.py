"""
model/layers.py
Core building blocks for LLaMA / Qwen2.5 architecture:
  RMSNorm, HuggingFace-compatible RoPE, Grouped-Query Attention (GQA) with paged KV cache,
  SwiGLU MLP, TransformerBlock.
"""
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x) * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE) - HuggingFace format
# ---------------------------------------------------------------------------

def precompute_freqs(head_dim: int, max_seq_len: int, theta: float = 1000000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos and sin tables for rotary position embedding."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation in HF format: (x * cos) + (rotate_half(x) * sin).
    x: (bsz, num_heads, seq_len, head_dim)
    cos, sin: (bsz, 1, seq_len, head_dim)
    """
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads when num_kv_heads < num_attention_heads."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Grouped-Query Attention (GQA) with optional Paged KV Cache
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Support biases (needed by Qwen2.5)
        has_bias = getattr(config, "attention_bias", True)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=has_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=has_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=has_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,                          # (bsz, q_len, hidden)
        cos: torch.Tensor,                        # (bsz, 1, q_len, head_dim)
        sin: torch.Tensor,                        # (bsz, 1, q_len, head_dim)
        kv_cache=None,                            # PagedKVCache | None
        layer_id: int = 0,
        block_table: Optional[List[int]] = None,
        num_cached_tokens: int = 0,
    ) -> torch.Tensor:
        bsz, q_len, _ = x.shape
        is_decode = (q_len == 1 and kv_cache is not None and block_table is not None and num_cached_tokens > 0)

        # 1. Projections
        q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 2. RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # 3. Paged KV cache write / gather
        if kv_cache is not None and block_table is not None:
            block_size = kv_cache.block_size
            if is_decode:
                # Write 1 token at num_cached_tokens
                pos = num_cached_tokens
                b_idx = pos // block_size
                b_offset = pos % block_size
                if b_idx < len(block_table):
                    phys_b = block_table[b_idx]
                    # k[:, :, 0, :]: shape (bsz, num_kv_heads, head_dim) -> (num_kv_heads, head_dim)
                    kv_cache.write_to_block(layer_id, phys_b, b_offset, k[0, :, 0, :], v[0, :, 0, :])

                # Gather full history
                seq_len = num_cached_tokens + 1
                k_hist, v_hist = kv_cache.gather_kv(layer_id, block_table, seq_len)
                # k_hist shape: (seq_len, num_kv_heads, head_dim) -> transpose to (1, num_kv_heads, seq_len, head_dim)
                k = k_hist.unsqueeze(0).transpose(1, 2).to(q.dtype)
                v = v_hist.unsqueeze(0).transpose(1, 2).to(q.dtype)
            else:
                # Prefill: write all tokens into allocated blocks
                for t in range(q_len):
                    b_idx = t // block_size
                    b_offset = t % block_size
                    if b_idx < len(block_table):
                        phys_b = block_table[b_idx]
                        kv_cache.write_to_block(layer_id, phys_b, b_offset, k[0, :, t, :], v[0, :, t, :])

        # 4. Repeat KV heads for GQA
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # 5. Attention with scaled dot product (F.scaled_dot_product_attention avoids FP16 overflow)
        is_causal = (q_len > 1 and not is_decode)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache=None,
        layer_id: int = 0,
        block_table: Optional[List[int]] = None,
        num_cached_tokens: int = 0,
    ) -> torch.Tensor:
        h = x + self.self_attn(
            self.input_layernorm(x),
            cos, sin, kv_cache, layer_id, block_table, num_cached_tokens,
        )
        return h + self.mlp(self.post_attention_layernorm(h))
