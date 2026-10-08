"""
model/llama.py
Full LLaMA / Qwen2.5 model (config + forward pass).
"""
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn

from .layers import TransformerBlock, RMSNorm, precompute_freqs


@dataclass
class LlamaConfig:
    vocab_size: int              = 151936
    hidden_size: int             = 1536
    num_hidden_layers: int       = 28
    num_attention_heads: int     = 12
    num_key_value_heads: int     = 2
    intermediate_size: int       = 8960
    max_position_embeddings: int = 32768
    rms_norm_eps: float          = 1e-6
    rope_theta: float            = 1000000.0
    tie_word_embeddings: bool    = True
    attention_bias: bool         = True


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE cos/sin tables
        head_dim = config.hidden_size // config.num_attention_heads
        cos, sin = precompute_freqs(head_dim, config.max_position_embeddings, config.rope_theta)
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,                  # (bsz, seq_len)
        positions: torch.Tensor,                  # (bsz, seq_len) absolute token positions
        kv_cache=None,
        block_table: Optional[List[int]] = None,
        num_cached_tokens: int = 0,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        # Extract cos/sin for these positions: (bsz, 1, seq_len, head_dim)
        cos = self.cos_table[positions].unsqueeze(1)
        sin = self.sin_table[positions].unsqueeze(1)

        for layer_id, layer in enumerate(self.layers):
            x = layer(
                x,
                cos,
                sin,
                kv_cache=kv_cache,
                layer_id=layer_id,
                block_table=block_table,
                num_cached_tokens=num_cached_tokens,
            )

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
