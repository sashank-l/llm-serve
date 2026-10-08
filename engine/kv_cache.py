"""
engine/kv_cache.py
Paged KV cache: pre-allocated physical memory blocks on GPU.
"""
from typing import List, Tuple
import torch


class PagedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device

        self.cache_k = torch.zeros(
            (num_layers, num_blocks, block_size, num_kv_heads, head_dim),
            dtype=torch.float16,
            device=device,
        )
        self.cache_v = torch.zeros(
            (num_layers, num_blocks, block_size, num_kv_heads, head_dim),
            dtype=torch.float16,
            device=device,
        )

    def write_to_block(
        self,
        layer_id: int,
        block_id: int,
        offset: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Write key and value states into a physical cache block at offset.
        k, v can be (num_kv_heads, head_dim) for single token
        or (n_tokens, num_kv_heads, head_dim) for multiple tokens.
        """
        if k.dim() == 2:
            self.cache_k[layer_id, block_id, offset] = k
            self.cache_v[layer_id, block_id, offset] = v
        else:
            n = k.shape[0]
            self.cache_k[layer_id, block_id, offset : offset + n] = k
            self.cache_v[layer_id, block_id, offset : offset + n] = v

    def gather_kv(
        self, layer_id: int, block_table: List[int], seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather full contiguous key and value sequence history from physical blocks.
        Returns: (seq_len, num_kv_heads, head_dim) tensors.
        """
        k_blocks = [self.cache_k[layer_id, b] for b in block_table]
        v_blocks = [self.cache_v[layer_id, b] for b in block_table]

        if not k_blocks:
            empty = torch.empty(0, self.num_kv_heads, self.head_dim, device=self.device, dtype=torch.float16)
            return empty, empty

        k_gathered = torch.cat(k_blocks, dim=0)[:seq_len]
        v_gathered = torch.cat(v_blocks, dim=0)[:seq_len]
        return k_gathered, v_gathered
