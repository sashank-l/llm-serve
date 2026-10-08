"""
bench/bench_memory_waste.py
Quantitative evaluation of KV cache memory waste:
Paged KV Cache vs Contiguous KV Cache allocation.
"""
import math
import random


def evaluate_memory_waste(
    num_requests: int = 128,
    max_context_len: int = 2048,
    block_size: int = 16,
    num_layers: int = 28,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    dtype_bytes: int = 2,  # FP16
):
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes  # K and V

    # Simulate realistic chatbot length distribution (lognormal distribution, mean ~250 tokens)
    random.seed(42)
    seq_lengths = [
        min(max_context_len, max(32, int(random.lognormvariate(5.5, 0.7))))
        for _ in range(num_requests)
    ]

    total_actual_tokens = sum(seq_lengths)
    actual_memory_mb = (total_actual_tokens * bytes_per_token) / (1024 * 1024)

    # 1. Contiguous Allocation:
    # Must pre-allocate maximum potential sequence length (max_context_len) for every active slot
    contiguous_allocated_tokens = num_requests * max_context_len
    contiguous_memory_mb = (contiguous_allocated_tokens * bytes_per_token) / (1024 * 1024)
    contiguous_wasted_tokens = contiguous_allocated_tokens - total_actual_tokens
    contiguous_waste_percent = (contiguous_wasted_tokens / contiguous_allocated_tokens) * 100

    # 2. Paged KV Cache Allocation:
    # Allocates fixed 16-token physical blocks on-demand. Waste is only within the last block of each sequence
    paged_blocks = [math.ceil(l / block_size) for l in seq_lengths]
    total_paged_blocks = sum(paged_blocks)
    paged_allocated_tokens = total_paged_blocks * block_size
    paged_memory_mb = (paged_allocated_tokens * bytes_per_token) / (1024 * 1024)
    paged_wasted_tokens = paged_allocated_tokens - total_actual_tokens
    paged_waste_percent = (paged_wasted_tokens / paged_allocated_tokens) * 100

    print("=" * 70)
    print("KV CACHE MEMORY CONSUMPTION & FRAGMENTATION EVALUATION")
    print("=" * 70)
    print(f"Concurrent requests:      {num_requests}")
    print(f"Max context length limit: {max_context_len} tokens")
    print(f"Paged block size:         {block_size} tokens")
    print(f"Model parameters:         {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head dim (FP16)")
    print(f"KV memory per token:      {bytes_per_token} bytes ({bytes_per_token / 1024:.2f} KB)")
    print(f"Total actual tokens:      {total_actual_tokens:,} tokens (Avg: {total_actual_tokens / num_requests:.1f} tokens/seq)")
    print(f"True useful KV memory:    {actual_memory_mb:.2f} MB\n")

    print(f"{'Allocation Strategy':<25} | {'Allocated (MB)':<15} | {'Wasted (MB)':<12} | {'Waste (%)':<10}")
    print("-" * 70)
    print(f"{'Contiguous (Pre-alloc)':<25} | {contiguous_memory_mb:<15.2f} | {(contiguous_memory_mb - actual_memory_mb):<12.2f} | {contiguous_waste_percent:<10.1f}%")
    print(f"{'Paged (On-demand Blocks)':<25} | {paged_memory_mb:<15.2f} | {(paged_memory_mb - actual_memory_mb):<12.2f} | {paged_waste_percent:<10.1f}%")
    print("-" * 70)

    memory_reduction = ((contiguous_memory_mb - paged_memory_mb) / contiguous_memory_mb) * 100
    effective_concurrency = contiguous_allocated_tokens / paged_allocated_tokens
    print(f"Memory Footprint Reduction:   {memory_reduction:.1f}% less memory needed!")
    print(f"Effective Concurrency Boost:  {effective_concurrency:.2f}x more concurrent requests in same VRAM")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_memory_waste()
