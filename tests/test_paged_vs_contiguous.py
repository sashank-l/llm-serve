"""
tests/test_paged_vs_contiguous.py
Verify mathematical and generation parity between:
  1. Contiguous KV cache (unpaged, single continuous tensor per request)
  2. Paged KV cache (16-token physical blocks gathered on demand)
"""
import pytest
import torch
import torch.nn.functional as F
import math


@pytest.fixture(scope="session")
def model_path():
    return "/home/surya/models/qwen2.5-1.5b"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_paged_vs_contiguous_kv_cache(model_path):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from transformers import AutoTokenizer
    from mini_engine.engine.engine import LLMEngine
    from mini_engine.engine.request import SamplingParams
    from mini_engine.model.layers import apply_rotary_emb, repeat_kv

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    engine = LLMEngine(model_path, device="cuda", block_size=16, gpu_memory_fraction=0.3)
    model = engine.model

    prompt = "Artificial intelligence and machine learning algorithms"
    input_ids = tok.encode(prompt, return_tensors="pt").to("cuda")
    seq_len = input_ids.shape[1]

    # --- 1. Generate 10 tokens using our Paged KV cache engine ---
    paged_params = SamplingParams(temperature=0.0, max_new_tokens=10)
    paged_output = engine.generate(prompt, paged_params)
    paged_tokens = tok.encode(paged_output)

    # --- 2. Step-by-step contiguous decode simulation ---
    # In contiguous decoding, we keep all previous tokens in a contiguous tensor and compute full causal attention
    curr_ids = input_ids.clone()
    contiguous_tokens = []

    with torch.no_grad():
        for _ in range(10):
            q_len = curr_ids.shape[1]
            pos = torch.arange(q_len, device="cuda").unsqueeze(0)
            logits = model(curr_ids, pos)
            next_token = logits[0, -1].argmax().item()
            contiguous_tokens.append(next_token)
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]], device="cuda")], dim=1)

    contiguous_output = tok.decode(contiguous_tokens)

    print("\n--- Paged vs Contiguous KV Cache Evaluation ---")
    print(f"Prompt: {prompt!r}")
    print(f"Paged KV Cache Output:      {paged_output!r}")
    print(f"Contiguous KV Cache Output: {contiguous_output!r}")
    print(f"Paged Token IDs:      {paged_tokens}")
    print(f"Contiguous Token IDs: {contiguous_tokens}")

    assert paged_tokens == contiguous_tokens, (
        f"Paged vs Contiguous mismatch!\nPaged: {paged_tokens}\nContiguous: {contiguous_tokens}"
    )
    print("PARITY: Paged KV cache and Contiguous KV cache are 100% IDENTICAL!")
