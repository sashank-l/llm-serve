"""
tests/test_sampling_reproducibility.py
Verify sampling reproducibility:
  1. Seeded runs with temperature > 0 produce identical token sequences.
  2. Runs with different seeds produce divergent token sequences.
  3. Greedy (temperature = 0) is deterministic.
"""
import pytest
import torch
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mini_engine.engine.sampler import Sampler
from mini_engine.engine.request import SamplingParams


def test_seeded_sampling_reproducibility():
    sampler = Sampler()
    vocab_size = 1000
    # Use non-uniform logits
    torch.manual_seed(1234)
    logits = torch.randn(vocab_size)

    # 1. Two runs with the exact same seed=42
    params_seed_a = SamplingParams(temperature=0.8, top_p=0.9, seed=42)
    params_seed_b = SamplingParams(temperature=0.8, top_p=0.9, seed=42)

    tokens_a = [sampler.sample(logits, params_seed_a) for _ in range(20)]
    tokens_b = [sampler.sample(logits, params_seed_b) for _ in range(20)]

    print(f"\nSeeded Run A (seed=42): {tokens_a[:10]}")
    print(f"Seeded Run B (seed=42): {tokens_b[:10]}")
    assert tokens_a == tokens_b, "Identical seeds must yield identical sampling outputs!"

    # 2. Run with different seed=999
    params_seed_c = SamplingParams(temperature=0.8, top_p=0.9, seed=999)
    tokens_c = [sampler.sample(logits, params_seed_c) for _ in range(20)]
    print(f"Seeded Run C (seed=999): {tokens_c[:10]}")
    assert tokens_a != tokens_c, "Different seeds should produce divergent outputs!"


def test_greedy_determinism():
    sampler = Sampler()
    logits = torch.tensor([1.2, 5.4, 2.1, 0.5])
    params = SamplingParams(temperature=0.0)
    samples = [sampler.sample(logits, params) for _ in range(10)]
    assert all(s == 1 for s in samples), "Greedy sampling must always pick argmax!"
