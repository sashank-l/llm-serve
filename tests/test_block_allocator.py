"""
tests/test_block_allocator.py
Unit tests for BlockAllocator:
  1. Allocation and deallocation tracking
  2. Double-free detection (raises ValueError)
  3. Out-of-bounds free detection (raises ValueError)
  4. Out-of-memory handling (raises RuntimeError when pool is exhausted)
  5. Leak detection across thousands of random allocation/free cycles
"""
import pytest
import random
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mini_engine.engine.block_alloc import BlockAllocator


def test_basic_allocation_and_free():
    alloc = BlockAllocator(num_blocks=10, block_size=16)
    assert alloc.free_blocks_count() == 10
    assert alloc.allocated_blocks_count() == 0

    b1 = alloc.allocate()
    b2 = alloc.allocate()
    assert alloc.free_blocks_count() == 8
    assert alloc.allocated_blocks_count() == 2

    alloc.free(b1)
    assert alloc.free_blocks_count() == 9
    assert alloc.allocated_blocks_count() == 1

    alloc.free(b2)
    assert alloc.free_blocks_count() == 10
    assert alloc.allocated_blocks_count() == 0
    assert not alloc.is_leaking()


def test_double_free_detection():
    alloc = BlockAllocator(num_blocks=5, block_size=16)
    b0 = alloc.allocate()
    alloc.free(b0)

    # Double freeing the same block MUST raise ValueError
    with pytest.raises(ValueError, match="Double free detected"):
        alloc.free(b0)

    # Freeing a block that was never allocated MUST raise ValueError
    with pytest.raises(ValueError, match="Double free detected"):
        alloc.free(1)


def test_out_of_bounds_free():
    alloc = BlockAllocator(num_blocks=5, block_size=16)
    with pytest.raises(ValueError, match="out of bounds"):
        alloc.free(-1)
    with pytest.raises(ValueError, match="out of bounds"):
        alloc.free(100)


def test_oom_exhaustion():
    alloc = BlockAllocator(num_blocks=3, block_size=16)
    alloc.allocate()
    alloc.allocate()
    alloc.allocate()
    assert alloc.free_blocks_count() == 0
    assert not alloc.can_allocate(1)

    with pytest.raises(RuntimeError, match="Out of memory"):
        alloc.allocate()


def test_no_leak_stress_test():
    total_blocks = 64
    alloc = BlockAllocator(num_blocks=total_blocks, block_size=16)
    held_blocks = []

    # Run 5,000 randomized alloc/free cycles
    for _ in range(5000):
        if alloc.can_allocate(1) and (not held_blocks or random.random() < 0.6):
            b = alloc.allocate()
            held_blocks.append(b)
        elif held_blocks:
            idx = random.randrange(len(held_blocks))
            b = held_blocks.pop(idx)
            alloc.free(b)

    # Clean up all remaining
    for b in held_blocks:
        alloc.free(b)

    assert alloc.free_blocks_count() == total_blocks
    assert alloc.allocated_blocks_count() == 0
    assert not alloc.is_leaking()
    print("\nBlockAllocator Stress Test: 5,000 cycles completed with ZERO leaks!")
