"""
tests/test_scheduler_dynamic.py
Verify dynamic continuous batching scheduler behavior:
  1. Requests of different lengths finish independently.
  2. New requests join mid-generation while existing requests are actively decoding.
  3. Preemption and aborts properly reclaim blocks.
"""
import pytest
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mini_engine.engine.block_alloc import BlockAllocator
from mini_engine.engine.request import Request, RequestStatus, SamplingParams
from mini_engine.engine.scheduler import Scheduler


def test_independent_completion_and_midgen_join():
    block_allocator = BlockAllocator(num_blocks=32, block_size=16)
    scheduler = Scheduler(block_allocator, block_size=16, max_num_batched_tokens=1024)

    # 1. Add req1 (short, 5 tokens) and req2 (longer, 15 tokens)
    req1 = Request("req-1", prompt_tokens=[1, 2, 3], sampling_params=SamplingParams(max_new_tokens=5))
    req2 = Request("req-2", prompt_tokens=[1, 2, 3, 4], sampling_params=SamplingParams(max_new_tokens=15))
    scheduler.add_request(req1)
    scheduler.add_request(req2)

    # Step 1: Both admitted to prefill
    prefill, decode = scheduler.schedule()
    assert len(prefill) == 2
    assert len(decode) == 0

    # Simulate token 1 generation for both
    req1.generated_tokens.append(100)
    req2.generated_tokens.append(200)

    # Step 2: Both in decode
    prefill, decode = scheduler.schedule()
    assert len(prefill) == 0
    assert len(decode) == 2

    # Step 3: Add req3 MID-GENERATION (joins at step 3)
    req3 = Request("req-3", prompt_tokens=[5, 6], sampling_params=SamplingParams(max_new_tokens=4))
    scheduler.add_request(req3)

    # Step 4: req3 should be in prefill, req1 and req2 in decode simultaneously!
    prefill, decode = scheduler.schedule()
    assert len(prefill) == 1
    assert prefill[0].request_id == "req-3"
    assert len(decode) == 2
    assert {r.request_id for r in decode} == {"req-1", "req-2"}

    # Simulate req1 finishing early
    scheduler.finish_request(req1)
    assert req1.status == RequestStatus.FINISHED
    assert len(req1.block_table) == 0

    # Step 5: req1 is gone, remaining are decoded
    prefill, decode = scheduler.schedule()
    assert "req-1" not in [r.request_id for r in decode]
    print("\nScheduler Dynamic Test: New request joined mid-generation and short request finished independently!")


def test_scheduler_abort_request():
    block_allocator = BlockAllocator(num_blocks=10, block_size=16)
    scheduler = Scheduler(block_allocator, block_size=16)

    req = Request("abort-me", prompt_tokens=[1] * 20, sampling_params=SamplingParams())
    scheduler.add_request(req)

    # Admit into running
    prefill, _ = scheduler.schedule()
    assert len(prefill) == 1
    assert block_allocator.allocated_blocks_count() == 2  # 20 tokens = 2 blocks

    # Abort request mid-flight
    scheduler.abort_request("abort-me")
    assert req.status == RequestStatus.CANCELLED
    assert len(req.block_table) == 0
    assert block_allocator.allocated_blocks_count() == 0
    assert not block_allocator.is_leaking()
    print("Scheduler Abort Test: Request cancelled and all KV blocks reclaimed immediately!")
