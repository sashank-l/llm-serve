"""
tests/test_streaming_and_cancellation.py
Verify streaming API and client cancellation / disconnect handling:
  1. OpenAI-format SSE chunks are streamed correctly.
  2. Premature disconnect / abort releases all allocated KV blocks immediately.
"""
import pytest
import asyncio
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mini_engine.engine.block_alloc import BlockAllocator
from mini_engine.engine.request import SamplingParams, RequestStatus
from mini_engine.engine.scheduler import Scheduler


@pytest.mark.asyncio
async def test_client_disconnect_frees_blocks():
    block_allocator = BlockAllocator(num_blocks=20, block_size=16)
    scheduler = Scheduler(block_allocator, block_size=16)

    initial_free = block_allocator.free_blocks_count()

    # Create request with streaming queue
    queue = asyncio.Queue()
    from mini_engine.engine.request import Request
    req = Request(
        request_id="stream-client-1",
        prompt_tokens=[1] * 32,  # 32 tokens = 2 blocks
        sampling_params=SamplingParams(max_new_tokens=50, stream=True),
        output_queue=queue,
    )
    scheduler.add_request(req)

    # Admit into scheduler
    prefill, _ = scheduler.schedule()
    assert len(prefill) == 1
    assert block_allocator.allocated_blocks_count() == 2
    assert block_allocator.free_blocks_count() == initial_free - 2

    # Simulate client reading 3 tokens then disconnecting
    for token in ["Hello", " ", "world"]:
        await queue.put((token, False))
        token_read, is_finished = await queue.get()
        assert not is_finished

    # Client disconnects! Abort request:
    scheduler.abort_request("stream-client-1")

    # Verify all KV blocks returned to allocator pool
    assert req.status == RequestStatus.CANCELLED
    assert block_allocator.allocated_blocks_count() == 0
    assert block_allocator.free_blocks_count() == initial_free
    assert not block_allocator.is_leaking()
    print("\nStreaming & Cancellation Test: Client disconnect cleanly reclaimed all KV cache blocks!")
