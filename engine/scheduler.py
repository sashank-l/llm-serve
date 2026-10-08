"""
engine/scheduler.py
Continuous-batching scheduler with:
  - Waiting queue  → new requests not yet started
  - Running set    → requests mid-generation
  - Block-aware admission control (only admit if enough KV blocks exist)
  - Greedy block allocation for both prefill and decode token expansions
  - Simple preemption: evict last running request if a decode step can't get a block
"""
import logging
from collections import deque
from typing import Deque, List, Tuple

from .block_alloc import BlockAllocator
from .request import Request, RequestStatus

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(
        self,
        block_allocator: BlockAllocator,
        block_size: int,
        max_num_batched_tokens: int = 2048,
    ):
        self.block_allocator       = block_allocator
        self.block_size            = block_size
        self.max_num_batched_tokens = max_num_batched_tokens

        self.waiting: Deque[Request] = deque()
        self.running: List[Request]  = []

    # -----------------------------------------------------------------------

    def add_request(self, req: Request):
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # -----------------------------------------------------------------------

    def schedule(self) -> Tuple[List[Request], List[Request]]:
        """
        Returns (prefill_batch, decode_batch).

        Prefill batch: waiting requests that have been admitted (blocks allocated).
        Decode batch:  already-running requests that need one more token (and block).
        """
        prefill_batch: List[Request] = []
        decode_batch:  List[Request] = []
        batched_tokens = 0

        # --- Admit waiting requests into prefill --------------------------
        temp_waiting: Deque[Request] = deque()
        while self.waiting:
            req = self.waiting.popleft()
            prompt_len        = len(req.prompt_tokens)
            blocks_needed     = _blocks_for(prompt_len, self.block_size)
            budget_ok         = (batched_tokens + prompt_len) <= self.max_num_batched_tokens

            if budget_ok and self.block_allocator.can_allocate(blocks_needed):
                for _ in range(blocks_needed):
                    req.block_table.append(self.block_allocator.allocate())
                req.status = RequestStatus.RUNNING
                self.running.append(req)
                prefill_batch.append(req)
                batched_tokens += prompt_len
            else:
                # Re-queue and stop admitting (FIFO order preserved)
                temp_waiting.append(req)
                # Drain the rest back without trying
                temp_waiting.extend(self.waiting)
                self.waiting.clear()
                break

        self.waiting = temp_waiting

        # --- Decode pass for already-running requests ---------------------
        to_preempt: List[Request] = []
        for req in self.running:
            if req in prefill_batch:
                continue  # handled above

            total_tokens = len(req.prompt_tokens) + len(req.generated_tokens) + 1
            # Do we need a new block for the next token?
            if total_tokens % self.block_size == 1:
                # Current last slot just rolled into a new block
                if self.block_allocator.can_allocate(1):
                    req.block_table.append(self.block_allocator.allocate())
                else:
                    to_preempt.append(req)
                    continue

            decode_batch.append(req)

        # --- Preempt if needed -------------------------------------------
        for req in to_preempt:
            log.warning("Preempting request %s (OOM on KV blocks).", req.request_id)
            self.running.remove(req)
            # Free all blocks
            for block_id in req.block_table:
                self.block_allocator.free(block_id)
            req.block_table.clear()
            req.num_cached_tokens = 0
            req.generated_tokens.clear()
            req.status = RequestStatus.WAITING
            self.waiting.appendleft(req)

        return prefill_batch, decode_batch

    # -----------------------------------------------------------------------

    def finish_request(self, req: Request):
        if req in self.running:
            self.running.remove(req)
        for block_id in req.block_table:
            self.block_allocator.free(block_id)
        req.block_table.clear()
        req.status = RequestStatus.FINISHED

    def abort_request(self, request_id: str):
        """Immediately abort request and reclaim its allocated KV blocks."""
        for req in list(self.waiting):
            if req.request_id == request_id:
                self.waiting.remove(req)
                req.status = RequestStatus.CANCELLED
                log.info("Request %s aborted from waiting queue.", request_id)
                return

        for req in list(self.running):
            if req.request_id == request_id:
                self.running.remove(req)
                for block_id in req.block_table:
                    self.block_allocator.free(block_id)
                req.block_table.clear()
                req.status = RequestStatus.CANCELLED
                log.info("Request %s aborted from running pool; blocks freed.", request_id)
                return


# ---------------------------------------------------------------------------

def _blocks_for(num_tokens: int, block_size: int) -> int:
    return (num_tokens + block_size - 1) // block_size
