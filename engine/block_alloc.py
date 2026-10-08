"""
engine/block_alloc.py
Paged KV cache block allocator with double-free detection, OOM handling,
and leak detection tracking.
"""
from typing import List, Set


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks: List[int] = list(range(num_blocks))
        self.allocated_blocks: Set[int] = set()

    def allocate(self) -> int:
        if not self.free_blocks:
            raise RuntimeError(f"Out of memory: no free blocks (total: {self.num_blocks})")
        block_id = self.free_blocks.pop()
        self.allocated_blocks.add(block_id)
        return block_id

    def free(self, block_id: int):
        if not (0 <= block_id < self.num_blocks):
            raise ValueError(f"Invalid block_id {block_id}: out of bounds [0, {self.num_blocks})")
        if block_id not in self.allocated_blocks:
            raise ValueError(f"Double free detected: block_id {block_id} is not currently allocated")
        self.allocated_blocks.remove(block_id)
        self.free_blocks.append(block_id)

    def free_blocks_count(self) -> int:
        return len(self.free_blocks)

    def allocated_blocks_count(self) -> int:
        return len(self.allocated_blocks)

    def can_allocate(self, n: int = 1) -> bool:
        return len(self.free_blocks) >= n

    def is_leaking(self) -> bool:
        """Returns True if there is an inconsistency in total block accounting."""
        return (len(self.free_blocks) + len(self.allocated_blocks)) != self.num_blocks
