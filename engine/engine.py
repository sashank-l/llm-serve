"""
engine/engine.py
LLMEngine: the central inference loop.
"""
import asyncio
import logging
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..model.loader import load_model
from .block_alloc import BlockAllocator
from .kv_cache import PagedKVCache
from .request import Request, RequestStatus, SamplingParams
from .sampler import Sampler
from .scheduler import Scheduler

log = logging.getLogger(__name__)


class LLMEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        block_size: int = 16,
        gpu_memory_fraction: float = 0.4,
        max_num_batched_tokens: int = 2048,
    ):
        self.device = device
        self.block_size = block_size
        self.model_path = model_path

        log.info("Loading tokenizer from %s …", model_path)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        log.info("Loading model from %s …", model_path)
        self.model = load_model(model_path, device)
        config = self.model.config

        num_blocks = self._estimate_num_blocks(config, block_size, gpu_memory_fraction)
        log.info("Pre-allocating %d KV-cache blocks (block_size=%d)", num_blocks, block_size)

        head_dim = config.hidden_size // config.num_attention_heads
        self.block_allocator = BlockAllocator(num_blocks, block_size)
        self.kv_cache = PagedKVCache(
            num_layers   = config.num_hidden_layers,
            num_blocks   = num_blocks,
            block_size   = block_size,
            num_kv_heads = config.num_key_value_heads,
            head_dim     = head_dim,
            device       = device,
        )
        self.scheduler = Scheduler(self.block_allocator, block_size, max_num_batched_tokens)
        self.sampler   = Sampler()

        self._loop_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None

        log.info("LLMEngine ready.")

    def _estimate_num_blocks(self, config, block_size: int, fraction: float) -> int:
        if self.device == "cpu" or not torch.cuda.is_available():
            return 512

        head_dim = config.hidden_size // config.num_attention_heads
        bytes_per_block = (
            2
            * config.num_hidden_layers
            * block_size
            * config.num_key_value_heads
            * head_dim
            * 2  # fp16
        )

        torch.cuda.empty_cache()
        free_mem, total_mem = torch.cuda.mem_get_info()
        usable = int(free_mem * fraction)
        # Cap safely (e.g. 8192 blocks = 131k tokens, ~1.8GB)
        num_blocks = min(8192, max(64, usable // bytes_per_block))
        return num_blocks

    def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        output_queue=None,
    ) -> Request:
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        req = Request(
            request_id      = request_id,
            prompt_tokens   = prompt_tokens,
            sampling_params = sampling_params,
            output_queue    = output_queue,
        )
        self.scheduler.add_request(req)
        return req

    def step(self) -> List[Tuple[str, int, bool]]:
        prefill_batch, decode_batch = self.scheduler.schedule()
        if not prefill_batch and not decode_batch:
            return []

        results: List[Tuple[str, int, bool]] = []

        with torch.no_grad():
            for req in prefill_batch:
                tok_id, finished = self._run_prefill(req)
                self._handle_new_token(req, tok_id, finished)
                results.append((req.request_id, tok_id, finished))

            for req in decode_batch:
                tok_id, finished = self._run_decode(req)
                self._handle_new_token(req, tok_id, finished)
                results.append((req.request_id, tok_id, finished))

        return results

    def generate(self, prompt: str, sampling_params: SamplingParams) -> str:
        req_id = f"sync-{uuid.uuid4().hex[:8]}"
        req    = self.add_request(req_id, prompt, sampling_params)
        while req.status != RequestStatus.FINISHED:
            self.step()
        return self.tokenizer.decode(req.generated_tokens, skip_special_tokens=True)

    def start_background_loop(self, event_loop: asyncio.AbstractEventLoop):
        self.event_loop = event_loop
        self._stop_event.clear()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True, name="engine-loop")
        self._loop_thread.start()
        log.info("Engine background loop started.")

    def stop_background_loop(self):
        self._stop_event.set()
        if self._loop_thread:
            self._loop_thread.join(timeout=5)

    def _run_loop(self):
        while not self._stop_event.is_set():
            if self.scheduler.has_work():
                self.step()
            else:
                time.sleep(0.002)

    def _run_prefill(self, req: Request) -> Tuple[int, bool]:
        input_ids = torch.tensor([req.prompt_tokens], device=self.device)
        positions  = torch.arange(len(req.prompt_tokens), device=self.device).unsqueeze(0)

        logits = self.model(
            input_ids,
            positions,
            kv_cache          = self.kv_cache,
            block_table       = req.block_table,
            num_cached_tokens = 0,
        )
        next_token = self.sampler.sample(logits[0, -1], req.sampling_params)
        req.num_cached_tokens = len(req.prompt_tokens)
        req.first_token_time  = time.time()
        finished = self._is_finished(req, next_token)
        return next_token, finished

    def _run_decode(self, req: Request) -> Tuple[int, bool]:
        last_token = req.generated_tokens[-1]
        input_ids  = torch.tensor([[last_token]], device=self.device)
        positions  = torch.tensor([[req.num_cached_tokens]], device=self.device)

        logits = self.model(
            input_ids,
            positions,
            kv_cache          = self.kv_cache,
            block_table       = req.block_table,
            num_cached_tokens = req.num_cached_tokens,
        )
        next_token = self.sampler.sample(logits[0, -1], req.sampling_params)
        req.num_cached_tokens += 1
        finished = self._is_finished(req, next_token)
        return next_token, finished

    def _is_finished(self, req: Request, token_id: int) -> bool:
        eos = self.tokenizer.eos_token_id
        if token_id == eos:
            return True
        if len(req.generated_tokens) + 1 >= req.sampling_params.max_new_tokens:
            return True
        if req.sampling_params.stop_strings:
            generated_so_far = self.tokenizer.decode(
                req.generated_tokens + [token_id], skip_special_tokens=True
            )
            for stop in req.sampling_params.stop_strings:
                if stop in generated_so_far:
                    return True
        return False

    def _handle_new_token(self, req: Request, token_id: int, finished: bool):
        req.generated_tokens.append(token_id)
        if req.output_queue is not None and self.event_loop is not None:
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            asyncio.run_coroutine_threadsafe(
                req.output_queue.put((token_text, finished)),
                self.event_loop,
            )
        if finished:
            req.finish_time = time.time()
            self.scheduler.finish_request(req)

    def abort_request(self, request_id: str):
        """Abort request and immediately free all allocated blocks."""
        self.scheduler.abort_request(request_id)
