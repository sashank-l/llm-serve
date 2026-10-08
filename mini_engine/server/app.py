"""
server/app.py
FastAPI OpenAI-compatible inference server.

Endpoints:
  POST /v1/chat/completions   – streaming (SSE) and non-streaming
  GET  /v1/models             – list available models
  GET  /health                – liveness check
  GET  /metrics               – basic stats

Streaming uses asyncio.Queue per request.  The engine background thread
pushes (token_text, finished) tuples; the async generator yields them as
SSE data frames.
"""
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request as StarletteRequest
from fastapi.responses import StreamingResponse

from ..engine.engine import LLMEngine
from ..engine.request import SamplingParams
from .schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_engine: Optional[LLMEngine] = None
_model_id: str = "unknown"


def init_engine(model_path: str, device: str = "cuda"):
    """Call this before starting uvicorn."""
    global _engine, _model_id
    _model_id = model_path.rstrip("/").split("/")[-1]
    _engine = LLMEngine(model_path, device)


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start engine background loop when the server starts
    if _engine is not None:
        loop = asyncio.get_event_loop()
        _engine.start_background_loop(loop)
    yield
    # Shutdown
    if _engine is not None:
        _engine.stop_background_loop()


app = FastAPI(title="Mini LLM Inference Engine", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health / Models
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model": _model_id}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": _model_id, "object": "model", "created": 0, "owned_by": "user"}],
    }


@app.get("/metrics")
async def metrics():
    if _engine is None:
        raise HTTPException(503, "Engine not initialised")
    sched = _engine.scheduler
    return {
        "waiting":  len(sched.waiting),
        "running":  len(sched.running),
        "free_blocks": _engine.block_allocator.free_blocks_count(),
    }


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, http_req: StarletteRequest):
    if _engine is None:
        raise HTTPException(503, "Engine not initialised")

    # Build prompt string (Qwen / ChatML template)
    prompt = _build_prompt(req, _engine)

    stop_strings = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
    default_stops = ["<|im_end|>", "<|endoftext|>"]
    for s in default_stops:
        if s not in stop_strings:
            stop_strings.append(s)

    sampling_params = SamplingParams(
        temperature    = req.temperature if req.temperature is not None else 1.0,
        top_p          = req.top_p if req.top_p is not None else 1.0,
        top_k          = req.top_k or -1,
        max_new_tokens = req.max_tokens or 256,
        stop_strings   = stop_strings,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    # ---- Streaming -------------------------------------------------------
    if req.stream:
        queue: asyncio.Queue = asyncio.Queue()
        _engine.add_request(request_id, prompt, sampling_params, output_queue=queue)

        async def sse_generator() -> AsyncGenerator[str, None]:
            created = int(time.time())
            # First chunk: role delta
            first = ChatCompletionChunk(
                id      = request_id,
                created = created,
                model   = _model_id,
                choices = [ChatCompletionChunkChoice(
                    index         = 0,
                    delta         = {"role": "assistant"},
                    finish_reason = None,
                )],
            )
            yield f"data: {first.model_dump_json()}\n\n"

            while True:
                if await http_req.is_disconnected():
                    _engine.abort_request(request_id)
                    break

                try:
                    token_text, finished = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    _engine.abort_request(request_id)
                    break

                if any(s in token_text for s in default_stops):
                    token_text = ""
                    finished = True

                if token_text or finished:
                    chunk = ChatCompletionChunk(
                        id      = request_id,
                        created = created,
                        model   = _model_id,
                        choices = [ChatCompletionChunkChoice(
                            index         = 0,
                            delta         = {"content": token_text} if token_text else {},
                            finish_reason = "stop" if finished else None,
                        )],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                if finished:
                    break

            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # ---- Non-streaming ---------------------------------------------------
    queue = asyncio.Queue()
    _engine.add_request(request_id, prompt, sampling_params, output_queue=queue)

    tokens: list[str] = []
    while True:
        token_text, finished = await asyncio.wait_for(queue.get(), timeout=120.0)
        tokens.append(token_text)
        if finished:
            break

    full_text = "".join(tokens)
    for s in default_stops:
        full_text = full_text.replace(s, "")
    full_text = full_text.strip()

    return ChatCompletionResponse(
        id      = request_id,
        created = int(time.time()),
        model   = _model_id,
        choices = [ChatCompletionChoice(
            index         = 0,
            message       = ChatMessage(role="assistant", content=full_text),
            finish_reason = "stop",
        )],
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(req: ChatCompletionRequest, engine: LLMEngine) -> str:
    """Try tokenizer.apply_chat_template; fall back to ChatML manually."""
    try:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        return engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Manual ChatML fallback
        parts = []
        for m in req.messages:
            parts.append(f"<|im_start|>{m.role}\n{m.content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
