# Mini LLM Inference Engine

A high-performance LLM serving engine built from scratch in PyTorch featuring:
- **Custom LLaMA / Qwen2.5 Architecture**: RMSNorm, HuggingFace-compatible Rotary Position Embeddings (RoPE), Grouped-Query Attention (GQA), and SwiGLU Feed-Forward Networks.
- **HuggingFace Weights Loader**: Native safetensors loader supporting tied embeddings, projection biases, and shard concatenation.
- **Iteration-Level Continuous Batching**: Dynamic FIFO scheduler that concurrently batches prefill and decode iterations across heterogeneous request lengths.
- **Paged KV Cache**: Fixed-block GPU memory allocator (block size = 16) eliminating fragmentation with logical-to-physical block mapping tables.
- **OpenAI-Compatible Serving**: FastAPI server with real-time SSE token streaming (`/v1/chat/completions`), metrics, and health endpoints.
- **Verified HF Greedy Parity**: Matches HuggingFace `generate` greedy outputs.

---

## 🏗️ Architecture

```
                       ┌────────────────────────┐
                       │   Client (cURL / SDK)  │
                       └───────────┬────────────┘
                                   │ HTTP / SSE
                                   ▼
                       ┌────────────────────────┐
                       │   FastAPI Server       │
                       │ (/v1/chat/completions) │
                       └───────────┬────────────┘
                                   │ asyncio.Queue
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                           LLMEngine                             │
│                                                                 │
│  ┌──────────────────────┐             ┌──────────────────────┐  │
│  │      Scheduler       │             │    BlockAllocator    │  │
│  │ (Waiting / Running)  ├────────────►│     (Free List)      │  │
│  └──────────┬───────────┘             └──────────┬───────────┘  │
│             │                                    │              │
│      schedule() batch                            │ block table  │
│             │                                    │              │
│             ▼                                    ▼              │
│  ┌──────────────────────┐             ┌──────────────────────┐  │
│  │    Prefill/Decode    │             │     PagedKVCache     │  │
│  │     Forward Pass     │◄────────────┤ [L, B, 16, Hkv, D]   │  │
│  └──────────┬───────────┘             └──────────────────────┘  │
│             │ logits                                            │
│             ▼                                                   │
│  ┌──────────────────────┐                                       │
│  │       Sampler        │                                       │
│  │(Greedy/Temp/TopK/TopP│                                       │
│  └──────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
mini-engine/
├── model/
│   ├── layers.py          # RMSNorm, RoPE, GroupedQueryAttention (GQA), SwiGLU
│   ├── llama.py           # Full LlamaModel & LlamaConfig
│   └── loader.py          # Safetensors weight loader & tied embedding handler
├── engine/
│   ├── block_alloc.py     # Paged KV cache block allocator
│   ├── kv_cache.py        # Paged KV cache tensor storage
│   ├── request.py         # Request, SamplingParams, RequestStatus
│   ├── scheduler.py       # Iteration-level continuous batching scheduler
│   ├── sampler.py         # Greedy, temperature, top-k, and top-p sampling
│   └── engine.py          # Core LLMEngine inference loop & background thread
├── server/
│   ├── app.py             # FastAPI server with SSE streaming
│   └── schemas.py         # OpenAI-compatible Pydantic request/response schemas
├── tests/
│   ├── conftest.py        # Pytest CLI configuration
│   ├── test_parity.py     # Greedy parity test against Hugging Face baseline
│   └── test_batching.py   # Continuous batching correctness test
├── bench/
│   └── bench_throughput.py# Concurrent request throughput benchmark
├── download_model.py      # HuggingFace snapshot downloader
├── run_server.py          # Server entrypoint
├── requirements.txt
└── setup.py
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone <repo>
cd mini-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Download Model

Download `Qwen/Qwen2.5-1.5B-Instruct` (or any Llama-style model):

```bash
python download_model.py --model Qwen/Qwen2.5-1.5B-Instruct --dir ~/models/qwen2.5-1.5b
```

### 3. Start the Server

```bash
python run_server.py --model ~/models/qwen2.5-1.5b --device cuda --port 8000
```

---

## 📡 API Usage

### Health & Models
```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

### Streaming Chat Completion (SSE)
```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-instruct",
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "stream": true,
    "temperature": 0.0,
    "max_tokens": 50
  }'
```

### Non-Streaming Chat Completion
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": false,
    "max_tokens": 50
  }'
```

---

## 🧪 Evaluation & Checkpoints

### Phase 1 Checkpoint: Hugging Face Parity
```bash
pytest tests/test_parity.py -v -s --model-path ~/models/qwen2.5-1.5b
```
- Token logits match Hugging Face within $0.03$ FP16 tolerance.
- Greedy outputs match Hugging Face token-for-token.

### Phase 2 Checkpoint: Continuous Batching
```bash
pytest tests/test_batching.py -v -s
```
- Verifies requests of disparate token budgets finish in independent iteration steps without blocking each other.
- Verifies batched outputs remain bit-identical to isolated single-request generations.

### Phase 6: Throughput Benchmark
```bash
python bench/bench_throughput.py --num-requests 5 --tokens-per-request 20
```
**Results on NVIDIA RTX A6000 (48GB VRAM):**
- **Throughput**: ~66.0 tokens / sec
- **Step Latency**: ~75.9 ms / step
- **KV Blocks Allocated**: 8,192 blocks (131,072 cached tokens capacity)
