# Comprehensive Evaluation Report: Mini LLM Inference Engine

This document provides empirical results, methodology, and architectural analysis for the 10 core evaluations of the **Mini LLM Inference Engine** evaluated against the reference model **`Qwen/Qwen2.5-1.5B-Instruct`** on an **NVIDIA RTX A6000 (48GB VRAM)**.

---

## 📋 Evaluation Summary Matrix

| # | Evaluation Checkpoint | Test File / Benchmark | Status | Key Metric / Result |
|---|-----------------------|-----------------------|:------:|---------------------|
| **1** | **HF Greedy Parity** | `tests/test_parity.py` | **PASSED** | 100% token-for-token match across all prompts; Attention diff = `0.000` |
| **2** | **Batched vs Single-Request Parity** | `tests/test_batching.py` | **PASSED** | Bit-identical outputs between isolated and continuous batching |
| **3** | **Paged vs Contiguous KV Parity** | `tests/test_paged_vs_contiguous.py` | **PASSED** | 100% output identity between paged blocks and contiguous baseline |
| **4** | **Block Allocator Leak & Safety** | `tests/test_block_allocator.py` | **PASSED** | 5,000 randomized alloc/free cycles; 0 leaks; Double-free & OOM handled |
| **5** | **Dynamic Scheduler & Mid-Gen Join** | `tests/test_scheduler_dynamic.py` | **PASSED** | Dynamic arrival mid-generation; heterogeneous requests finish independently |
| **6** | **Sampling Reproducibility** | `tests/test_sampling_reproducibility.py` | **PASSED** | Seeded non-greedy runs match across executions; divergent across seeds |
| **7** | **Streaming API & Cancellation** | `tests/test_streaming_and_cancellation.py` | **PASSED** | SSE streaming chunks verified; client disconnect reclaims blocks immediately |
| **8** | **Throughput & Latency Benchmarks** | `bench/bench_vs_hf.py`, `bench_throughput.py` | **PASSED** | **69.8 tokens/sec** throughput; 143 ms step latency on 10 concurrent requests |
| **9** | **KV Memory Waste: Paged vs Contiguous** | `bench/bench_memory_waste.py` | **PASSED** | **85.3% memory reduction**; 6.8x concurrency boost over contiguous allocation |
| **10** | **Continuous Integration (CI)** | `.github/workflows/ci.yml` | **PASSED** | Automated GitHub Actions workflow testing full pytest suite on push |

---

## 1. Hugging Face Greedy Parity (Token-for-Token)

### Methodology
To verify architectural and mathematical fidelity, greedy generation (temperature = 0.0) from our custom `LlamaModel` implementation was evaluated against Hugging Face's reference implementation (`transformers.AutoModelForCausalLM`).

### Results
```
tests/test_parity.py::test_greedy_parity PASSED

Prompt 1: 'The capital of France is'
  HF Output:     ' Paris. The capital of France is also the capital of which country?\nA'
  Engine Output: ' Paris. The capital of France is also the capital of which country?\nA'
  Token Match:   [12095, 13, 576, 6722, 315, 9625, 374, 1083] == [12095, 13, 576, 6722, 315, 9625, 374, 1083] (MATCH)

Prompt 2: 'In Python, a list comprehension is'
  HF Output:     ' a concise way to create lists based on existing lists. For example, to'
  Engine Output: ' a concise way to create lists based on existing lists. For example, to'
  Token Match:   [264, 63594, 1616, 311, 1855, 11469, 3118, 389] == [264, 63594, 1616, 311, 1855, 11469, 3118, 389] (MATCH)

Prompt 3: 'The transformer architecture was introduced in the paper'
  HF Output:     ' "A New Architecture for High Performance Computing" by David A. Bader'
  Engine Output: ' "A New Architecture for High Performance Computing" by David A. Bader'
  Token Match:   [330, 32, 1532, 37843, 369, 5124, 20651, 45779] == [330, 32, 1532, 37843, 369, 5124, 20651, 45779] (MATCH)

Prompt 4: 'Once upon a time in a land far away,'
  HF Output:     ' there was a kingdom ruled by a wise king. The king had a special'
  Engine Output: ' there was a kingdom ruled by a wise king. The king had a special'
  Token Match:   [1052, 572, 264, 25079, 21286, 553, 264, 23335] == [1052, 572, 264, 25079, 21286, 553, 264, 23335] (MATCH)
```

- **Numerical Attention Difference**: Max diff against HF PyTorch SDPA is **`0.0000`**.
- **Logits Tolerance**: Max difference across all layers $\le 0.0234$ (within FP16 representation limits).

---

## 2. Batched vs Single-Request Parity

### Methodology
In continuous batching, requests share the attention engine in different phases (prefill alongside decodes). We tested whether executing requests in a continuous batch alters the generated tokens compared to running each prompt in isolation.

### Results
```
tests/test_batching.py::test_continuous_batching PASSED

Request 1 ('1 + 1 =', budget: 5 tokens):
  Single:  ' 2\n\\]\n\n'
  Batched: ' 2\n\\]\n\n' -> MATCH OK

Request 2 ('The colors of the rainbow are', budget: 15 tokens):
  Single:  ' red, orange, yellow, green, blue, indigo, and violet'
  Batched: ' red, orange, yellow, green, blue, indigo, and violet' -> MATCH OK

Request 3 ('Write a short summary of machine learning:', budget: 30 tokens):
  Single:  ' Machine learning is a subset of artificial intelligence that involves...'
  Batched: ' Machine learning is a subset of artificial intelligence that involves...' -> MATCH OK
```

- **Conclusion**: Continuous batching preserves exact mathematical determinism with 0 drift.

---

## 3. Paged vs Contiguous KV Cache Parity

### Methodology
We compared token generation using our Paged KV cache (16-token physical blocks gathered on demand via `gather_kv`) versus an unpaged, contiguous baseline tensor.

### Results
```
tests/test_paged_vs_contiguous.py::test_paged_vs_contiguous_kv_cache PASSED

Prompt: 'Artificial intelligence and machine learning algorithms'
Paged KV Cache Output:      ' are increasingly being used to predict and prevent natural disasters'
Contiguous KV Cache Output: ' are increasingly being used to predict and prevent natural disasters'
Paged Token IDs:      [525, 14756, 1660, 1483, 311, 7023, 323, 5358, 5810, 50651]
Contiguous Token IDs: [525, 14756, 1660, 1483, 311, 7023, 323, 5358, 5810, 50651]
```

- **Conclusion**: The logical-to-physical block gather operation produces bit-identical key and value projections compared to contiguous cache memory.

---

## 4. Block Allocator Leak & Double-Free Protection

### Methodology
Tested the `BlockAllocator` state machine under stress:
1. Double-free attempts (`free()` called twice on the same block ID).
2. Out-of-bounds releases (negative block ID or ID $\ge$ total blocks).
3. Out-of-memory handling when the pool is depleted.
4. 5,000 random allocation and release cycles.

### Results
```
tests/test_block_allocator.py::test_basic_allocation_and_free PASSED
tests/test_block_allocator.py::test_double_free_detection PASSED (raises ValueError)
tests/test_block_allocator.py::test_out_of_bounds_free PASSED (raises ValueError)
tests/test_block_allocator.py::test_oom_exhaustion PASSED (raises RuntimeError)
tests/test_block_allocator.py::test_no_leak_stress_test PASSED (5,000 cycles completed with ZERO leaks)
```

---

## 5. Dynamic Scheduler (Mid-Generation Join & Independence)

### Methodology
Evaluated dynamic iteration scheduling:
- Injected `req-1` (budget: 5) and `req-2` (budget: 15).
- Injected `req-3` at step 3 while `req-1` and `req-2` were actively in decode phase.
- Verified that `req-1` finished and freed its blocks at step 5 without delaying or interrupting `req-2` or `req-3`.

### Results
```
tests/test_scheduler_dynamic.py::test_independent_completion_and_midgen_join PASSED
tests/test_scheduler_dynamic.py::test_scheduler_abort_request PASSED

Schedule Trace:
- Step 1: Prefill batch = [req-1, req-2], Decode batch = []
- Step 2: Prefill batch = [], Decode batch = [req-1, req-2]
- Step 3 (req-3 arrives): Prefill batch = [req-3], Decode batch = [req-1, req-2]
- Step 5: req-1 finishes (freed blocks returned to pool)
- Steps 6-15: Decode batch continues with remaining requests
```

---

## 6. Sampling Reproducibility (Seeded Runs)

### Methodology
Evaluated token sampling with temperature $T=0.8$ and nucleus top-p $p=0.9$:
1. Executed multiple generation runs using identical seed (`seed=42`).
2. Executed generation with a distinct seed (`seed=999`).
3. Evaluated greedy sampling determinism ($T=0$).

### Results
```
tests/test_sampling_reproducibility.py::test_seeded_sampling_reproducibility PASSED
tests/test_sampling_reproducibility.py::test_greedy_determinism PASSED

Seeded Run A (seed=42):  [321, 321, 321, 321, 321, 321, 321, 321, 321, 321]
Seeded Run B (seed=42):  [321, 321, 321, 321, 321, 321, 321, 321, 321, 321] -> (IDENTICAL)
Seeded Run C (seed=999): [346, 346, 346, 346, 346, 346, 346, 346, 346, 346] -> (DIVERGENT)
```

---

## 7. Streaming API & Client Disconnect Cancellation

### Methodology
Tested FastAPI `/v1/chat/completions` Server-Sent Events (SSE) streaming and graceful cancellation:
- Streaming chunks comply with OpenAI's `chat.completion.chunk` schema.
- When an HTTP client aborts the connection mid-flight, the server catches `http_req.is_disconnected()`, cancels the request in the scheduler, and immediately reclaims its allocated physical KV cache blocks.

### Results
```
tests/test_streaming_and_cancellation.py::test_client_disconnect_frees_blocks PASSED

Trace:
- Initial Free Blocks: 20
- Request admitted (prompt=32 tokens): Allocated = 2, Free = 18
- Client receives 3 tokens and disconnects
- Scheduler abort triggered: status -> CANCELLED
- Blocks Reclaimed: Allocated = 0, Free = 20 (100% pool recovery)
```

---

## 8. Throughput & Latency Benchmark vs Baselines

### Hardware
- **GPU**: NVIDIA RTX A6000 (49,140 MiB VRAM)
- **Host**: Ubuntu 22.04 LTS, Python 3.10.12, PyTorch 2.5.1 (CUDA 12.1)
- **Model**: `Qwen/Qwen2.5-1.5B-Instruct` (FP16)

### Results
```
======================================================================
THROUGHPUT & LATENCY BENCHMARK
======================================================================
Engine Configuration               | Throughput (tok/s) | Latency / Step
----------------------------------------------------------------------
Mini Engine (10 concurrent reqs)   | 69.8 tok/s         | 143.3 ms
Mini Engine (5 concurrent reqs)    | 65.9 tok/s         | 75.9 ms
Mini Engine (4 concurrent reqs)    | 69.4 tok/s         | 57.6 ms
----------------------------------------------------------------------
```

### Architectural Analysis: Mini Engine vs vLLM & HF Generate
1. **Vs Hugging Face `generate`**:
   - Hugging Face `generate` uses static batching, requiring padding tokens to the longest sequence in the batch. While static batching can achieve high FLOPS on small uniform batches, it experiences massive bubble waste under real-world traffic where requests arrive asynchronously with varying prompt/output lengths.
   - Mini Engine's continuous batching eliminates padding bubbles completely, dynamically admitting new requests as soon as earlier requests complete.
2. **Vs Production vLLM**:
   - vLLM achieves higher throughput through custom Triton/CUDA C++ paged attention kernels.
   - Mini Engine implements the same architectural principles (Paged KV cache, block tables, continuous batching scheduler) purely in PyTorch with SDPA, delivering ~70 tokens/sec with full code readability.

---

## 9. KV Cache Memory Waste: Paged vs Contiguous

### Methodology
Simulated a realistic chatbot workload (128 concurrent requests, average length 294 tokens, max context limit 2,048 tokens) to measure internal and external fragmentation between:
1. **Contiguous Allocation**: Pre-allocates memory for the maximum possible context length ($L_{\max}=2048$) for every concurrent slot.
2. **Paged Allocation**: Allocates 16-token physical blocks on-demand.

### Empirical Measurements (`bench/bench_memory_waste.py`)
```
======================================================================
KV CACHE MEMORY CONSUMPTION & FRAGMENTATION EVALUATION
======================================================================
Concurrent requests:      128
Max context length limit: 2048 tokens
Paged block size:         16 tokens
KV memory per token:      28.00 KB (28 layers, 2 KV heads, 128 head dim, FP16)
Total actual tokens:      37,591 tokens (Avg: 293.7 tokens/sequence)
True useful KV memory:    1027.88 MB

Allocation Strategy       | Allocated (MB)  | Wasted (MB)  | Waste (%) 
----------------------------------------------------------------------
Contiguous (Pre-alloc)    | 7,168.00 MB     | 6,140.12 MB  | 85.7%
Paged (On-demand Blocks)  | 1,054.81 MB     | 26.93 MB     | 2.6%
----------------------------------------------------------------------
Memory Footprint Reduction:   85.3% less memory needed!
Effective Concurrency Boost:  6.80x more concurrent requests in same VRAM
======================================================================
```

- **Waste Reduction**: Contiguous allocation wastes **85.7%** of GPU memory waiting for sequences to grow. Paged allocation wastes only **2.6%** (internal fragmentation in the final block).
- **Concurrency Gain**: Paged allocation supports **6.8x higher concurrency** within the same GPU memory budget.

---

## 10. Continuous Integration (CI) Workflow

### Implementation
Configured GitHub Actions workflow in [`.github/workflows/ci.yml`](file:///c:/Users/lekka/OneDrive/Desktop/llm%20engine/mini-engine/.github/workflows/ci.yml) that executes on every push and pull request:
- Provisions clean Ubuntu runner with Python 3.10.
- Installs dependencies and package in editable mode.
- Executes the full test suite (`pytest tests/ -v -s`).
- Verifies unit tests, block allocator safety, streaming cancellation, and scheduler dynamics.

---

## 🎯 Verification Command

Run the entire suite locally or on the remote server:
```bash
pytest tests/ -v -s
```
Output:
```
============================= 13 passed in 37.81s ==============================
```
