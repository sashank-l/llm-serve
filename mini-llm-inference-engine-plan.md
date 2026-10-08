# Mini LLM Inference Engine — Project Plan

A small LLM serving engine built from scratch in PyTorch: continuous batching, a paged KV cache, and an OpenAI-compatible streaming API.

**Goal:** show engineering depth in LLM inference (not just API calls), with every layer verified against a Hugging Face baseline.

**Timeline:** about 3–4 weeks for Phases 0–4. Phases 5–6 are optional and later.

---

## Principles

1. **Build in layers.** Each phase must work and be verified before the next starts.
2. **Parity first.** Greedy output must match Hugging Face `generate` token for token at every checkpoint. This is the safety net for every later bug.
3. **Correct before fast.** A slow, readable PyTorch implementation comes first. Kernels and optimizations come later.
4. **Log from day one.** Record per-request latency, tokens/sec and memory so benchmarks are easy to add later.

---

## Model choice

- A small Llama-style model: Qwen2.5-1.5B or Llama-3.2-1B (Qwen2.5-0.5B for fast iteration).
- Any GPU with 8 GB+ is enough.
- Components to implement: RMSNorm, RoPE, grouped-query attention (GQA), SwiGLU MLP, and a loader for the HF weights.

---

## Repo layout

```
mini-engine/
  model/          # llama-style layers, weight loader
  engine/         # scheduler, block allocator, kv cache, sampler
  server/         # FastAPI app, SSE streaming
  tests/          # parity tests vs HF (greedy match)
  bench/          # benchmarks (later)
  README.md
```

---

## Phase 0 — Setup and reading (day 1)

- [ ] Create the repo, environment and a basic project structure.
- [ ] Download the chosen model and tokenizer.
- [ ] Read the vLLM PagedAttention paper.
- [ ] Read the Orca paper (continuous / iteration-level batching).
- [ ] Skim the small open-source `nano-vllm` repo for structure (read for ideas, write your own code).

---

## Phase 1 — Single-request generation (days 2–4)

- [ ] Implement the model forward pass (RMSNorm, RoPE, GQA attention, SwiGLU).
- [ ] Write the HF weight loader (map parameter names to your modules).
- [ ] Add a simple contiguous KV cache and a prefill + decode loop.
- [ ] Implement sampling: greedy, temperature, top-k, top-p.
- [ ] Add stop conditions (EOS, max tokens, stop strings).

**Checkpoint:**
- Logits for a fixed prompt are close to HF logits.
- Greedy generation matches HF `generate` token for token on a set of test prompts.

---

## Phase 2 — Continuous batching (week 2)

- [ ] Build a request object (prompt tokens, sampling params, state, generated tokens).
- [ ] Build a scheduler with a waiting queue and a running set.
- [ ] Each engine step: prefill newly admitted requests, run one decode step for running ones.
- [ ] Remove finished requests immediately and admit new ones mid-generation.
- [ ] Handle padding and per-sequence positions within a batch.

**Checkpoint:**
- Batched outputs are identical to single-request outputs (greedy).
- Requests of very different lengths finish independently without blocking each other.

---

## Phase 3 — Paged KV cache (weeks 2–3)

- [ ] Pre-allocate the KV cache as fixed-size blocks (e.g. 16 tokens per block).
- [ ] Implement a block allocator (free list, allocate, free).
- [ ] Keep a block table per sequence mapping logical positions to physical blocks.
- [ ] Implement attention by gathering K/V through the block table (plain PyTorch gather first).
- [ ] Handle running out of blocks: queue new requests or preempt running ones.

**Checkpoint:**
- Outputs unchanged vs Phase 2.
- Much lower memory waste than per-request contiguous allocation (measure it).

---

## Phase 4 — Serving (week 3)

- [ ] FastAPI app with an OpenAI-compatible `/v1/chat/completions` endpoint.
- [ ] Token streaming over SSE (and a non-streaming mode).
- [ ] Engine loop running in a background thread, communicating with request handlers through queues.
- [ ] Basic request validation, cancellation on client disconnect, and error handling.
- [ ] A short README with run instructions and a curl example.

**Checkpoint:**
- A standard OpenAI client can talk to your server.
- Several concurrent streaming requests work correctly.

---

## Phase 5 — Optional extras

Pick based on time and interest.

- [ ] Prefix caching (reuse blocks for shared prompt prefixes).
- [ ] Chunked prefill (avoid long prompts stalling decode).
- [ ] Quantized weights (INT8 or AWQ).
- [ ] A Triton kernel for paged attention.

---

## Phase 6 — Benchmarks and evals (later)

Planned for after the build.

- [ ] Throughput (tokens/sec) and latency (p50 / p95 time-to-first-token and per-token) under concurrent load.
- [ ] GPU memory use and KV cache utilization.
- [ ] Comparison against plain HF `generate` and against vLLM on the same model and hardware.
- [ ] Correctness suite: greedy parity on a larger prompt set.
- [ ] Results tables and plots in the README.

---

## Definition of done

- Greedy parity with HF on the test prompt set.
- Continuous batching and paged KV cache both working.
- OpenAI-compatible streaming API running.
- README with architecture diagram, how it works, and how to run it.
- (Later) benchmark results vs HF and vLLM.

---

## Suggested weekly schedule

| Week | Focus |
|------|-------|
| 1 | Phase 0 and Phase 1 (model, weight loader, KV cache, sampling, parity) |
| 2 | Phase 2 and start of Phase 3 (scheduler, continuous batching, block allocator) |
| 3 | Finish Phase 3, then Phase 4 (serving, streaming, README) |
| 4+ | Phase 5 extras, then Phase 6 benchmarks |

---

## First step today

Create the repo and write `model/` with the weight loader. Run one prompt through your forward pass and compare the logits against Hugging Face. Everything else builds on that.
