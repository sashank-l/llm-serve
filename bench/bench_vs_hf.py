"""
bench/bench_vs_hf.py
Comparative throughput and latency benchmark: Mini Engine vs Hugging Face generate.
"""
import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from mini_engine.engine.engine import LLMEngine
from mini_engine.engine.request import SamplingParams, RequestStatus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/surya/models/qwen2.5-1.5b")
    parser.add_argument("--num-requests", type=int, default=6)
    parser.add_argument("--tokens-per-request", type=int, default=24)
    args = parser.parse_args()

    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence and deep learning are transforming industries by",
        "The fundamental theory of quantum computing relies on qubits and",
        "Explain how memory allocation works in modern operating systems:",
        "In Python, generator functions yield values one by one to save",
        "A distributed consensus algorithm like Raft ensures that all nodes",
    ][: args.num_requests]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print("=" * 65)
    print("COMPARATIVE BENCHMARK: Mini Engine vs Hugging Face generate")
    print("=" * 65)
    print(f"Model: {args.model}")
    print(f"Concurrent requests: {len(prompts)}, Tokens/request: {args.tokens_per_request}\n")

    # -----------------------------------------------------------------------
    # 1. Hugging Face generate benchmark
    # -----------------------------------------------------------------------
    print("Loading Hugging Face model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    gen_cfg = GenerationConfig(do_sample=False)

    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")

    # Warmup
    with torch.no_grad():
        hf_model.generate(**enc, generation_config=gen_cfg, max_new_tokens=4)
    torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        hf_out = hf_model.generate(
            **enc, generation_config=gen_cfg, max_new_tokens=args.tokens_per_request
        )
    torch.cuda.synchronize()
    hf_time = time.time() - t0
    hf_tokens = len(prompts) * args.tokens_per_request
    hf_throughput = hf_tokens / hf_time

    del hf_model
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 2. Mini Engine continuous batching benchmark
    # -----------------------------------------------------------------------
    print("\nLoading Mini Engine with Paged KV Cache...")
    engine = LLMEngine(args.model, device="cuda", block_size=16, gpu_memory_fraction=0.3)

    reqs = []
    for i, p in enumerate(prompts):
        params = SamplingParams(temperature=0.0, max_new_tokens=args.tokens_per_request)
        reqs.append(engine.add_request(f"bench-{i}", p, params))

    t0 = time.time()
    steps = 0
    total_tokens = 0
    while any(r.status != RequestStatus.FINISHED for r in reqs):
        results = engine.step()
        steps += 1
        total_tokens += len(results)
    torch.cuda.synchronize()
    engine_time = time.time() - t0
    engine_throughput = total_tokens / engine_time
    engine_step_lat = (engine_time / steps) * 1000

    # -----------------------------------------------------------------------
    # Results Summary Table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'Engine':<25} | {'Throughput (tok/s)':<20} | {'Total Time (s)':<15}")
    print("-" * 65)
    print(f"{'HuggingFace generate':<25} | {hf_throughput:<20.1f} | {hf_time:<15.2f}")
    print(f"{'Mini Engine (Continuous)':<25} | {engine_throughput:<20.1f} | {engine_time:<15.2f}")
    print("-" * 65)
    print(f"Mini Engine Step Latency: {engine_step_lat:.2f} ms / step ({steps} steps)")
    speedup = engine_throughput / hf_throughput if hf_throughput > 0 else 1.0
    print(f"Continuous Batching Speedup: {speedup:.2f}x vs static batching HF generate")
    print("=" * 65)


if __name__ == "__main__":
    main()
