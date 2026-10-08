"""
bench/bench_throughput.py
Benchmark inference engine throughput (tokens/sec) and latency.
"""
import argparse
import time
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/surya/models/qwen2.5-1.5b")
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--tokens-per-request", type=int, default=32)
    args = parser.parse_args()

    from mini_engine.engine.engine import LLMEngine
    from mini_engine.engine.request import SamplingParams, RequestStatus

    print(f"Initializing engine for model: {args.model}...")
    engine = LLMEngine(args.model, device="cuda", block_size=16, gpu_memory_fraction=0.3)

    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming industries such as",
        "In modern computer science, algorithms are evaluated by",
        "The history of quantum mechanics begins with Max Planck and",
        "Deep learning models require significant compute resources because",
    ]

    reqs = []
    for i in range(args.num_requests):
        p = prompts[i % len(prompts)]
        params = SamplingParams(temperature=0.7, top_p=0.9, max_new_tokens=args.tokens_per_request)
        r = engine.add_request(f"bench-{i}", p, params)
        reqs.append(r)

    print(f"\nRunning benchmark: {args.num_requests} concurrent requests, {args.tokens_per_request} tokens each...")
    t0 = time.time()
    steps = 0
    total_tokens_generated = 0

    while any(r.status != RequestStatus.FINISHED for r in reqs):
        results = engine.step()
        steps += 1
        total_tokens_generated += len(results)

    elapsed = time.time() - t0
    throughput = total_tokens_generated / elapsed
    ms_per_step = (elapsed / steps) * 1000

    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total requests:          {args.num_requests}")
    print(f"Total tokens generated:  {total_tokens_generated}")
    print(f"Total engine steps:      {steps}")
    print(f"Total elapsed time:      {elapsed:.2f} s")
    print(f"Throughput:              {throughput:.1f} tokens/second")
    print(f"Latency per step:        {ms_per_step:.2f} ms")
    print(f"KV blocks allocated:     {len(engine.block_allocator.free_blocks)} free / {engine.block_allocator.num_blocks} total")
    print("=" * 60)


if __name__ == "__main__":
    main()
