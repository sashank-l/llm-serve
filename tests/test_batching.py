"""
tests/test_batching.py
Test continuous batching:
1. Batched outputs are identical to single-request outputs (greedy).
2. Requests of very different lengths finish independently without blocking each other.
"""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_continuous_batching():
    import os, sys, time
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from mini_engine.engine.engine import LLMEngine
    from mini_engine.engine.request import SamplingParams, RequestStatus

    model_path = "/home/surya/models/qwen2.5-1.5b"
    engine = LLMEngine(model_path, device="cuda", block_size=16, gpu_memory_fraction=0.3)

    prompts = [
        ("req-short", "1 + 1 =", 5),
        ("req-medium", "The colors of the rainbow are", 15),
        ("req-long", "Write a short summary of machine learning:", 30),
    ]

    # 1. Run each individually (single-request baseline)
    single_outputs = {}
    for req_id, p, max_tokens in prompts:
        params = SamplingParams(temperature=0.0, max_new_tokens=max_tokens)
        single_outputs[req_id] = engine.generate(p, params)
        print(f"Single [{req_id}]: {single_outputs[req_id]!r}")

    # 2. Add all to continuous batching scheduler concurrently
    req_objs = []
    for req_id, p, max_tokens in prompts:
        params = SamplingParams(temperature=0.0, max_new_tokens=max_tokens)
        req = engine.add_request(req_id, p, params)
        req_objs.append(req)

    # 3. Step engine until all finish
    step_count = 0
    finished_order = []
    while any(r.status != RequestStatus.FINISHED for r in req_objs):
        results = engine.step()
        step_count += 1
        for r_id, tok_id, is_fin in results:
            if is_fin and r_id not in finished_order:
                finished_order.append(r_id)
                print(f"Request {r_id} finished at step {step_count}!")

    print(f"\nAll requests completed in {step_count} engine steps.")
    print(f"Finish order: {finished_order}")

    # Verify short request finished before long request
    assert finished_order[0] == "req-short", "Short request should finish first in continuous batching!"

    # Verify outputs match single-request generation
    for req in req_objs:
        batched_text = engine.tokenizer.decode(req.generated_tokens, skip_special_tokens=True)
        print(f"\nChecking {req.request_id}:")
        print(f"  Single:  {single_outputs[req.request_id]!r}")
        print(f"  Batched: {batched_text!r}")
        assert batched_text == single_outputs[req.request_id], f"Batched output differs from single for {req.request_id}"
        print("  MATCH OK!")
