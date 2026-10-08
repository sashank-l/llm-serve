"""
tests/test_parity.py
Greedy parity test: verify our model produces the exact same token sequence as
HuggingFace `generate` (greedy, no sampling) on reference prompts.

Run with:
  pytest tests/test_parity.py -v -s
"""
import pytest
import torch


@pytest.fixture(scope="session")
def model_path():
    return "/home/surya/models/qwen2.5-1.5b"


PROMPTS = [
    "The capital of France is",
    "In Python, a list comprehension is",
    "The transformer architecture was introduced in the paper",
    "Once upon a time in a land far away,",
]

MAX_NEW_TOKENS = 15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_greedy_parity(model_path):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    from mini_engine.engine.engine import LLMEngine
    from mini_engine.engine.request import SamplingParams

    print(f"\nLoading HF model from {model_path}...")
    hf_tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cuda"
    ).eval()

    print(f"Loading custom mini_engine from {model_path}...")
    engine = LLMEngine(model_path, device="cuda", block_size=16, gpu_memory_fraction=0.3)
    greedy = SamplingParams(temperature=0.0, max_new_tokens=MAX_NEW_TOKENS)

    # Pure greedy configuration (clears any Qwen generation_config defaults)
    gen_config = GenerationConfig(do_sample=False)

    print("\nRunning parity checks...")
    for prompt in PROMPTS:
        enc = hf_tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            hf_out = hf_model.generate(
                **enc, generation_config=gen_config, max_new_tokens=MAX_NEW_TOKENS
            )
        hf_tokens = hf_out[0, enc["input_ids"].shape[1] :].tolist()
        hf_gen_text = hf_tok.decode(hf_tokens)

        # Our engine
        our_text = engine.generate(prompt, greedy)
        our_tokens = hf_tok.encode(our_text)

        print(f"\nPrompt: {prompt!r}")
        print(f"  HF output:   {hf_gen_text!r}")
        print(f"  Engine out:  {our_text!r}")
        print(f"  HF tokens:   {hf_tokens[:8]}")
        print(f"  Ours tokens: {our_tokens[:8]}")

        # Assert token-for-token equality
        assert hf_tokens == our_tokens, (
            f"Parity mismatch on {prompt!r}:\nHF: {hf_tokens}\nOurs: {our_tokens}"
        )
        print("  TOKEN-FOR-TOKEN PARITY: MATCH!")
