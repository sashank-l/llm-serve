"""
model/loader.py
Load HuggingFace model weights into our LlamaModel.
"""
import glob
import json
import logging
import os
from typing import Dict

import torch
from .llama import LlamaConfig, LlamaModel

log = logging.getLogger(__name__)


def load_model(model_path: str, device: str = "cuda") -> LlamaModel:
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    config = LlamaConfig(
        vocab_size              = cfg.get("vocab_size", 151936),
        hidden_size             = cfg.get("hidden_size", 1536),
        num_hidden_layers       = cfg.get("num_hidden_layers", 28),
        num_attention_heads     = cfg.get("num_attention_heads", 12),
        num_key_value_heads     = cfg.get("num_key_value_heads", 2),
        intermediate_size       = cfg.get("intermediate_size", 8960),
        max_position_embeddings = cfg.get("max_position_embeddings", 32768),
        rms_norm_eps            = cfg.get("rms_norm_eps", 1e-6),
        rope_theta              = float(cfg.get("rope_theta", 1000000.0)),
        tie_word_embeddings     = cfg.get("tie_word_embeddings", True),
        attention_bias          = True,
    )

    model = LlamaModel(config)

    # Safetensors weight loading
    shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if shard_files:
        from safetensors.torch import load_file
        state_dict: Dict[str, torch.Tensor] = {}
        for shard in shard_files:
            sd = load_file(shard, device="cpu")
            for k, v in sd.items():
                our_k = k[6:] if k.startswith("model.") else k
                state_dict[our_k] = v.to(torch.float16)

        if config.tie_word_embeddings and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log.info("Loaded safetensors shards: missing=%s, unexpected=%s", len(missing), len(unexpected))
    else:
        log.warning("No safetensors found, initializing randomly.")

    model.to(device=device, dtype=torch.float16).eval()
    return model
