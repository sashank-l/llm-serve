"""
engine/sampler.py
Sampling utilities: greedy, temperature scaling, top-k, top-p (nucleus),
and seed-based reproducibility.
"""
from typing import Optional
import torch
from .request import SamplingParams


class Sampler:
    def sample(self, logits: torch.Tensor, params: SamplingParams) -> int:
        logits = logits.float()

        # 1. Greedy sampling
        if params.temperature == 0.0 or params.top_k == 1:
            return torch.argmax(logits, dim=-1).item()

        # 2. Temperature scaling
        logits = logits / params.temperature

        # 3. Top-K filtering
        if params.top_k > 0:
            top_k = min(params.top_k, logits.size(-1))
            k_val = torch.topk(logits, top_k, dim=-1)[0][..., -1]
            if logits.dim() == 1:
                logits[logits < k_val] = float("-inf")
            else:
                logits[logits < k_val.unsqueeze(-1)] = float("-inf")

        # 4. Top-P (nucleus) filtering
        if params.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > params.top_p
            # Shift right by 1 so the first token above threshold is kept
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            if logits.dim() == 1:
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float("-inf")
            else:
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

        probs = torch.softmax(logits, dim=-1)

        # 5. Multinomial sampling with optional seed / generator
        generator: Optional[torch.Generator] = None
        if params.seed is not None:
            dev = logits.device if logits.is_cuda else "cpu"
            generator = torch.Generator(device=dev)
            generator.manual_seed(params.seed)

        if generator is not None:
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        return next_token.item() if next_token.dim() == 1 else next_token[0].item()
