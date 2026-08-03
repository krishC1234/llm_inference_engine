"""Sampling — Phase 1.

Turn one row of logits ([vocab]) into the next token id. Greedy first, then the
temperature / top-k / top-p (nucleus) knobs a serving system exposes per request.

Lab: plans/phase-1-baseline.md  (Build order steps 4-5)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    temperature: float = 1.0     # 0.0 => greedy
    top_k: int | None = None
    top_p: float | None = None
    max_tokens: int = 128
    stop_token_id: int | None = None


class Sampler:
    """Stateless; one `sample()` call produces one token id."""

    def sample(self, logits: torch.Tensor, params: SamplingParams) -> int:
        """logits: [vocab] for ONE position -> next token id.

        TODO (step 4-5):
            if params.temperature == 0.0: return int(logits.argmax())      # greedy
            else: temp-scale -> top_k filter -> top_p filter -> softmax
                  -> torch.multinomial(probs, 1) -> int(token)
        """
        return int(logits.argmax()) # TODO: We need to change this away from greedy

    def _apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """logits: [vocab] -> [vocab], scaled before softmax (divide by temperature)."""
        raise NotImplementedError

    def _top_k_filter(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Keep k highest; mask the rest to -inf. logits: [vocab] -> [vocab]."""
        raise NotImplementedError

    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Nucleus: keep smallest set with cumprob >= p. logits: [vocab] -> [vocab]."""
        raise NotImplementedError
