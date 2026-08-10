"""Sampling — turn one logits row into a next-token id (greedy / temp / top-k / top-p)."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    temperature: float = 1.0     
    top_k: int | None = None
    top_p: float | None = None
    max_tokens: int = 128
    stop_token_id: int | None = None


class Sampler:
    """Stateless; one `sample()` call produces one token id."""

    def sample(self, logits: torch.Tensor, params: SamplingParams) -> int:
        """logits: [vocab] -> next token id. temp==0 is greedy."""
        if params.temperature == 0.0:
            return int(logits.argmax())
        logits = self._apply_temperature(logits, params.temperature)
        if params.top_k:
            logits = self._top_k_filter(logits, params.top_k)
        if params.top_p:
            logits = self._top_p_filter(logits, params.top_p)
        probs = torch.softmax(logits, -1)
        return int(torch.multinomial(probs, 1))


    def _apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Scale logits before softmax: <1 sharpens, >1 flattens."""
        return logits / temperature

    def _top_k_filter(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Keep the k highest logits; mask the rest to -inf."""
        threshold = torch.topk(logits, k).values[-1]
        logits = logits.clone()
        logits[logits < threshold] = float("-inf")
        return logits

    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Nucleus: keep the smallest set of tokens with cumprob >= p; mask the rest."""
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        remove_sorted = (cumprobs - probs) >= p        # exclusive prefix keeps the crossing token
        remove = torch.zeros_like(remove_sorted)
        remove.scatter_(-1, sorted_idx, remove_sorted)
        logits = logits.clone()
        logits[remove] = float("-inf")
        return logits
