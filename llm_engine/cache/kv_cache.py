"""Contiguous per-sequence KV cache — pre-allocated to max_seq_len, no torch.cat growth."""
from __future__ import annotations

import torch

from ..config import ModelConfig


class KVCache:
    """Per-sequence K/V store: [n_layers, 2, n_kv_heads, max_seq_len, head_dim]."""

    def __init__(self, config: ModelConfig, max_seq_len: int, device: str = "cuda") -> None:
        # Full store allocated once and never grown; _length tracks filled positions.
        self.kv = torch.zeros(config.n_layers, 2, config.n_kv_heads, max_seq_len, config.head_dim, dtype=config.dtype, device=device)
        self._length = 0

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write this layer's k,v for the new token at slot [length]. k,v: [n_kv_heads, 1, head_dim]."""
        # TODO (step 5.5): generalize for prefill — accept a chunk k,v: [n_kv_heads, seq, head_dim]
        #                  and write into [length : length + seq] so one call covers the whole prompt.
        self.kv[layer, 0, :, self._length, :] = k.squeeze(1)
        self.kv[layer, 1, :, self._length, :] = v.squeeze(1)

    def advance(self, n: int = 1) -> None:
        """Commit tokens: advance the fill pointer by n (1 per decode step, prompt_len in prefill)."""
        self._length += n

    def get(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """K,V up to current length: each [n_kv_heads, length, head_dim]."""
        return self.kv[layer, 0, :, :self.length, :], self.kv[layer, 1, :, :self.length, :]

    @property
    def length(self) -> int:
        """Filled positions == next write slot == RoPE index for the next token."""
        return self._length
