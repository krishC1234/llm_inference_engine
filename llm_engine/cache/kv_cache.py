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
        self.kv[layer, 0, :, self._length, :] = k.squeeze(1)
        self.kv[layer, 1, :, self._length, :] = v.squeeze(1)

    def advance(self) -> None:
        """Commit one token: advance the fill pointer (call once per token, after all layers)."""
        self._length += 1

    def get(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """K,V up to current length: each [n_kv_heads, length, head_dim]."""
        return self.kv[layer, 0, :, :self.length, :], self.kv[layer, 1, :, :self.length, :]

    @property
    def length(self) -> int:
        """Filled positions == next write slot == RoPE index for the next token."""
        return self._length
