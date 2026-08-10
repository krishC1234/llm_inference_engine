"""Model dimensions, read from the HF config so nothing downstream is hardcoded."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoConfig


@dataclass
class ModelConfig:
    n_layers: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int              # GQA; falls back to n_heads if absent
    head_dim: int
    vocab_size: int
    max_position: int
    dtype: torch.dtype = torch.float16

    @classmethod
    def from_hf(cls, model_name: str) -> "ModelConfig":
        """Read dims from the HF config — never hardcode."""
        cfg = AutoConfig.from_pretrained(model_name)
        return cls(
            n_layers = cfg.num_hidden_layers,
            hidden_size = cfg.hidden_size, 
            n_heads = cfg.num_attention_heads,
            n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            head_dim = cfg.hidden_size // cfg.num_attention_heads,
            vocab_size = cfg.vocab_size,
            max_position = cfg.max_position_embeddings,
        )
