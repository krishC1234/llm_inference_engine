"""Model configuration — Phase 1.

`ModelConfig` holds every model dimension, read from the HuggingFace config so
nothing downstream hardcodes 22 / 2048. Keeping this config-driven is what lets
the same code path later load Llama-2-7B (see plans/system-design.md).

Lab: plans/phase-1-baseline.md  (Build order step 1)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import pipeline, AutoConfig


@dataclass
class ModelConfig:
    n_layers: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int              
    head_dim: int
    vocab_size: int
    max_position: int
    dtype: torch.dtype = torch.float16

    @classmethod
    def from_hf(cls, model_name: str) -> "ModelConfig":
        """Read dims from the HF config — never hardcode.

        (Step 1): load `transformers.AutoConfig.from_pretrained(model_name)`
        and map its fields onto this dataclass. Watch the names:
            n_layers     <- num_hidden_layers
            hidden_size  <- hidden_size
            n_heads      <- num_attention_heads
            n_kv_heads   <- num_key_value_heads   (GQA; falls back to n_heads if absent)
            head_dim     <- hidden_size // num_attention_heads
            vocab_size   <- vocab_size
            max_position <- max_position_embeddings

        For TinyLlama assert n_kv_heads == 4 and n_layers == 22 — but read them from
        the config, not literals. Point this at a bigger Llama config to confirm
        nothing is hardcoded.
        Link: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
        """
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
