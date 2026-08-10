"""Model runner — baseline (no KV cache): full forward pass every step."""
from __future__ import annotations

import torch

from ..config import ModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelRunner:
    def __init__(self, model, config: ModelConfig, tokenizer=None, device: str = "cuda"):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def load(cls, model_name: str, device: str = "cuda") -> "ModelRunner":
        """Load fp16 weights + tokenizer onto `device` in eval mode."""
        config = ModelConfig.from_hf(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=config.dtype).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(torch.cuda.memory_allocated() / 1e9)  # weights VRAM (GB)
        return cls(model, config, tokenizer, device)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [seq_len] -> logits: [seq_len, vocab]. Full recompute, no cache."""
        ids = input_ids.to(self.device)
        out = self.model(ids[None])
        return out.logits[0]
