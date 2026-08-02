"""Model runner — Phase 1 (baseline, no KV cache).

`ModelRunner` loads TinyLlama and runs a FULL forward pass every step. This is the
deliberately naive O(n^2) baseline; Phase 2 adds prefill/decode + the KV cache.

Lab: plans/phase-1-baseline.md  (Build order steps 2-3)
"""
from __future__ import annotations

import torch

from ..config import ModelConfig


class ModelRunner:
    def __init__(self, model, config: ModelConfig, tokenizer=None, device: str = "cuda"):
        # store model (put in eval mode), config, tokenizer, device.
        # tokenizer lives here so generate_naive() can go str -> ids -> str.
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def load(cls, model_name: str, device: str = "cuda") -> "ModelRunner":
        """Load tokenizer + fp16 weights to GPU; build ModelConfig.from_hf().

        TODO (step 2):
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=fp16)
                        .to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            config = ModelConfig.from_hf(model_name)
            print(torch.cuda.memory_allocated()) -> expect ~2.2 GB for TinyLlama fp16
            return cls(model, config, tokenizer, device)
        """
        raise NotImplementedError("Phase 1, step 2: implement ModelRunner.load()")

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [seq_len] -> logits: [seq_len, vocab]. FULL recompute, no cache.

        TODO (step 3): embed -> n_layers of attention+MLP -> final RMSNorm -> lm_head.
        Simplest start: call the HF model with input_ids[None] (add batch dim) and
        return logits[0] (drop it). Assert output shape == (seq_len, vocab_size)
        BEFORE you write any generation loop.
        """
        raise NotImplementedError("Phase 1, step 3: implement ModelRunner.forward()")
