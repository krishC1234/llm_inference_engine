"""Model runner — Phase 1 (baseline, no KV cache).

`ModelRunner` loads TinyLlama and runs a FULL forward pass every step. This is the
deliberately naive O(n^2) baseline; Phase 2 adds prefill/decode + the KV cache.

Lab: plans/phase-1-baseline.md  (Build order steps 2-3)
"""
from __future__ import annotations

import torch

from ..config import ModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        """Build a ready-to-run ModelRunner for a HuggingFace model.

        Reads the model's dimensions into a ModelConfig, downloads the fp16
        weights and tokenizer, moves the weights onto `device`, and puts the
        model in inference (eval) mode. The first call downloads the weights
        into the local HF cache (~2.2 GB for TinyLlama); later calls reuse it.

        Args:
            model_name: HuggingFace model id, e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0".
            device: where to place the weights ("cuda" on the T4; "cpu"/"mps" locally).

        Returns:
            A ModelRunner holding the loaded model, its ModelConfig, and tokenizer.
        """
        config = ModelConfig.from_hf(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=config.dtype).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(torch.cuda.memory_allocated() / 1e9) # Get the memory used by model ~ 2.2 GB
        return cls(model, config, tokenizer, device)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [seq_len] -> logits: [seq_len, vocab]. FULL recompute, no cache.

        TODO (step 3): embed -> n_layers of attention+MLP -> final RMSNorm -> lm_head.
        Simplest start: call the HF model with input_ids[None] (add batch dim) and
        return logits[0] (drop it). Assert output shape == (seq_len, vocab_size)
        BEFORE you write any generation loop.
        """
        ids = input_ids.to(self.device)
        out = self.model(ids[None])
        return out.logits[0]
