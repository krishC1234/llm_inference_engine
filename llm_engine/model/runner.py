"""Model runner — baseline forward (P1) + cached prefill/decode (P2)."""
from __future__ import annotations

import torch

from ..config import ModelConfig
from ..cache.kv_cache import KVCache
from .attention import attention
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

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, cache: KVCache) -> torch.Tensor:
        """Whole prompt in one pass, fill the cache at every layer -> last-position logits: [vocab]."""
        # TODO (step 7):
        #   7.1 Embed the prompt tokens; their positions are 0..prompt_len-1.
        #   7.2 For each layer: norm, project Q/K/V for ALL prompt tokens, apply RoPE at the true
        #       positions, write K/V into the cache, and attend causally (each token sees only earlier ones).
        #   7.3 Finish each layer as usual (output projection + residual, then MLP + residual).
        #   7.4 Final norm + lm_head; return ONLY the last position's logits — the prompt is already known.
        ...

    @torch.no_grad()
    def decode_step(self, token_id: int, cache: KVCache) -> torch.Tensor:
        """One new token, append its K/V, attend over the whole cache -> next-token logits: [vocab]."""
        # TODO (step 8):
        #   8.1 Embed the single token; its RoPE position is the current cache length (the true absolute index).
        #   8.2 For each layer: project Q/K/V for this one token, apply RoPE, append its K/V to the cache,
        #       and attend the new query over ALL cached K/V.
        #   8.3 Finish each layer (output projection + residual, MLP + residual), same as prefill.
        #   8.4 Commit the token in the cache; final norm + lm_head; return the logits.
        ...
