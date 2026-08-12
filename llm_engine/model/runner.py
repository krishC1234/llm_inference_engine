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
    def _forward_cached(self, input_ids: torch.Tensor, cache: KVCache,
                        positions: torch.Tensor, causal: bool) -> torch.Tensor:
        """Shared cached forward for any number of tokens -> logits: [seq_len, vocab].

        prefill and decode_step both delegate here; they differ only in positions/causal/length.
        """
        # TODO (step 7):
        #   7.1 Embed the input tokens.
        #   7.2 For each layer: norm, project Q/K/V, apply RoPE at `positions`, append K/V to the cache,
        #       and attend over the cached K/V with the given `causal` flag.
        #   7.3 Finish each layer as usual (output projection + residual, then MLP + residual).
        #   7.4 Final norm + lm_head; advance the cache by the number of new tokens; return all-position logits.
        ...

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, cache: KVCache) -> torch.Tensor:
        """Whole prompt in one pass -> last-position logits: [vocab]. Positions 0..prompt_len-1, causal."""
        # TODO (step 8a): give the prompt its absolute positions, run the shared forward causally,
        #                 and hand back only the next-token distribution.
        ...

    @torch.no_grad()
    def decode_step(self, token_id: int, cache: KVCache) -> torch.Tensor:
        """One new token, attend over the whole cache -> next-token logits: [vocab]. Position = cache.length."""
        # TODO (step 8b): place the single new token at its true position (continuing after the prompt),
        #                 run the shared forward over the whole cache, and hand back the next-token logits.
        ...
