"""Phase 2 (KV cache) vs Phase 1 baseline on the same prompts — warm-up/sync'd."""
from __future__ import annotations

import torch, time

from ..model.runner import ModelRunner
from ..cache.kv_cache import KVCache
from ..sampling.sampler import Sampler, SamplingParams


def generate_cached(runner: ModelRunner, prompt: str, params: SamplingParams) -> str:
    """Prefill the prompt, then loop decode_step, sampling each next token with the P1 Sampler."""
    # TODO (step 9a):
    #   9a.1 Tokenize the prompt; make a fresh KVCache sized to the max sequence length.
    #   9a.2 Prefill to fill the cache and get the first next-token logits.
    #   9a.3 Loop: sample a token (reuse Sampler); stop on the stop token or max_tokens; otherwise
    #        decode_step to produce the next logits.
    #   9a.4 Decode the generated token ids back to text.
    ...


def bench_kv(runner: ModelRunner, prompt: str, params: SamplingParams) -> dict:
    """{tokens_per_sec, per_step_latency_ms:[...], vram_mb_vs_tokens:[...]}.

    Warm up once; torch.cuda.synchronize() around every timed region; exclude prompt tokens.
    Record per-step latency AND torch.cuda.memory_allocated() after each decode step.
    """
    # TODO (step 9b): time prefill on its own (that is TTFT), then run a timed decode
    #                 loop recording per-step latency and VRAM after each step.
    ...
