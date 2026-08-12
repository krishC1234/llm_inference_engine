"""Phase 2 (KV cache) vs Phase 1 baseline on the same prompts — warm-up/sync'd."""
from __future__ import annotations

import torch, time

from ..model.runner import ModelRunner
from ..cache.kv_cache import KVCache
from ..sampling.sampler import Sampler, SamplingParams


def generate_cached(runner: ModelRunner, prompt: str, params: SamplingParams) -> str:
    """Prefill the prompt, then loop decode_step, sampling each next token with the P1 Sampler."""
    ids_out = []
    ids = torch.tensor(runner.tokenizer.encode(prompt), device=runner.device)
    kv_cache = KVCache(runner.config, runner.config.max_position, device=runner.device)
    sampler = Sampler()

    # prefill the prompt, then sample the first token
    logits = runner.prefill(ids, kv_cache)
    ids_out.append(sampler.sample(logits, params))

    # decode one token at a time until EOS or max_tokens
    for _ in range(params.max_tokens):
        logits = runner.decode_step(ids_out[-1], kv_cache)
        next_token = sampler.sample(logits, params)
        if next_token == params.stop_token_id:
            break
        ids_out.append(next_token)
    return runner.tokenizer.decode(ids_out)


def bench_kv(runner: ModelRunner, prompt: str, params: SamplingParams) -> dict:
    """{tokens_per_sec, per_step_latency_ms:[...], vram_mb_vs_tokens:[...]}.

    Warm up once; torch.cuda.synchronize() around every timed region; exclude prompt tokens.
    Record per-step latency AND torch.cuda.memory_allocated() after each decode step.
    """
    # TODO (step 9b): time prefill on its own (that is TTFT), then run a timed decode
    #                 loop recording per-step latency and VRAM after each step.
    ...
