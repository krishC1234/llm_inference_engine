"""Phase 2 (KV cache) vs Phase 1 baseline on the same prompts — warm-up/sync'd."""
from __future__ import annotations

import torch, time

from ..model.runner import ModelRunner
from ..sampling.sampler import Sampler, SamplingParams


def bench_kv(runner: ModelRunner, prompt: str, params: SamplingParams) -> dict:
    """{tokens_per_sec, per_step_latency_ms:[...], vram_mb_vs_tokens:[...]}.

    Warm up once; torch.cuda.synchronize() around every timed region; exclude prompt tokens.
    Record per-step latency AND torch.cuda.memory_allocated() after each decode step.
    """
    # TODO (step 9b): time prefill on its own (that is TTFT), then run a timed decode
    #                 loop recording per-step latency and VRAM after each step.
    ...
