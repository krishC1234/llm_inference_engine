"""Baseline generation + benchmark — Phase 1.

`generate_naive` is the O(n^2) decode loop (re-runs the whole sequence each step).
`bench_baseline` times it with proper GPU warm-up + synchronization.

Run (from repo root):  python -m llm_engine.benchmarks.bench_baseline
Lab: plans/phase-1-baseline.md  (Build order steps 6-8)
"""
from __future__ import annotations

import torch

from ..model.runner import ModelRunner
from ..sampling.sampler import Sampler, SamplingParams


def generate_naive(runner: ModelRunner, prompt: str, params: SamplingParams) -> str:
    """Re-run the ENTIRE sequence every step (O(n^2)); append sampled token, repeat.

    TODO (step 6):
        ids = tensor(runner.tokenizer.encode(prompt), device=runner.device)   # [T0]
        sampler = Sampler()
        for _ in range(params.max_tokens):
            logits = runner.forward(ids)              # [seq_len, vocab] <- FULL recompute
            tok = sampler.sample(logits[-1], params)  # LAST row only
            if tok == params.stop_token_id: break
            ids = torch.cat([ids, tensor([tok])])
        return runner.tokenizer.decode(ids[T0:])      # decode only the new tokens
    """
    ids = torch.tensor(runner.tokenizer.encode(prompt), device=runner.device)
    t0 = len(ids)
    sampler = Sampler() 
    for _ in range(params.max_tokens):
        logits = runner.forward(ids)
        token = sampler.sample(logits[-1], params)
        if token == params.stop_token_id:
            break
        ids = torch.cat([ids, torch.tensor([token], device=runner.device)])
    return runner.tokenizer.decode(ids[t0:])


def bench_baseline(runner: ModelRunner, prompt: str, params: SamplingParams) -> dict:
    """{tokens_per_sec, per_step_latency_ms:[...]}. Warm up + torch.cuda.synchronize().

    TODO (step 7):
        - run one discarded warm-up generate_naive() before timing (first CUDA launch
          pays a one-time cost)
        - torch.cuda.synchronize() before every timer stop (GPU work is async)
        - record per-step latency alongside the current sequence length
        - report tokens_per_sec over the DECODE phase only
    """
    raise NotImplementedError("Phase 1, step 7: implement bench_baseline()")


if __name__ == "__main__":
    # Smoke entry point. Fill in the TODOs above, then run:
    #   python -m llm_engine.benchmarks.bench_baseline
    runner = ModelRunner.load("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    params = SamplingParams(temperature=0.0, max_tokens=128)
    print(generate_naive(runner, "The key idea behind PagedAttention is", params))
    print(bench_baseline(runner, "Once upon a time,", params))
