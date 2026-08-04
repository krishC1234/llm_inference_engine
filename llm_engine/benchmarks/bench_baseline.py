"""Baseline generation + benchmark — Phase 1.

`generate_naive` is the O(n^2) decode loop (re-runs the whole sequence each step).
`bench_baseline` times it with proper GPU warm-up + synchronization.

Run (from repo root):  python -m llm_engine.benchmarks.bench_baseline
Lab: plans/phase-1-baseline.md  (Build order steps 6-8)
"""
from __future__ import annotations

import torch, time, matplotlib
matplotlib.use("Agg")            # no display over SSH — render straight to a file
import matplotlib.pyplot as plt

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
    """Measure the performance of the naive (no-cache) generation baseline.

    Purpose: run generate_naive and quantify two things — overall throughput
    (tokens/sec) and how per-token latency grows as the sequence gets longer.
    Together these make the O(n^2) cost of recomputing the whole sequence every
    step *visible* — establishing the concrete Phase 1 numbers that Phase 2's KV
    cache (and every later phase) will be measured against. Without a baseline to
    beat, "faster" is meaningless; this is that baseline.

    Uses a warm-up run and torch.cuda.synchronize() so the GPU timings are
    accurate (the first CUDA launch is slow, and GPU work is asynchronous).

    Returns:
        {"tokens_per_sec": float, "per_step_latency_ms": [float, ...]}
    """
    ids = torch.tensor(runner.tokenizer.encode(prompt), device=runner.device)
    sampler = Sampler() 

    # Warmup so that the first latency run doesn't eat up extra timing
    runner.forward(ids)
    torch.cuda.synchronize()

    per_step_latency_ms = []
    for _ in range(params.max_tokens):
        start_time = time.perf_counter()
        logits = runner.forward(ids)
        torch.cuda.synchronize() 
        time_taken = (time.perf_counter() - start_time) * 1000 
        per_step_latency_ms.append(time_taken)
        token = sampler.sample(logits[-1], params)
        if token == params.stop_token_id:
            break
        ids = torch.cat([ids, torch.tensor([token], device=runner.device)])
    total_sec = sum(per_step_latency_ms) / 1000
    return {
        "tokens_per_sec": len(per_step_latency_ms) / total_sec,
        "per_step_latency_ms": per_step_latency_ms
    }


def plot_latency(result: dict, out: str = "baseline_latency.png", start_len: int = 0) -> str:
    """Plot per-step latency vs sequence length and save to `out` (a PNG).

    Headless-safe: forces the non-interactive 'Agg' backend and writes a file,
    so it works over SSH on the GPU box (no display, no `plt.show()`).

    Args:
        result: the dict returned by bench_baseline (needs "per_step_latency_ms").
        out: path to write the PNG to.
        start_len: offsets the x-axis to the TRUE sequence length. Pass the prompt's
            token count to plot vs length; default 0 plots vs decode-step index.

    Returns:
        The path written (download it with scp to view).
    """

    latencies = result["per_step_latency_ms"]
    x = [start_len + i for i in range(len(latencies))]

    plt.figure(figsize=(8, 5))
    plt.plot(x, latencies, marker=".", linewidth=1)
    plt.xlabel("sequence length (tokens)" if start_len else "decode step")
    plt.ylabel("per-step latency (ms)")
    plt.title("Naive baseline: per-step latency rises with length (O(n²) total)")
    plt.grid(True, alpha=0.3)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved plot -> {out}")
    return out
