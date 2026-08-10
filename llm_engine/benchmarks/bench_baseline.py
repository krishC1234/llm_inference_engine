"""Baseline O(n^2) decode loop + a warm-up/sync'd throughput benchmark."""
from __future__ import annotations

import torch, time, matplotlib
matplotlib.use("Agg")            # no display over SSH — render straight to a file
import matplotlib.pyplot as plt

from ..model.runner import ModelRunner
from ..sampling.sampler import Sampler, SamplingParams


def generate_naive(runner: ModelRunner, prompt: str, params: SamplingParams) -> str:
    """Re-run the whole sequence each step, append the sampled token, repeat."""
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
    """Time the naive loop -> {tokens_per_sec, per_step_latency_ms}. Warm-up + sync'd."""
    ids = torch.tensor(runner.tokenizer.encode(prompt), device=runner.device)
    sampler = Sampler()

    runner.forward(ids)              # warm-up: first CUDA launch pays a one-time cost
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
    """Plot per-step latency vs sequence length -> PNG. start_len offsets x to true length."""
    latencies = result["per_step_latency_ms"]
    x = [start_len + i for i in range(len(latencies))]

    plt.figure(figsize=(8, 5))
    plt.plot(x, latencies, marker=".", linewidth=1)
    plt.xlabel("sequence length (tokens)" if start_len else "decode step")
    plt.ylabel("per-step latency (ms)")
    plt.title("Naive baseline: per-step latency (O(N) per step)")
    plt.grid(True, alpha=0.3)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved plot -> {out}")
    return out


def plot_cumulative_latency(result: dict, out: str = "baseline_cumulative.png",
                            start_len: int = 0) -> str:
    """Plot cumulative time (running total of per-step latency) -> the O(N^2) curve."""
    from itertools import accumulate
    import matplotlib
    matplotlib.use("Agg")            # no display over SSH — render straight to a file
    import matplotlib.pyplot as plt

    latencies = result["per_step_latency_ms"]
    cumulative_s = [ms / 1000 for ms in accumulate(latencies)]   # ms -> running total in seconds
    x = [start_len + i for i in range(len(latencies))]

    plt.figure(figsize=(8, 5))
    plt.plot(x, cumulative_s, linewidth=1.5)
    plt.xlabel("sequence length (tokens)" if start_len else "decode step")
    plt.ylabel("cumulative time (s)")
    plt.title("Naive baseline: cumulative generation time (O(N²))")
    plt.grid(True, alpha=0.3)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved plot -> {out}")
    return out
