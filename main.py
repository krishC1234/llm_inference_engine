"""Interactive entry point for the Phase 1 engine.

Run from the repo root:

    python main.py

Loads TinyLlama once, then reads prompts from you and generates a completion
for each using the engine you built (ModelRunner + Sampler + generate_naive).
Type 'quit' (or press Ctrl-C) to exit.
"""
from dataclasses import replace

from llm_engine.model.runner import ModelRunner
from llm_engine.sampling.sampler import SamplingParams
from llm_engine.benchmarks.bench_baseline import (
    generate_naive, bench_baseline, plot_latency, plot_cumulative_latency,
)

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def main():
    print(f"Loading {MODEL} ...")
    runner = ModelRunner.load(MODEL)              # step 2
    print("Ready. Enter a prompt ('quit' to exit).\n")

    while True:
        prompt = input("prompt> ").strip()
        if prompt.lower() in {"quit", "exit", ""}:
            break

        params = SamplingParams(
            temperature=0.0,                      # 0.0 = greedy (deterministic)
            max_tokens=2000,
            stop_token_id=runner.tokenizer.eos_token_id,
        )
        # switch to sampling once _top_k/_top_p are done (steps 4-5):
        #   params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128, ...)

        print(generate_naive(runner, prompt, params))

        # benchmark ignores EOS so it runs the full max_tokens (fixed-length curve)
        bench_params = replace(params, stop_token_id=None)
        result = bench_baseline(runner, prompt, bench_params)
        print(f"\nthroughput: {result['tokens_per_sec']:.1f} tok/s "
              f"over {len(result['per_step_latency_ms'])} steps")

        prompt_len = len(runner.tokenizer.encode(prompt))
        plot_latency(result, start_len=prompt_len)             # per-step: O(N) per step
        plot_cumulative_latency(result, start_len=prompt_len)  # total: O(N^2) parabola


if __name__ == "__main__":
    main()
