"""Interactive entry point for the Phase 1 engine.

Run from the repo root:

    python main.py

Loads TinyLlama once, then reads prompts from you and generates a completion
for each using the engine you built (ModelRunner + Sampler + generate_naive).
Type 'quit' (or press Ctrl-C) to exit.
"""
from llm_engine.model.runner import ModelRunner
from llm_engine.sampling.sampler import SamplingParams
from llm_engine.benchmarks.bench_baseline import generate_naive

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
            max_tokens=128,
            stop_token_id=runner.tokenizer.eos_token_id,
        )
        # switch to sampling once _top_k/_top_p are done (steps 4-5):
        #   params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128, ...)

        output = generate_naive(runner, prompt, params)   # step 6
        print(f"\n{output}\n")


if __name__ == "__main__":
    main()
