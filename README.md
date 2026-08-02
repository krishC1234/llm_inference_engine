# LLM Inference Engine

A from-scratch, vLLM-style LLM inference engine with continuous batching
and paged KV-cache management to maximize throughput on a single GPU.
The goal: serve more concurrent requests at lower latency and cost on
commodity hardware (NVIDIA T4), turning expensive model serving into an
efficient, production-ready system.
