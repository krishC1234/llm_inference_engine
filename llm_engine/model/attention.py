"""Cached attention op: new-token query vs cached K/V, with GQA broadcast + causal mask."""
from __future__ import annotations

import torch, math


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """q:[n_heads,q_len,head_dim] k,v:[n_kv_heads,kv_len,head_dim] -> [n_heads,q_len,head_dim]. RoPE already applied to q,k."""
    # GQA: broadcast each KV head across its group of query heads.
    n_heads, seq_len, head_dim = q.shape
    kv_heads = k.shape[0]

    n_rep = n_heads // kv_heads
    k = k.repeat_interleave(n_rep, dim=0)
    v = v.repeat_interleave(n_rep, dim=0)

    # scaled dot-product scores
    scores = q @ k.transpose(-2, -1)
    scores /= math.sqrt(head_dim)

    # causal mask: a query never attends to a later key
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

    # softmax over keys, then weighted sum of values
    scores = torch.softmax(scores, dim=-1)
    weights = scores @ v
    return weights
