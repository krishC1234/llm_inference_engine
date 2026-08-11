"""Cached attention op: new-token query vs cached K/V, with GQA broadcast + causal mask."""
from __future__ import annotations

import torch


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """q:[n_heads,q_len,head_dim]; k,v:[n_kv_heads,kv_len,head_dim] -> [n_heads,q_len,head_dim].

    q,k already carry RoPE at their absolute positions (applied in the runner).
    """
    # TODO:
    #   6.1 GQA: make each query head see a KV head — there are fewer KV heads than
    #       query heads, so share/broadcast the KV heads across their query-head group.
    #   6.2 Score every query against every key and scale so the softmax is well-conditioned.
    #   6.3 Causal mask: stop a query position from attending to keys that come after it.
    #       (Prefill has many query positions at once; decode has a single query over the whole cache.)
    #   6.4 Turn scores into weights and take the weighted sum of the values.
    ...
