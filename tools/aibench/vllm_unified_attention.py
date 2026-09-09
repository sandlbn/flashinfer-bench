"""ai-bench Model that calls vLLM's unified attention *in place* -- nothing is extracted.

Lifting a kernel out of a serving stack is the step that silently goes wrong: the signature
is long, several arguments derive from engine objects, and a reconstruction that is subtly
off still runs and still benchmarks. None of that is necessary to evaluate the kernel. The
ai-bench contract is a `Model` with `forward`, `get_inputs` and `get_init_inputs`, and
`forward` is free to import the production kernel and call it.

So the baseline here *is* the kernel vLLM runs, imported from vLLM, at shapes taken from a
real definition in the dataset. An optimizer working on this file replaces the body with its
own implementation and is measured against the real thing rather than against a copy of it.

Shapes follow `gqa_paged_decode_h32_kv8_d128_ps64`: 32 query heads, 8 KV heads, head_dim
128, page size 64.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 64
BATCH = 64
CONTEXT = 1024
DTYPE = torch.bfloat16


class Model(nn.Module):
    """Paged GQA decode attention, as vLLM executes it."""

    def __init__(self, head_dim: int = HEAD_DIM) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

    def forward(self, q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table):
        # Imported here rather than at module scope so the file can be read and analysed
        # without vLLM installed; the call itself is the production path, untouched.
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention

        out = torch.empty_like(q)
        unified_attention(
            q=q,
            k=k_cache,
            v=v_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seqused_k,
            max_seqlen_k=CONTEXT,
            softmax_scale=self.scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
        )
        return out


def get_inputs():
    """One decode step: a single query row per sequence against a paged KV cache."""
    device = "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    pages_per_seq = (CONTEXT + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = BATCH * pages_per_seq
    return [
        torch.randn(BATCH, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=device),
        torch.randn(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device),
        torch.randn(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device),
        torch.arange(BATCH + 1, dtype=torch.int32, device=device),
        torch.full((BATCH,), CONTEXT, dtype=torch.int32, device=device),
        torch.arange(num_pages, dtype=torch.int32, device=device).view(BATCH, pages_per_seq),
    ]


def get_init_inputs():
    return [HEAD_DIM]
