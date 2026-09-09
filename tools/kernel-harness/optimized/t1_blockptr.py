"""Trial 1: paged decode attention with block pointers for the KV stream.

Hypothesis. The baseline streams KV through ordinary strided loads. In a
[pages, page_size, kv_heads, head_dim] cache one head's rows are 256 contiguous bytes
separated by a 2048-byte stride, which DRAM serves at reduced efficiency. Intel's Triton
lowers `tl.make_block_ptr` to 2D block I/O, which issues the load as a shaped block rather
than a gather, so the same bytes should arrive faster.

Decode only: one query row per sequence, so the whole cost is the KV stream and the softmax
is online over tiles.
"""

import sys

import torch
import torch.nn as nn
import triton
import triton.language as tl

sys.path.insert(0, "tools/kernel-harness")
from vllm_unified_attention import HEAD_DIM, get_init_inputs, get_inputs  # noqa: E402,F401


@triton.jit
def _paged_decode(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    bt_ptr,
    used_ptr,
    q_s0,
    q_s1,
    k_s0,
    k_s1,
    k_s2,
    bt_s0,
    scale,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    PS: tl.constexpr,
    TILE: tl.constexpr,
    GROUP: tl.constexpr,
):
    seq = tl.program_id(0)
    kvh = tl.program_id(1)
    used = tl.load(used_ptr + seq)

    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, GROUP)
    # The GROUP query heads that share this KV head, loaded once and kept in registers.
    q = tl.load(q_ptr + seq * q_s0 + (kvh * GROUP + offs_g)[:, None] * q_s1 + offs_d[None, :])
    q = q.to(tl.float32) * scale

    m_i = tl.full((GROUP,), float("-inf"), tl.float32)
    l_i = tl.zeros((GROUP,), tl.float32)
    acc = tl.zeros((GROUP, D), tl.float32)

    for start in tl.range(0, used, TILE):
        offs_t = start + tl.arange(0, TILE)
        page = tl.load(bt_ptr + seq * bt_s0 + offs_t // PS, mask=offs_t < used, other=0)
        slot = offs_t % PS
        base = page[:, None] * k_s0 + slot[:, None] * k_s1 + kvh * k_s2 + offs_d[None, :]
        mask = (offs_t < used)[:, None]
        k = tl.load(k_ptr + base, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + base, mask=mask, other=0.0).to(tl.float32)

        s = tl.dot(q, tl.trans(k)) if GROUP >= 16 else tl.sum(q[:, None, :] * k[None, :, :], 2)
        s = tl.where(offs_t[None, :] < used, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], 1)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        out_ptr + seq * q_s0 + (kvh * GROUP + offs_g)[:, None] * q_s1 + offs_d[None, :],
        acc.to(out_ptr.dtype.element_ty),
    )


class Model(nn.Module):
    def __init__(self, head_dim: int = HEAD_DIM):
        super().__init__()
        self.scale = head_dim**-0.5

    def forward(self, q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table):
        B, HQ, D = q.shape
        HKV = k_cache.shape[2]
        PS = k_cache.shape[1]
        out = torch.empty_like(q)
        _paged_decode[(B, HKV)](
            q,
            k_cache,
            v_cache,
            out,
            block_table,
            seqused_k,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            block_table.stride(0),
            self.scale,
            HQ=HQ,
            HKV=HKV,
            D=D,
            PS=PS,
            TILE=32,
            GROUP=HQ // HKV,
            num_warps=4,
        )
        return out
