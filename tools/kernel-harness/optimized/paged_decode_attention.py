"""Trial 5: split the KV sequence across programs (flash-decoding), then combine.

t1-t4 established that tile shape and warp count are exhausted: 32/4 warps is best and
every variation loses. That is the signature of a kernel limited by how much memory traffic
it can keep in flight, not by how it computes -- one program streams a whole 1024-token
sequence for one KV head, so there are only B*HKV = 512 programs to hide latency behind.

Splitting the sequence into SPLITS chunks multiplies parallelism by SPLITS. Each chunk runs
an independent online softmax and writes its partial accumulator with the running max and
sum; a second pass rescales them onto a common maximum and reduces. The arithmetic is the
same, there is just more of it in flight.
"""

import sys

import torch
import torch.nn as nn
import triton
import triton.language as tl

sys.path.insert(0, "tools/kernel-harness")
from vllm_unified_attention import HEAD_DIM, get_init_inputs, get_inputs  # noqa: E402,F401

SPLITS = 4


@triton.jit
def _partial(
    q_ptr,
    k_ptr,
    v_ptr,
    bt_ptr,
    used_ptr,
    acc_ptr,
    m_ptr,
    l_ptr,
    q_s0,
    q_s1,
    k_s0,
    k_s1,
    k_s2,
    bt_s0,
    a_s0,
    a_s1,
    a_s2,
    a_s3,
    s_s0,
    s_s1,
    s_s2,
    scale,
    D: tl.constexpr,
    PS: tl.constexpr,
    TILE: tl.constexpr,
    GROUP: tl.constexpr,
    SPLITS: tl.constexpr,
):
    seq, kvh, sp = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    used = tl.load(used_ptr + seq)
    chunk = tl.cdiv(used, SPLITS)
    lo = sp * chunk
    hi = tl.minimum(lo + chunk, used)

    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, GROUP)
    q = tl.load(q_ptr + seq * q_s0 + (kvh * GROUP + offs_g)[:, None] * q_s1 + offs_d[None, :])
    q = q.to(tl.float32) * scale

    m_i = tl.full((GROUP,), float("-inf"), tl.float32)
    l_i = tl.zeros((GROUP,), tl.float32)
    acc = tl.zeros((GROUP, D), tl.float32)

    for start in tl.range(lo, hi, TILE):
        offs_t = start + tl.arange(0, TILE)
        live = offs_t < hi
        page = tl.load(bt_ptr + seq * bt_s0 + offs_t // PS, mask=live, other=0)
        base = page[:, None] * k_s0 + (offs_t % PS)[:, None] * k_s1 + kvh * k_s2 + offs_d[None, :]
        k = tl.load(k_ptr + base, mask=live[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptr + base, mask=live[:, None], other=0.0).to(tl.float32)

        s = tl.sum(q[:, None, :] * k[None, :, :], 2)
        s = tl.where(live[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], 1)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    tl.store(
        acc_ptr + seq * a_s0 + kvh * a_s1 + sp * a_s2 + offs_g[:, None] * a_s3 + offs_d[None, :],
        acc,
    )
    tl.store(m_ptr + seq * s_s0 + kvh * s_s1 + sp * s_s2 + offs_g, m_i)
    tl.store(l_ptr + seq * s_s0 + kvh * s_s1 + sp * s_s2 + offs_g, l_i)


@triton.jit
def _combine(
    acc_ptr,
    m_ptr,
    l_ptr,
    out_ptr,
    a_s0,
    a_s1,
    a_s2,
    a_s3,
    s_s0,
    s_s1,
    s_s2,
    o_s0,
    o_s1,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    SPLITS: tl.constexpr,
):
    seq, kvh = tl.program_id(0), tl.program_id(1)
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, GROUP)
    offs_s = tl.arange(0, SPLITS)

    m = tl.load(m_ptr + seq * s_s0 + kvh * s_s1 + offs_s[None, :] * s_s2 + offs_g[:, None])
    l = tl.load(l_ptr + seq * s_s0 + kvh * s_s1 + offs_s[None, :] * s_s2 + offs_g[:, None])
    m_star = tl.max(m, 1)
    w = tl.exp(m - m_star[:, None])  # rescale every chunk onto one maximum
    denom = tl.sum(w * l, 1)

    out = tl.zeros((GROUP, D), tl.float32)
    for sp in tl.range(0, SPLITS):
        a = tl.load(
            acc_ptr + seq * a_s0 + kvh * a_s1 + sp * a_s2 + offs_g[:, None] * a_s3 + offs_d[None, :]
        )
        out += (
            a * tl.load(m_ptr + seq * s_s0 + kvh * s_s1 + sp * s_s2 + offs_g)[:, None] * 0
            + a
            * tl.exp(tl.load(m_ptr + seq * s_s0 + kvh * s_s1 + sp * s_s2 + offs_g) - m_star)[
                :, None
            ]
        )
    out = out / denom[:, None]
    tl.store(
        out_ptr + seq * o_s0 + (kvh * GROUP + offs_g)[:, None] * o_s1 + offs_d[None, :],
        out.to(out_ptr.dtype.element_ty),
    )


class Model(nn.Module):
    def __init__(self, head_dim: int = HEAD_DIM):
        super().__init__()
        self.scale = head_dim**-0.5

    def forward(self, q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table):
        B, HQ, D = q.shape
        HKV, PS = k_cache.shape[2], k_cache.shape[1]
        G = HQ // HKV
        dev = q.device
        acc = torch.empty(B, HKV, SPLITS, G, D, dtype=torch.float32, device=dev)
        m = torch.empty(B, HKV, SPLITS, G, dtype=torch.float32, device=dev)
        l = torch.empty(B, HKV, SPLITS, G, dtype=torch.float32, device=dev)
        out = torch.empty_like(q)

        _partial[(B, HKV, SPLITS)](
            q,
            k_cache,
            v_cache,
            block_table,
            seqused_k,
            acc,
            m,
            l,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            block_table.stride(0),
            acc.stride(0),
            acc.stride(1),
            acc.stride(2),
            acc.stride(3),
            m.stride(0),
            m.stride(1),
            m.stride(2),
            self.scale,
            D=D,
            PS=PS,
            TILE=32,
            GROUP=G,
            SPLITS=SPLITS,
            num_warps=4,
        )
        _combine[(B, HKV)](
            acc,
            m,
            l,
            out,
            acc.stride(0),
            acc.stride(1),
            acc.stride(2),
            acc.stride(3),
            m.stride(0),
            m.stride(1),
            m.stride(2),
            out.stride(0),
            out.stride(1),
            D=D,
            GROUP=G,
            SPLITS=SPLITS,
            num_warps=4,
        )
        return out
