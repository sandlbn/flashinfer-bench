"""A decode-shaped GEMM with the consuming elementwise op folded into its epilogue.

At decode the GEMM is a skinny product -- a handful of rows against a weight that streams
from device memory once -- and the op that consumes it (a residual-add norm, a gated
activation) is a separate launch that costs more to issue than to run. This kernel does the
product as a GEMV over the weight's rows and applies the consumer in the epilogue, so the
pair becomes one launch and the intermediate never round-trips through memory.

One sub-group owns one output column (one row of the ``[N, K]`` weight, read contiguously,
each lane a vector of it per step); the activation rows are re-read from cache per column,
which is cheap next to the weight stream. The epilogue is selected at build time:

    gemm      D = A B^T                                  (control: the product alone)
    addnorm   R += D; D' = rmsnorm(R) * gamma            (fused_add_rms_norm semantics)
    swiglu    D' = silu(D[:, :d]) * D[:, d:]             (silu_and_mul semantics)

``addnorm`` needs the whole row's sum of squares, which no single work-group holds, so each
work-group writes its partial and the last one to finish -- found with a device-scope
counter -- applies the scale. That final pass reads back what the others wrote; the fences
around the counter order it.

Rounding follows the two kernels it replaces: the product is rounded to the activation
dtype where oneDNN would have stored it, and the consumer's arithmetic is repeated on that
rounded value, so the fused result agrees with the pair to the same bits their own rounding
allows.

Configuration (compile-time; swept by the trial loop, never fixed here):

    sg      sub-group width
    vec     elements each lane loads per step
    rows    output columns per work-group (one sub-group each)
    unroll  weight loads issued before their FMAs

The row count of the activation is specialised at build time from the tensor handed in.
"""

from __future__ import annotations

import importlib.util
import math
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

BASE_ENV = "FIB_HARNESS_BASE"
POOL_ENV = "FIB_WEIGHT_POOL_MB"

MODES = {"gemm": 0, "addnorm": 1, "swiglu": 2}

SOURCE = r"""
#include <sycl/sycl.hpp>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cstdint>

namespace fib_trials {

using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int SG = @SG@;
constexpr int VEC = @VEC@;
constexpr int ROWS = @ROWS@;
constexpr int UNROLL = @UNROLL@;
constexpr int MODE = @MODE@;   // 0 gemm, 1 addnorm, 2 swiglu
constexpr int M = @M@;
constexpr int FENCE = @FENCE@;   // 1: explicit device-scope fences around the counter
constexpr int SLMRED = @SLMRED@; // 1: sum the work-group's rows in SLM; 0: one partial per sub-group
constexpr int TWOROW = @TWOROW@; // swiglu: 1 walks gate and up rows in one loop
constexpr int SWZ = @SWZ@;       // swiglu: 1 pairs a gate sub-group with an up sub-group via SLM
constexpr int KS = @KS@;         // sub-groups per output column, each over K/KS (met in SLM)
constexpr int RB = @RB@;         // weight rows per sub-group sharing each activation load
constexpr int COOP = @COOP@;     // epilogue v3: 1 sums the work-group's slots across sub-group 0's lanes; 0 one lane walks them
constexpr int EPI = @EPI@;       // addnorm epilogue: 1 lane-0 serial chain; 2 lane-parallel, one barrier
constexpr int STEP = SG * VEC;  // elements of K one sub-group covers per step

using u16v = sycl::vec<uint16_t, VEC>;

inline float bf(uint16_t bits) { return sycl::bit_cast<float>(static_cast<uint32_t>(bits) << 16); }
inline uint16_t tobf(float f) { return sycl::bit_cast<uint16_t>(bf16(f)); }
inline float round_bf(float f) { return static_cast<float>(bf16(f)); }

struct Args {
  const uint16_t* x;      // [M, K]
  const uint16_t* w;      // [N, K]
  uint16_t* aux0;         // addnorm: residual [M, N] (updated in place)
  const uint16_t* aux1;   // addnorm: gamma [N]
  float* partials;        // addnorm: [num_wg, M]
  int* counter;           // addnorm: [1]
  uint16_t* out;          // gemm/addnorm: [M, N]; swiglu: [M, N/2]
  int K, N;
  int64_t ldw;            // elements between consecutive weight rows (a padded pitch is allowed)
  float eps;
};

// One sub-group: the dot products of `klen` elements of one weight row (from `wrow`) against
// the matching slice of every activation row (from `xrow0`, rows `K` apart).
inline void gemv_row(const Args& a, int lane, const uint16_t* wrow, const uint16_t* xrow0,
                     int klen, float (&acc)[M]) {
  const int steps = klen / STEP;
  const u16v* wv = reinterpret_cast<const u16v*>(wrow) + lane;
  const u16v* xv = reinterpret_cast<const u16v*>(xrow0) + lane;
  const int xstride = a.K / VEC;
  for (int s = 0; s < steps; s += UNROLL) {
    u16v wq[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) wq[u] = wv[(s + u) * SG];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const u16v xq = xv[m * xstride + (s + u) * SG];
        float part = 0.0f;
#pragma unroll
        for (int j = 0; j < VEC; ++j) part = sycl::fma(bf(wq[u][j]), bf(xq[j]), part);
        acc[m] += part;
      }
    }
  }
}

// RB weight rows per step, all dotted against the same activation vectors: the activation
// re-read from cache is divided by RB and RB weight vectors are in flight per lane.
inline void gemv_rows(const Args& a, int lane, const uint16_t* wrow0, int64_t ldw,
                      const uint16_t* xrow0, int klen, float (&acc)[RB][M]) {
  const int steps = klen / STEP;
  const u16v* wv[RB];
#pragma unroll
  for (int r = 0; r < RB; ++r) wv[r] = reinterpret_cast<const u16v*>(wrow0 + r * ldw) + lane;
  const u16v* xv = reinterpret_cast<const u16v*>(xrow0) + lane;
  const int xstride = a.K / VEC;
  for (int s = 0; s < steps; s += UNROLL) {
    u16v wq[UNROLL][RB];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
      for (int r = 0; r < RB; ++r) wq[u][r] = wv[r][(s + u) * SG];
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const u16v xq = xv[m * xstride + (s + u) * SG];
        float xf[VEC];
#pragma unroll
        for (int j = 0; j < VEC; ++j) xf[j] = bf(xq[j]);
#pragma unroll
        for (int r = 0; r < RB; ++r) {
          float part = 0.0f;
#pragma unroll
          for (int j = 0; j < VEC; ++j) part = sycl::fma(bf(wq[u][r][j]), xf[j], part);
          acc[r][m] += part;
        }
      }
    }
  }
}

// Two weight rows per step, both dotted against the same activation vectors: twice the
// weight bytes in flight per lane and half the activation re-reads.
inline void gemv_row2(const Args& a, int lane, const uint16_t* rowA, const uint16_t* rowB,
                      float (&accA)[M], float (&accB)[M]) {
  const int steps = a.K / STEP;
  const u16v* wa = reinterpret_cast<const u16v*>(rowA) + lane;
  const u16v* wb = reinterpret_cast<const u16v*>(rowB) + lane;
  const u16v* xv = reinterpret_cast<const u16v*>(a.x) + lane;
  const int xstride = a.K / VEC;
  for (int s = 0; s < steps; s += UNROLL) {
    u16v qa[UNROLL], qb[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) { qa[u] = wa[(s + u) * SG]; qb[u] = wb[(s + u) * SG]; }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const u16v xq = xv[m * xstride + (s + u) * SG];
        float pa = 0.0f, pb = 0.0f;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          const float xf = bf(xq[j]);
          pa = sycl::fma(bf(qa[u][j]), xf, pa);
          pb = sycl::fma(bf(qb[u][j]), xf, pb);
        }
        accA[m] += pa;
        accB[m] += pb;
      }
    }
  }
}

void Kernel(sycl::nd_item<1> it, const Args a, float* s_scale, int* s_flag) {
  auto sg = it.get_sub_group();
  const int lane = static_cast<int>(sg.get_local_id()[0]);
  const int sgid = static_cast<int>(sg.get_group_id()[0]);
  const int wg = static_cast<int>(it.get_group(0));
  const int num_wg = static_cast<int>(it.get_group_range(0));
  // Sub-groups [0, ROWS) own columns; with KS > 1 the next ROWS sub-groups take the second
  // K slice of the same columns, and so on. Only the slice-0 sub-group runs the epilogue.
  // Folded to constants when there is no K split, so the common path carries no extra
  // index arithmetic or conditionally-initialised state.
  const int col_local = (KS == 1) ? sgid : sgid % ROWS;
  const int split = (KS == 1) ? 0 : sgid / ROWS;
  const int n = (wg * ROWS + col_local) * RB;  // first of this sub-group's RB output columns
  const int klen = (KS == 1) ? a.K : a.K / KS;
  const int kbeg = (KS == 1) ? 0 : split * klen;

  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.0f;

  if constexpr (MODE == 2 && SWZ == 1) {
    // One weight row per sub-group per RB, as in the plain GEMV; the first half of the
    // work-group holds gate columns, the second half the matching up columns, and they meet
    // in SLM.
    const int d = a.N / 2;
    constexpr int HALF = ROWS / 2;
    const bool is_up = sgid >= HALF;
    const int j0 = (wg * HALF + (is_up ? sgid - HALF : sgid)) * RB;
    const int row0 = is_up ? d + j0 : j0;
    float accb[RB][M];
#pragma unroll
    for (int r = 0; r < RB; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) accb[r][m] = 0.0f;
    if constexpr (RB == 1) {
      gemv_row(a, lane, a.w + static_cast<int64_t>(row0) * a.ldw, a.x, a.K, accb[0]);
    } else {
      gemv_rows(a, lane, a.w + static_cast<int64_t>(row0) * a.ldw, a.ldw, a.x, a.K, accb);
    }
#pragma unroll
    for (int r = 0; r < RB; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) accb[r][m] = sycl::reduce_over_group(sg, accb[r][m], sycl::plus<float>());
    float* s_up = s_scale + M;  // [HALF][RB][M]
    if (is_up && lane == 0) {
#pragma unroll
      for (int r = 0; r < RB; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) s_up[((sgid - HALF) * RB + r) * M + m] = accb[r][m];
    }
    sycl::group_barrier(it.get_group());
    if (!is_up && lane == 0) {
#pragma unroll
      for (int r = 0; r < RB; ++r) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const float g = round_bf(accb[r][m]);
          const float u = round_bf(s_up[(sgid * RB + r) * M + m]);
          const float sv = round_bf(g / (1.0f + sycl::exp(-g)));
          a.out[static_cast<int64_t>(m) * d + j0 + r] = tobf(sv * u);
        }
      }
    }
    return;
  }

  if constexpr (MODE == 2) {
    const int d = a.N / 2;
    float accu[M];
#pragma unroll
    for (int m = 0; m < M; ++m) accu[m] = 0.0f;
    if constexpr (TWOROW == 1) {
      gemv_row2(a, lane, a.w + static_cast<int64_t>(n) * a.ldw,
                a.w + static_cast<int64_t>(n + d) * a.ldw, acc, accu);
    } else {
      gemv_row(a, lane, a.w + static_cast<int64_t>(n) * a.ldw, a.x, a.K, acc);
      gemv_row(a, lane, a.w + static_cast<int64_t>(n + d) * a.ldw, a.x, a.K, accu);
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      acc[m] = sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
      accu[m] = sycl::reduce_over_group(sg, accu[m], sycl::plus<float>());
    }
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float g = round_bf(acc[m]);
        const float u = round_bf(accu[m]);
        const float s = round_bf(g / (1.0f + sycl::exp(-g)));
        a.out[static_cast<int64_t>(m) * d + n] = tobf(s * u);
      }
    }
    return;
  }

  float res_pref = 0.0f;
  float res_all[RB][M];
  if constexpr (MODE == 1 && EPI == 2) {
    if (lane < M) res_pref = bf(a.aux0[static_cast<int64_t>(lane) * a.N + n]);
  }
  if constexpr (MODE == 1 && (EPI == 3 || EPI == 5)) {
    // Issued before the GEMV so their latency hides under it; all loads are independent.
    if (KS == 1 || split == 0) {
#pragma unroll
      for (int r = 0; r < RB; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) res_all[r][m] = bf(a.aux0[static_cast<int64_t>(m) * a.N + n + r]);
    }
  }
  float accb[RB][M];
#pragma unroll
  for (int r = 0; r < RB; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) accb[r][m] = 0.0f;
  if constexpr (RB == 1 && KS == 1) {
    gemv_row(a, lane, a.w + static_cast<int64_t>(n) * a.ldw, a.x, a.K, accb[0]);
  } else if constexpr (RB == 1) {
    gemv_row(a, lane, a.w + static_cast<int64_t>(n) * a.ldw + kbeg, a.x + kbeg, klen, accb[0]);
  } else {
    gemv_rows(a, lane, a.w + static_cast<int64_t>(n) * a.ldw + kbeg, a.ldw, a.x + kbeg, klen, accb);
  }
#pragma unroll
  for (int r = 0; r < RB; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) accb[r][m] = sycl::reduce_over_group(sg, accb[r][m], sycl::plus<float>());
  if constexpr (KS > 1) {
    float* s_acc = s_scale + M + ROWS * RB * M;  // [ROWS][KS][RB][M]
    if (split > 0 && lane == 0) {
#pragma unroll
      for (int r = 0; r < RB; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) s_acc[((col_local * KS + split) * RB + r) * M + m] = accb[r][m];
    }
    sycl::group_barrier(it.get_group());
    if (split == 0) {
#pragma unroll
      for (int r = 0; r < RB; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) {
#pragma unroll
          for (int q = 1; q < KS; ++q) accb[r][m] += s_acc[((col_local * KS + q) * RB + r) * M + m];
        }
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = accb[0][m];  // EPI 1/2 paths handle a single column

  if constexpr (MODE == 0) {
    if (split == 0 && lane == 0) {
#pragma unroll
      for (int r = 0; r < RB; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) a.out[static_cast<int64_t>(m) * a.N + n + r] = tobf(accb[r][m]);
    }
    return;
  }

  if constexpr (EPI == 2) {
    // Lane m owns row m of this column: its residual was prefetched before the GEMV, so the
    // add, store and square cost no dependent load here; the work-group's sums meet in SLM
    // once, and sub-group 0 alone carries the counter and, if last, the scaled write.
    float sq_lane = 0.0f;
#pragma unroll
    for (int m = 0; m < M; ++m) {
      if (lane == m) {
        const float v = round_bf(round_bf(acc[m]) + res_pref);
        a.aux0[static_cast<int64_t>(m) * a.N + n] = tobf(v);
        sq_lane = v * v;
      }
    }
    float* s_part = s_scale + M;  // [ROWS][M]
    if (lane < M) s_part[sgid * M + lane] = sq_lane;
    sycl::group_barrier(it.get_group());
    if (sgid != 0) return;
    float tot = 0.0f;
    if (lane < M) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r) tot += s_part[r * M + lane];
      a.partials[static_cast<int64_t>(wg) * M + lane] = tot;
    }
    int seen = 0;
    if (lane == 0) {
      if constexpr (FENCE == 1) sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
      sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          cnt(*a.counter);
      seen = cnt.fetch_add(1);
    }
    seen = sycl::group_broadcast(sg, seen, 0);
    if (seen != num_wg - 1) return;
    if (lane == 0) {
      if constexpr (FENCE == 1) sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
      *a.counter = 0;
    }
    float rstd_lane = 0.0f;
    if (lane < M) {
      float t = 0.0f;
      for (int g = 0; g < num_wg; ++g) t += a.partials[static_cast<int64_t>(g) * M + lane];
      rstd_lane = sycl::rsqrt(t / static_cast<float>(a.N) + a.eps);
    }
    float rstd[M];
#pragma unroll
    for (int m = 0; m < M; ++m) rstd[m] = sycl::group_broadcast(sg, rstd_lane, m);
    const int nvec = a.N / VEC;
    const u16v* rv = reinterpret_cast<const u16v*>(a.aux0);
    const u16v* gv = reinterpret_cast<const u16v*>(a.aux1);
    u16v* ov = reinterpret_cast<u16v*>(a.out);
    for (int i = lane; i < M * nvec; i += SG) {
      const int m = i / nvec;
      const int c = i - m * nvec;
      float scale = rstd[0];
#pragma unroll
      for (int mm = 1; mm < M; ++mm) scale = (m == mm) ? rstd[mm] : scale;
      const u16v r = rv[i];
      const u16v g = gv[c];
      u16v o;
#pragma unroll
      for (int j = 0; j < VEC; ++j) o[j] = tobf(bf(r[j]) * scale * bf(g[j]));
      ov[i] = o;
    }
    return;
  }

  if constexpr (EPI == 3) {
    float* s_part = s_scale + M;  // [ROWS][M]
    if (split == 0 && lane == 0) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float sq = 0.0f;
#pragma unroll
        for (int r = 0; r < RB; ++r) {
          const float v = round_bf(round_bf(accb[r][m]) + res_all[r][m]);
          a.aux0[static_cast<int64_t>(m) * a.N + n + r] = tobf(v);
          sq += v * v;
        }
        s_part[col_local * M + m] = sq;
      }
    }
    sycl::group_barrier(it.get_group());
    if constexpr (COOP == 1) {
      if (sgid == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
          float t = 0.0f;
          for (int r = lane; r < ROWS; r += SG) t += s_part[r * M + m];
          t = sycl::reduce_over_group(sg, t, sycl::plus<float>());
          if (lane == 0) a.partials[static_cast<int64_t>(wg) * M + m] = t;
        }
      }
    } else if (it.get_local_id(0) == 0) {
      // One lane walks the slots: SLM loads are cheap enough that this measured faster
      // than the sub-group-wide reduction on the shapes tried.
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float t = 0.0f;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) t += s_part[r * M + m];
        a.partials[static_cast<int64_t>(wg) * M + m] = t;
      }
    }
    if (it.get_local_id(0) == 0) {
      if constexpr (FENCE == 1) sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
      sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          cnt(*a.counter);
      const int seen = cnt.fetch_add(1);
      *s_flag = (seen == num_wg - 1) ? 1 : 0;
      if (seen == num_wg - 1) {
        if constexpr (FENCE == 1) sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
        *a.counter = 0;
      }
    }
    sycl::group_barrier(it.get_group());
    if (*s_flag == 0) return;
    const int lid = static_cast<int>(it.get_local_id(0));
    const int lsize = static_cast<int>(it.get_local_range(0));
    if (sgid == 0) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float t = 0.0f;
        for (int g = lane; g < num_wg; g += SG) t += a.partials[static_cast<int64_t>(g) * M + m];
        t = sycl::reduce_over_group(sg, t, sycl::plus<float>());
        if (lane == 0) s_scale[m] = sycl::rsqrt(t / static_cast<float>(a.N) + a.eps);
      }
    }
    sycl::group_barrier(it.get_group());
    const int nvec = a.N / VEC;
    const u16v* rv = reinterpret_cast<const u16v*>(a.aux0);
    const u16v* gv = reinterpret_cast<const u16v*>(a.aux1);
    u16v* ov = reinterpret_cast<u16v*>(a.out);
    for (int i = lid; i < M * nvec; i += lsize) {
      const int m = i / nvec;
      const int c = i - m * nvec;
      const u16v r = rv[i];
      const u16v g = gv[c];
      const float scale = s_scale[m];
      u16v o;
#pragma unroll
      for (int j = 0; j < VEC; ++j) o[j] = tobf(bf(r[j]) * scale * bf(g[j]));
      ov[i] = o;
    }
    return;
  }

  if constexpr (EPI == 5) {
    // First of two kernels: residual add in place and one partial per work-group. The scale
    // pass is a second kernel on the same in-order queue, so no counter, flag or tail here.
    float* s_part = s_scale + M;  // [ROWS][M]
    if (split == 0 && lane == 0) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float sq = 0.0f;
#pragma unroll
        for (int r = 0; r < RB; ++r) {
          const float v = round_bf(round_bf(accb[r][m]) + res_all[r][m]);
          a.aux0[static_cast<int64_t>(m) * a.N + n + r] = tobf(v);
          sq += v * v;
        }
        s_part[col_local * M + m] = sq;
      }
    }
    sycl::group_barrier(it.get_group());
    if (sgid == 0) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float t = 0.0f;
        for (int r = lane; r < ROWS; r += SG) t += s_part[r * M + m];
        t = sycl::reduce_over_group(sg, t, sycl::plus<float>());
        if (lane == 0) a.partials[static_cast<int64_t>(wg) * M + m] = t;
      }
    }
    return;
  }

  // MODE 1: residual add in place, partial sum of squares per row, last block scales.
  float sq[M];
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const int64_t idx = static_cast<int64_t>(m) * a.N + n;
      const float v = round_bf(round_bf(acc[m]) + bf(a.aux0[idx]));
      a.aux0[idx] = tobf(v);
      sq[m] = v * v;
    }
  } else {
#pragma unroll
    for (int m = 0; m < M; ++m) sq[m] = 0.0f;
  }
  // Sum the work-group's ROWS columns, then publish one partial per work-group (SLMRED) or
  // one per sub-group; the last work-group sums whatever was published.
#pragma unroll
  for (int m = 0; m < M; ++m) sq[m] = sycl::reduce_over_group(sg, sq[m], sycl::plus<float>());
  if constexpr (SLMRED == 1) {
    if (sgid == 0 && lane == 0) {
#pragma unroll
      for (int m = 0; m < M; ++m) s_scale[m] = 0.0f;
    }
    sycl::group_barrier(it.get_group());
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space>
            slot(s_scale[m]);
        slot.fetch_add(sq[m]);
      }
    }
    sycl::group_barrier(it.get_group());
    if (it.get_local_id(0) == 0) {
#pragma unroll
      for (int m = 0; m < M; ++m) a.partials[static_cast<int64_t>(wg) * M + m] = s_scale[m];
    }
  } else {
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < M; ++m) a.partials[static_cast<int64_t>(n) * M + m] = sq[m];
    }
    sycl::group_barrier(it.get_group());
  }
  if (it.get_local_id(0) == 0) {
    if constexpr (FENCE == 1) sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        cnt(*a.counter);
    const int seen = cnt.fetch_add(1);
    *s_flag = (seen == num_wg - 1) ? 1 : 0;
    if (seen == num_wg - 1) {
      if constexpr (FENCE == 1) sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
      *a.counter = 0;  // self-cleaning: every other work-group is past its increment
    }
  }
  sycl::group_barrier(it.get_group());
  if (*s_flag == 0) return;

  // Last work-group: total sum of squares per row, then the scaled write of every element.
  const int lid = static_cast<int>(it.get_local_id(0));
  const int lsize = static_cast<int>(it.get_local_range(0));
  if (sgid == 0) {
    const int nparts = (SLMRED == 1) ? num_wg : num_wg * ROWS;
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float t = 0.0f;
      for (int g = lane; g < nparts; g += SG) t += a.partials[static_cast<int64_t>(g) * M + m];
      t = sycl::reduce_over_group(sg, t, sycl::plus<float>());
      if (lane == 0) s_scale[m] = sycl::rsqrt(t / static_cast<float>(a.N) + a.eps);
    }
  }
  sycl::group_barrier(it.get_group());
  const int nvec = a.N / VEC;
  const u16v* rv = reinterpret_cast<const u16v*>(a.aux0);
  const u16v* gv = reinterpret_cast<const u16v*>(a.aux1);
  u16v* ov = reinterpret_cast<u16v*>(a.out);
  for (int i = lid; i < M * nvec; i += lsize) {
    const int m = i / nvec;
    const int c = i - m * nvec;
    const u16v r = rv[i];
    const u16v g = gv[c];
    const float scale = s_scale[m];
    u16v o;
#pragma unroll
    for (int j = 0; j < VEC; ++j) o[j] = tobf(bf(r[j]) * scale * bf(g[j]));
    ov[i] = o;
  }
}

void FusedGemv(tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView aux0,
               tvm::ffi::TensorView aux1, tvm::ffi::TensorView partials,
               tvm::ffi::TensorView counter, double eps, tvm::ffi::TensorView out) {
  TVM_FFI_ICHECK_EQ(x.ndim(), 2) << "x must be [M, K]";
  TVM_FFI_ICHECK_EQ(w.ndim(), 2) << "w must be [N, K]";
  TVM_FFI_ICHECK_EQ(x.size(0), M) << "kernel specialised for a different row count";
  TVM_FFI_ICHECK_EQ(x.size(1), w.size(1)) << "K mismatch";
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w.size(0));
  TVM_FFI_ICHECK_EQ(K % (STEP * UNROLL), 0) << "K must be a multiple of sg*vec*unroll";
  TVM_FFI_ICHECK_EQ(N % ROWS, 0) << "N must be a multiple of rows per work-group";
  TVM_FFI_ICHECK_EQ(N % VEC, 0) << "N must be a multiple of vec";
  if constexpr (MODE == 2 && SWZ == 0) TVM_FFI_ICHECK_EQ(N % (2 * ROWS), 0) << "N/2 must be a multiple of rows";
  if constexpr (MODE == 2 && SWZ == 1) TVM_FFI_ICHECK_EQ((N / 2) % (ROWS / 2), 0) << "N/2 must be a multiple of rows/2";
  if constexpr (MODE == 1 && EPI == 2) TVM_FFI_ICHECK_LE(M, SG) << "lane-parallel epilogue needs M <= sub-group";
  if constexpr (MODE == 1 && EPI != 3 && EPI != 5) TVM_FFI_ICHECK_EQ(RB, 1) << "only epilogues v3/v5 handle several columns per sub-group";

  DLDevice dev = x.device();
  sycl::queue* q = static_cast<sycl::queue*>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";

  Args a;
  a.x = static_cast<const uint16_t*>(x.data_ptr());
  a.w = static_cast<const uint16_t*>(w.data_ptr());
  a.aux0 = static_cast<uint16_t*>(aux0.data_ptr());
  a.aux1 = static_cast<const uint16_t*>(aux1.data_ptr());
  a.partials = static_cast<float*>(partials.data_ptr());
  a.counter = static_cast<int*>(counter.data_ptr());
  a.out = static_cast<uint16_t*>(out.data_ptr());
  a.K = K;
  a.N = N;
  {
    auto st = w.strides();
    a.ldw = (st.size() == 2 && st[0] > 0) ? static_cast<int64_t>(st[0]) : static_cast<int64_t>(K);
  }
  TVM_FFI_ICHECK_EQ(a.ldw % VEC, 0) << "weight row pitch must keep rows vector-aligned";
  a.eps = static_cast<float>(eps);

  const int columns = ((MODE == 2 && SWZ == 0) ? N / 2 : N) / RB;  // sub-groups per K slice
  TVM_FFI_ICHECK_EQ(N % (RB * ROWS), 0) << "N must be a multiple of rb*rows";
  if constexpr (MODE == 2 && SWZ == 1) TVM_FFI_ICHECK_EQ((N / 2) % (RB * ROWS / 2), 0) << "N/2 must be a multiple of rb*rows/2";
  const size_t local = static_cast<size_t>(ROWS) * KS * SG;
  const size_t global = static_cast<size_t>(columns) * KS * SG;
  TVM_FFI_ICHECK_EQ(K % (KS * STEP * UNROLL), 0) << "K/KS must be a multiple of sg*vec*unroll";
  if constexpr (MODE == 2) TVM_FFI_ICHECK_EQ(KS, 1) << "swiglu takes no K split";
  q->submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> s_scale(sycl::range<1>(M + ROWS * RB * M + ROWS * KS * RB * M), cgh);
    sycl::local_accessor<int, 1> s_flag(sycl::range<1>(1), cgh);
    cgh.parallel_for(sycl::nd_range<1>(global, local),
                     [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                       Kernel(it, a,
                              s_scale.get_multi_ptr<sycl::access::decorated::no>().get(),
                              s_flag.get_multi_ptr<sycl::access::decorated::no>().get());
                     });
  });
  if constexpr (MODE == 1 && EPI == 5) {
    // Scale pass: every element of out from the summed residual, the row's partials and gamma.
    const int nvec = N / VEC;
    const int num_wg = columns / ROWS;
    const size_t total = static_cast<size_t>(M) * nvec;
    constexpr size_t kBlock = 256;
    const size_t rounded = (total + kBlock - 1) / kBlock * kBlock;
    q->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_rstd(sycl::range<1>(M), cgh);
      cgh.parallel_for(sycl::nd_range<1>(rounded, kBlock), [=](sycl::nd_item<1> it) {
        // Every work-group reduces the row partials once, cooperatively, then scales its slice.
        const int lid = static_cast<int>(it.get_local_id(0));
#pragma unroll
        for (int m = 0; m < M; ++m) {
          float t = 0.0f;
          for (int g = lid; g < num_wg; g += static_cast<int>(kBlock)) t += a.partials[static_cast<int64_t>(g) * M + m];
          t = sycl::reduce_over_group(it.get_group(), t, sycl::plus<float>());
          if (lid == 0) s_rstd[m] = sycl::rsqrt(t / static_cast<float>(a.N) + a.eps);
        }
        sycl::group_barrier(it.get_group());
        const size_t i = it.get_global_id(0);
        if (i >= total) return;
        const int m = static_cast<int>(i / nvec);
        const int c = static_cast<int>(i - static_cast<size_t>(m) * nvec);
        const float scale = s_rstd[m];
        const u16v r = reinterpret_cast<const u16v*>(a.aux0)[i];
        const u16v gm = reinterpret_cast<const u16v*>(a.aux1)[c];
        u16v o;
#pragma unroll
        for (int j = 0; j < VEC; ++j) o[j] = tobf(bf(r[j]) * scale * bf(gm[j]));
        reinterpret_cast<u16v*>(a.out)[i] = o;
      });
    });
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_gemv, FusedGemv);

}  // namespace fib_trials
"""


def _definition(mode: str):
    """An in-memory Definition with every axis variable: the builder wants a name and a
    signature, and a shape belongs to the tensors handed in, not to the kernel."""
    from flashinfer_bench.data import Definition

    outputs = {"out": {"shape": ["M", "D"], "dtype": "bfloat16"}}
    return Definition.model_validate(
        {
            "name": f"fused_gemv_{mode}",
            "op_type": "gemm",
            "description": f"decode GEMV with a fused {mode} epilogue (trial)",
            "axes": {k: {"type": "var"} for k in ("M", "K", "N", "D", "P", "S")},
            "inputs": {
                "x": {"shape": ["M", "K"], "dtype": "bfloat16"},
                "w": {"shape": ["N", "K"], "dtype": "bfloat16"},
                "aux0": {"shape": ["M", "N"], "dtype": "bfloat16"},
                "aux1": {"shape": ["N"], "dtype": "bfloat16"},
                "partials": {"shape": ["P"], "dtype": "float32"},
                "counter": {"shape": ["S"], "dtype": "int32"},
            },
            "outputs": outputs,
            "reference": "import torch\n\ndef run(x, w, aux0, aux1, partials, counter):\n    return x @ w.t()\n",
        }
    )


_BUILT: Dict[Tuple, Any] = {}


def build_kernel(mode: str, m: int, cfg: Dict[str, int]):
    key = (mode, m, tuple(sorted(cfg.items())))
    fn = _BUILT.get(key)
    if fn is None:
        import sys

        sys.path.insert(0, "tools/kernel-harness")
        from sycl_harness import build

        src = SOURCE
        cfg = {k: v for k, v in cfg.items() if k != "pad"}  # a host-side choice, not a kernel constant
        for name, value in dict({"fence": 1, "slmred": 1, "epi": 1, "tworow": 0, "swz": 0, "ks": 1, "rb": 1, "coop": 0, "pad": 0}, **cfg, MODE=MODES[mode], M=m).items():
            src = src.replace(f"@{name.upper()}@", str(int(value)))
        fn = _BUILT[key] = build(_definition(mode), src)
    return fn


def _base_module():
    path = os.environ.get(BASE_ENV)
    if not path:
        raise SystemExit(f"{BASE_ENV} must name the pair harness this candidate is measured against.")
    spec = importlib.util.spec_from_file_location("fused_pair_base", pathlib.Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pool_copies(w: torch.Tensor) -> int:
    pool_mb = int(os.environ.get(POOL_ENV, "0"))
    if pool_mb <= 0:
        return 1
    return max(1, math.ceil(pool_mb * 2**20 / (w.numel() * w.element_size())))


class _Fused(nn.Module):
    """Runs the fused kernel on the pair harness's arguments: producer inputs, then the
    consumer's remaining inputs. The weight is rotated over a pool exactly as the baseline
    rotates it, so both arms stream the same bytes."""

    def __init__(self, mode: str, cfg: Dict[str, int], control: bool = False):
        super().__init__()
        self.mode, self.cfg, self.control = mode, dict(cfg), control
        base = _base_module()
        self.scalars = list(getattr(base, "SCALARS", []))
        self.replaces = int(getattr(base, "REPLACES", 0))
        # The consumer op itself, for the control arm (our GEMM, their epilogue kernel).
        self.consumer = base._Pair().consumer if control else None
        self._ptr, self._pool, self._calls = None, [], 0
        self._scratch: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._dummy: Optional[torch.Tensor] = None

    def _prepare(self, w: torch.Tensor) -> torch.Tensor:
        if not self.cfg.get("pad"):
            return w
        from flashinfer_bench.integration import weight_layout as wl

        padded = wl.pad_rows_off_channel_period(w)
        return w if padded is None else padded

    def _weight(self, w: torch.Tensor) -> torch.Tensor:
        if w.data_ptr() != self._ptr:
            self._ptr = w.data_ptr()
            self._pool = [self._prepare(w)] + [
                self._prepare(w.clone()) for _ in range(_pool_copies(w) - 1)
            ]
            self._calls = 0
        if len(self._pool) == 1:
            return self._pool[0]
        w_i = self._pool[self._calls % len(self._pool)]
        self._calls += 1
        return w_i

    def _scratch_for(self, m: int, n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (m, n, str(device))
        s = self._scratch.get(key)
        if s is None:
            s = self._scratch[key] = (
                torch.zeros(n * m, dtype=torch.float32, device=device),
                torch.zeros(1, dtype=torch.int32, device=device),
            )
        return s

    def forward(self, x: torch.Tensor, w: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
        w = self._weight(w)
        m, n = x.shape[0], w.shape[0]
        eps = float(self.scalars[0]) if self.scalars else 0.0
        if self._dummy is None:
            self._dummy = torch.zeros(1, dtype=torch.bfloat16, device=x.device)
        partials, counter = self._scratch_for(m, n, x.device)
        kernel_mode = "gemm" if self.control else self.mode
        fn = build_kernel(kernel_mode, m, self.cfg)
        if self.control:
            h = torch.empty(m, n, dtype=x.dtype, device=x.device)
            fn(x, w, self._dummy, self._dummy, partials, counter, eps, h)
            args = list(rest)
            args.insert(self.replaces, h)
            return self.consumer(*args)
        if self.mode == "addnorm":
            residual, gamma = rest[0], rest[1]
            out = torch.empty(m, n, dtype=x.dtype, device=x.device)
            fn(x, w, residual, gamma, partials, counter, eps, out)
            return out
        if self.mode == "swiglu":
            out = rest[0]
            fn(x, w, self._dummy, self._dummy, partials, counter, eps, out)
            return out
        out = torch.empty(m, n, dtype=x.dtype, device=x.device)
        fn(x, w, self._dummy, self._dummy, partials, counter, eps, out)
        return out


def make(mode: str, cfg: Dict[str, int], control: bool = False):
    base = _base_module()

    class Model(_Fused):
        def __init__(self):
            super().__init__(mode, cfg, control)

    return Model, base.get_inputs, base.get_init_inputs
