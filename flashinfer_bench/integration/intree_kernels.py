"""In-tree kernels, generated as Solutions for every definition they implement.

This is the counterpart of :mod:`flashinfer_bench.integration.xpu_kernels`. That module
wraps kernels somebody else ships; this one carries kernels the project owns, and expands
one source template into a Solution for every definition whose signature it satisfies.

The reason to have both is structural, and the ROCm port of FlashInfer arrived at the same
split: an in-tree kernel for every operation, with the vendor library as an *overlay* on
the subset it covers. That way a definition is never unimplementable merely because
``vllm-xpu-kernels`` has no entry for it, and every upstream kernel has something to be
measured against that is not the PyTorch reference.

Templates are parameterized rather than duplicated. A kernel is written once against a
signature; the element type, vector width and epsilon are substituted per definition, so
fifteen RMSNorm definitions across nine hidden sizes come from one source. Substitution
uses :class:`string.Template` because the sources are C++ and full of braces that
``str.format`` would try to interpret.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from string import Template
from typing import Dict, List, Optional, Tuple

from flashinfer_bench.data import BuildSpec, Definition, Solution, SourceFile, SupportedLanguages

from .xpu_kernels import definition_eps

logger = logging.getLogger(__name__)

_SYCL_SCALARS: Dict[str, str] = {
    "bfloat16": "sycl::ext::oneapi::bfloat16",
    "float16": "sycl::half",
    "float32": "float",
}
"""Definition dtype to the SYCL type a kernel should use for it."""

_VEC_WIDTHS: Dict[str, int] = {"bfloat16": 8, "float16": 8, "float32": 4}
"""Elements per access, chosen so one load is 16 bytes.

Sixteen bytes is the widest single access the memory pipe takes. A memory-bound kernel
reading one element per work-item leaves most of that idle, which measured as a 1.35x
penalty at prefill batch sizes on Battlemage.
"""


def _scalar_type(definition: Definition) -> str:
    dtype = next(iter(definition.inputs.values())).dtype
    try:
        return _SYCL_SCALARS[dtype]
    except KeyError:
        raise ValueError(f"No SYCL scalar type for dtype '{dtype}'") from None


def _vec_size(definition: Definition) -> int:
    return _VEC_WIDTHS[next(iter(definition.inputs.values())).dtype]


def _sub_group(definition: Definition) -> str:
    """Sub-group width to size a Triton launch against.

    32 on every Intel AOT target today (Battlemage reports {16, 32}); the per-device value
    is ``Capabilities.preferred_sub_group_size`` for code running on a known device. A
    generated solution is shared, so it cannot bake in one machine's answer.
    """
    return "32"


def _num_stages(definition: Definition) -> str:
    """Pipeline depth. CUDA uses this for async-copy pipelining, which Intel lacks; a
    measured sweep on Battlemage found 2 at or near the optimum across shapes and never
    worse than 1, while 3 regressed badly at small batch (0.0168 ms vs 0.0082 ms).
    """
    return "2"


_SUBSTITUTIONS = {
    "eps": lambda d: repr(definition_eps(d)),
    "scalar_t": _scalar_type,
    "vec_size": lambda d: str(_vec_size(d)),
    "sub_group": _sub_group,
    "num_stages": _num_stages,
    "accel": lambda d: "xpu",
}
"""Placeholders a template may use, and how each is resolved from the definition."""


@dataclass(frozen=True)
class InTreeKernel:
    """One SYCL source, and the definition signature it implements.

    Parameters
    ----------
    name : str
        Short kernel name; becomes part of the generated solution's name.
    op_type, inputs, outputs
        The signature this kernel satisfies. Matched exactly, in order, for the same
        reason the upstream registry is strict: a kernel bound to the wrong definition
        runs cleanly and computes something else.
    dtypes : Tuple[str, ...]
        Input dtypes the template supports. Empty means every dtype in
        :data:`_SYCL_SCALARS`.
    language : SupportedLanguages
        Which builder compiles this template. Both SYCL and Triton are first-class: SYCL
        gives control over the memory access pattern, Triton gives a much shorter kernel
        and portability, and which wins is a per-op question rather than a policy.
    destination_passing_style : bool
        Whether outputs are pre-allocated and passed in. SYCL kernels here take them;
        Triton kernels allocate and return, which is the convention the Triton builder
        and the existing dataset solutions use.
    filename : str
        Source file name inside the generated solution.
    symbol : str
        Exported entry point, as passed to ``TVM_FFI_DLL_EXPORT_TYPED_FUNC``.
    source : str
        C++ template using ``$eps``, ``$scalar_t`` and ``$vec_size``.
    description : str
        What the kernel does, recorded on the solution.
    """

    name: str
    op_type: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    filename: str
    symbol: str
    source: str
    description: str
    dtypes: Tuple[str, ...] = ()
    language: SupportedLanguages = SupportedLanguages.SYCL
    destination_passing_style: bool = True
    fi_api: Optional[str] = None

    def matches(self, definition: Definition) -> bool:
        """Whether this kernel implements ``definition``."""
        if definition.op_type != self.op_type:
            return False
        if tuple(definition.inputs) != self.inputs or tuple(definition.outputs) != self.outputs:
            return False
        dtypes = self.dtypes or tuple(_SYCL_SCALARS)
        if not all(spec.dtype in dtypes for spec in definition.inputs.values()):
            return False
        # Every gated activation shares one signature, so identity has to come from the
        # definition's fi_api tag or a template binds to the wrong operation.
        if self.fi_api is None:
            return True
        return f"fi_api:{self.fi_api}" in definition.tags

    def render(self, definition: Definition) -> str:
        """The kernel source specialized for ``definition``.

        Only the placeholders this template actually uses are resolved. Resolving all of
        them would run every resolver -- including epsilon, which warns when a definition
        does not state one, and an activation definition has no reason to.
        """
        used = {
            m.group("named") or m.group("braced") for m in Template.pattern.finditer(self.source)
        }
        values = {k: fn(definition) for k, fn in _SUBSTITUTIONS.items() if k in used}
        return Template(self.source).substitute(values)


_RMSNORM_SYCL = r"""// RMSNorm in SYCL for Intel GPUs. Generated from a template.
//
// One work-group per row: each work-item accumulates a partial sum of squares over its
// slice, the work-group reduces them, and the row is rescaled.
//
// The access pattern is what matters. Reading one element per work-item leaves most of
// the memory pipe idle, and this kernel is bandwidth-bound at any interesting batch size;
// loading $vec_size elements at a time makes each access 16 bytes and measured 1.35x
// faster at prefill sizes on Battlemage. The row is deliberately re-read for the rescale
// rather than cached across the reduction -- caching it measured within noise, and the
// registers cost occupancy.

#include <sycl/sycl.hpp>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace fib_sycl {
namespace {

using scalar_t = $scalar_t;

constexpr int kVecSize = $vec_size;
constexpr int kSubGroup = 32;
constexpr size_t kMaxWorkGroup = 1024;
constexpr float kEps = $eps;

template <typename T, int N>
struct alignas(sizeof(T) * N) VecN {
  T val[N];
};

size_t work_group_for(int64_t items) {
  size_t wg = static_cast<size_t>((items + kSubGroup - 1) / kSubGroup) * kSubGroup;
  return std::min(std::max(wg, static_cast<size_t>(kSubGroup)), kMaxWorkGroup);
}

}  // namespace

void RmsNorm(tvm::ffi::TensorView hidden_states, tvm::ffi::TensorView weight,
             tvm::ffi::TensorView out) {
  TVM_FFI_ICHECK_EQ(hidden_states.ndim(), 2) << "hidden_states must be [batch, hidden]";
  TVM_FFI_ICHECK_EQ(weight.size(0), hidden_states.size(1)) << "weight/hidden mismatch";
  TVM_FFI_ICHECK_EQ(out.size(1), hidden_states.size(1)) << "out/hidden mismatch";

  const int64_t batch = hidden_states.size(0);
  const int64_t hidden = hidden_states.size(1);
  if (batch == 0 || hidden == 0) return;

  DLDevice dev = hidden_states.device();
  sycl::queue* q = static_cast<sycl::queue*>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";

  const scalar_t* __restrict__ x = static_cast<const scalar_t*>(hidden_states.data_ptr());
  const scalar_t* __restrict__ w = static_cast<const scalar_t*>(weight.data_ptr());
  scalar_t* __restrict__ o = static_cast<scalar_t*>(out.data_ptr());
  const float inv_hidden = 1.0f / static_cast<float>(hidden);

  constexpr size_t kAlign = sizeof(scalar_t) * kVecSize;
  const bool wide = (hidden % kVecSize == 0) &&
                    (reinterpret_cast<uintptr_t>(x) % kAlign == 0) &&
                    (reinterpret_cast<uintptr_t>(w) % kAlign == 0) &&
                    (reinterpret_cast<uintptr_t>(o) % kAlign == 0);

  const int64_t vec_items = wide ? hidden / kVecSize : 0;

  // Short rows: a row narrower than one sub-group leaves most of a work-group idle, and
  // one work-group per row means a launch per row. Pack several rows into a work-group
  // instead. Local ids linearize with the last dimension fastest, so a (rows, kSubGroup)
  // work-group puts each row in exactly one sub-group -- which is what lets the row be
  // reduced with a plain sub-group collective rather than a hand-written shuffle tree.
  if (wide && vec_items <= kSubGroup) {
    using Vec = VecN<scalar_t, kVecSize>;
    constexpr size_t kRowsPerGroup = 8;
    const size_t groups = (static_cast<size_t>(batch) + kRowsPerGroup - 1) / kRowsPerGroup;
    q->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(groups * kRowsPerGroup, kSubGroup),
                          sycl::range<2>(kRowsPerGroup, kSubGroup)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
          const int64_t row = static_cast<int64_t>(it.get_global_id(0));
          // Every lane of a sub-group shares a row, so this exit is sub-group uniform and
          // the collective below is still reached by all participating lanes.
          if (row >= batch) return;
          const size_t lane = it.get_local_id(1);
          const Vec* __restrict__ vin = reinterpret_cast<const Vec*>(x + row * hidden);
          const Vec* __restrict__ vw = reinterpret_cast<const Vec*>(w);
          Vec* __restrict__ vout = reinterpret_cast<Vec*>(o + row * hidden);

          float partial = 0.0f;
          for (int64_t i = static_cast<int64_t>(lane); i < vec_items; i += kSubGroup) {
            const Vec c = vin[i];
#pragma unroll
            for (int j = 0; j < kVecSize; ++j) {
              const float v = static_cast<float>(c.val[j]);
              partial += v * v;
            }
          }
          const float sum =
              sycl::reduce_over_group(it.get_sub_group(), partial, sycl::plus<float>());
          const float scale = sycl::rsqrt(sum * inv_hidden + kEps);

          for (int64_t i = static_cast<int64_t>(lane); i < vec_items; i += kSubGroup) {
            const Vec c = vin[i];
            const Vec g = vw[i];
            Vec d;
#pragma unroll
            for (int j = 0; j < kVecSize; ++j) {
              d.val[j] = static_cast<scalar_t>(static_cast<float>(c.val[j]) * scale *
                                               static_cast<float>(g.val[j]));
            }
            vout[i] = d;
          }
        });
    return;
  }

  if (wide) {
    using Vec = VecN<scalar_t, kVecSize>;
    const int64_t items = vec_items;
    const size_t wg = work_group_for(items);
    q->parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(batch) * wg, wg),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
          const int64_t row = static_cast<int64_t>(it.get_group(0));
          const size_t lid = it.get_local_id(0);
          const size_t lsize = it.get_local_range(0);
          const Vec* __restrict__ vin = reinterpret_cast<const Vec*>(x + row * hidden);
          const Vec* __restrict__ vw = reinterpret_cast<const Vec*>(w);
          Vec* __restrict__ vout = reinterpret_cast<Vec*>(o + row * hidden);

          float partial = 0.0f;
          for (int64_t i = static_cast<int64_t>(lid); i < items; i += static_cast<int64_t>(lsize)) {
            const Vec c = vin[i];
#pragma unroll
            for (int j = 0; j < kVecSize; ++j) {
              const float v = static_cast<float>(c.val[j]);
              partial += v * v;
            }
          }
          const float sum = sycl::reduce_over_group(it.get_group(), partial, sycl::plus<float>());
          const float scale = sycl::rsqrt(sum * inv_hidden + kEps);

          for (int64_t i = static_cast<int64_t>(lid); i < items; i += static_cast<int64_t>(lsize)) {
            const Vec c = vin[i];
            const Vec g = vw[i];
            Vec d;
#pragma unroll
            for (int j = 0; j < kVecSize; ++j) {
              d.val[j] = static_cast<scalar_t>(static_cast<float>(c.val[j]) * scale *
                                               static_cast<float>(g.val[j]));
            }
            vout[i] = d;
          }
        });
    return;
  }

  const size_t wg = work_group_for(hidden);
  q->parallel_for(sycl::nd_range<1>(static_cast<size_t>(batch) * wg, wg),
                  [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
                    const int64_t row = static_cast<int64_t>(it.get_group(0));
                    const size_t lid = it.get_local_id(0);
                    const size_t lsize = it.get_local_range(0);
                    const scalar_t* __restrict__ ri = x + row * hidden;
                    scalar_t* __restrict__ ro = o + row * hidden;

                    float partial = 0.0f;
                    for (int64_t i = static_cast<int64_t>(lid); i < hidden;
                         i += static_cast<int64_t>(lsize)) {
                      const float v = static_cast<float>(ri[i]);
                      partial += v * v;
                    }
                    const float sum =
                        sycl::reduce_over_group(it.get_group(), partial, sycl::plus<float>());
                    const float scale = sycl::rsqrt(sum * inv_hidden + kEps);
                    for (int64_t i = static_cast<int64_t>(lid); i < hidden;
                         i += static_cast<int64_t>(lsize)) {
                      ro[i] = static_cast<scalar_t>(static_cast<float>(ri[i]) * scale *
                                                    static_cast<float>(w[i]));
                    }
                  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm_sycl, RmsNorm);

}  // namespace fib_sycl
"""


_FUSED_ADD_RMSNORM_SYCL = r"""// Fused residual-add + RMSNorm in SYCL for Intel GPUs. Generated from a template.
//
// Computes rmsnorm(hidden_states + residual) * weight in one pass over memory. The
// definition declares a single output -- the normalized result -- so the summed residual
// is consumed here rather than written back, which is what makes the fusion worth having.
//
// Same access rules as the plain norm: $vec_size elements per load so each access is 16
// bytes, sub-group pinned to 32, work-group sized to the row.

#include <sycl/sycl.hpp>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace fib_sycl {
namespace {

using scalar_t = $scalar_t;

constexpr int kVecSize = $vec_size;
constexpr int kSubGroup = 32;
constexpr size_t kMaxWorkGroup = 1024;
constexpr float kEps = $eps;

template <typename T, int N>
struct alignas(sizeof(T) * N) VecN {
  T val[N];
};

size_t work_group_for(int64_t items) {
  size_t wg = static_cast<size_t>((items + kSubGroup - 1) / kSubGroup) * kSubGroup;
  return std::min(std::max(wg, static_cast<size_t>(kSubGroup)), kMaxWorkGroup);
}

}  // namespace

void FusedAddRmsNorm(tvm::ffi::TensorView hidden_states, tvm::ffi::TensorView residual,
                     tvm::ffi::TensorView weight, tvm::ffi::TensorView out) {
  TVM_FFI_ICHECK_EQ(hidden_states.ndim(), 2) << "hidden_states must be [batch, hidden]";
  TVM_FFI_ICHECK_EQ(residual.size(1), hidden_states.size(1)) << "residual/hidden mismatch";
  TVM_FFI_ICHECK_EQ(weight.size(0), hidden_states.size(1)) << "weight/hidden mismatch";

  const int64_t batch = hidden_states.size(0);
  const int64_t hidden = hidden_states.size(1);
  if (batch == 0 || hidden == 0) return;

  DLDevice dev = hidden_states.device();
  sycl::queue* q = static_cast<sycl::queue*>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";

  const scalar_t* __restrict__ x = static_cast<const scalar_t*>(hidden_states.data_ptr());
  const scalar_t* __restrict__ r = static_cast<const scalar_t*>(residual.data_ptr());
  const scalar_t* __restrict__ w = static_cast<const scalar_t*>(weight.data_ptr());
  scalar_t* __restrict__ o = static_cast<scalar_t*>(out.data_ptr());
  const float inv_hidden = 1.0f / static_cast<float>(hidden);

  constexpr size_t kAlign = sizeof(scalar_t) * kVecSize;
  const bool wide = (hidden % kVecSize == 0) &&
                    (reinterpret_cast<uintptr_t>(x) % kAlign == 0) &&
                    (reinterpret_cast<uintptr_t>(r) % kAlign == 0) &&
                    (reinterpret_cast<uintptr_t>(w) % kAlign == 0) &&
                    (reinterpret_cast<uintptr_t>(o) % kAlign == 0);

  if (wide) {
    using Vec = VecN<scalar_t, kVecSize>;
    const int64_t items = hidden / kVecSize;
    const size_t wg = work_group_for(items);
    q->parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(batch) * wg, wg),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
          const int64_t row = static_cast<int64_t>(it.get_group(0));
          const size_t lid = it.get_local_id(0);
          const size_t lsize = it.get_local_range(0);
          const Vec* __restrict__ vx = reinterpret_cast<const Vec*>(x + row * hidden);
          const Vec* __restrict__ vr = reinterpret_cast<const Vec*>(r + row * hidden);
          const Vec* __restrict__ vw = reinterpret_cast<const Vec*>(w);
          Vec* __restrict__ vout = reinterpret_cast<Vec*>(o + row * hidden);

          float partial = 0.0f;
          for (int64_t i = static_cast<int64_t>(lid); i < items; i += static_cast<int64_t>(lsize)) {
            const Vec a = vx[i];
            const Vec b = vr[i];
#pragma unroll
            for (int j = 0; j < kVecSize; ++j) {
              const float v = static_cast<float>(a.val[j]) + static_cast<float>(b.val[j]);
              partial += v * v;
            }
          }
          const float sum = sycl::reduce_over_group(it.get_group(), partial, sycl::plus<float>());
          const float scale = sycl::rsqrt(sum * inv_hidden + kEps);

          for (int64_t i = static_cast<int64_t>(lid); i < items; i += static_cast<int64_t>(lsize)) {
            const Vec a = vx[i];
            const Vec b = vr[i];
            const Vec g = vw[i];
            Vec d;
#pragma unroll
            for (int j = 0; j < kVecSize; ++j) {
              const float v = static_cast<float>(a.val[j]) + static_cast<float>(b.val[j]);
              d.val[j] = static_cast<scalar_t>(v * scale * static_cast<float>(g.val[j]));
            }
            vout[i] = d;
          }
        });
    return;
  }

  const size_t wg = work_group_for(hidden);
  q->parallel_for(sycl::nd_range<1>(static_cast<size_t>(batch) * wg, wg),
                  [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
                    const int64_t row = static_cast<int64_t>(it.get_group(0));
                    const size_t lid = it.get_local_id(0);
                    const size_t lsize = it.get_local_range(0);
                    const scalar_t* __restrict__ rx = x + row * hidden;
                    const scalar_t* __restrict__ rr = r + row * hidden;
                    scalar_t* __restrict__ ro = o + row * hidden;

                    float partial = 0.0f;
                    for (int64_t i = static_cast<int64_t>(lid); i < hidden;
                         i += static_cast<int64_t>(lsize)) {
                      const float v = static_cast<float>(rx[i]) + static_cast<float>(rr[i]);
                      partial += v * v;
                    }
                    const float sum =
                        sycl::reduce_over_group(it.get_group(), partial, sycl::plus<float>());
                    const float scale = sycl::rsqrt(sum * inv_hidden + kEps);
                    for (int64_t i = static_cast<int64_t>(lid); i < hidden;
                         i += static_cast<int64_t>(lsize)) {
                      const float v = static_cast<float>(rx[i]) + static_cast<float>(rr[i]);
                      ro[i] = static_cast<scalar_t>(v * scale * static_cast<float>(w[i]));
                    }
                  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_add_rmsnorm_sycl, FusedAddRmsNorm);

}  // namespace fib_sycl
"""


_TRITON_PREAMBLE = r"""# Generated from a template. Follows the patterns Intel uses in its own Triton
# benchmarks (intel-xpu-backend-for-triton, benchmarks/triton_kernels_benchmark):
#
#   * autotune over `warp_size` as well as `num_warps`. `warp_size` is an Intel knob with
#     no CUDA equivalent -- the device reports {16, 32} and which one wins is not
#     predictable from the shape. Triton accepts it in a Config without the kernel taking
#     it as a parameter.
#   * two-dimensional blocking, with rows-per-program derived from the device's
#     `max_work_group_size` rather than a constant, so a short row packs several rows into
#     one program instead of leaving most of a work-group idle.
#   * device properties read from the Triton driver, never assumed.
#
# Deliberately device-agnostic: everything is derived from the inputs and the driver, and
# no vendor runtime is named. Every Triton solution shipped in the dataset today fails on
# Intel for that reason alone -- a hand-written availability guard refuses before the
# kernel is ever reached, though the kernel itself would have run.

import torch
import triton
import triton.language as tl
from triton.runtime import driver

_PROPS = driver.active.utils.get_device_properties(torch.$accel.current_device())
MAX_WORK_GROUP_SIZE = _PROPS["max_work_group_size"]
SUB_GROUP_SIZES = tuple(_PROPS.get("sub_group_sizes", ($sub_group,)))

# warp_size x num_warps, the sweep Intel's own row-wise benchmarks use.
_CONFIGS = [
    triton.Config({"warp_size": ws}, num_warps=nw, num_stages=$num_stages)
    for ws in SUB_GROUP_SIZES
    for nw in (4, 8, 16, 32)
]


def _rows_per_program(block_x):
    # Rows per program, so a row narrower than a work-group does not waste one.
    return max(1, min(MAX_WORK_GROUP_SIZE // max(block_x, 1), 16))
"""


_RMSNORM_TRITON = _TRITON_PREAMBLE + r"""
EPS = $eps


@triton.autotune(configs=_CONFIGS, key=["hidden"], restore_value=["o_ptr"])
@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, o_ptr, batch, hidden, eps,
                    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
    row0 = tl.program_id(0) * BLOCK_Y
    cols = tl.arange(0, BLOCK_X)
    rows = tl.arange(0, BLOCK_Y)
    cmask = cols[None, :] < hidden
    rmask = (row0 + rows)[:, None] < batch
    mask = cmask & rmask
    offs = (row0 + rows)[:, None] * hidden + cols[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, axis=1) / hidden + eps)[:, None]
    w = tl.load(w_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)[None, :]
    tl.store(o_ptr + offs, (x * scale * w).to(o_ptr.dtype.element_ty), mask=mask)


def run(hidden_states, weight):
    batch, hidden = hidden_states.shape
    out = torch.empty_like(hidden_states)
    if batch == 0 or hidden == 0:
        return out
    block_x = triton.next_power_of_2(hidden)
    block_y = _rows_per_program(block_x)
    grid = (triton.cdiv(batch, block_y),)
    _rmsnorm_kernel[grid](hidden_states, weight, out, batch, hidden, EPS,
                          BLOCK_X=block_x, BLOCK_Y=block_y)
    return out
"""


_FUSED_ADD_RMSNORM_TRITON = _TRITON_PREAMBLE + r"""
EPS = $eps


@triton.autotune(configs=_CONFIGS, key=["hidden"], restore_value=["o_ptr"])
@triton.jit
def _fused_add_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, o_ptr, batch, hidden, eps,
                              BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
    row0 = tl.program_id(0) * BLOCK_Y
    cols = tl.arange(0, BLOCK_X)
    rows = tl.arange(0, BLOCK_Y)
    mask = (cols[None, :] < hidden) & ((row0 + rows)[:, None] < batch)
    offs = (row0 + rows)[:, None] * hidden + cols[None, :]

    v = (tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
         + tl.load(r_ptr + offs, mask=mask, other=0.0).to(tl.float32))
    scale = tl.rsqrt(tl.sum(v * v, axis=1) / hidden + eps)[:, None]
    w = tl.load(w_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)[None, :]
    tl.store(o_ptr + offs, (v * scale * w).to(o_ptr.dtype.element_ty), mask=mask)


def run(hidden_states, residual, weight):
    batch, hidden = hidden_states.shape
    out = torch.empty_like(hidden_states)
    if batch == 0 or hidden == 0:
        return out
    block_x = triton.next_power_of_2(hidden)
    block_y = _rows_per_program(block_x)
    grid = (triton.cdiv(batch, block_y),)
    _fused_add_rmsnorm_kernel[grid](hidden_states, residual, weight, out, batch, hidden,
                                    EPS, BLOCK_X=block_x, BLOCK_Y=block_y)
    return out
"""


_SILU_AND_MUL_SYCL = r"""// SwiGLU (silu_and_mul) in SYCL for Intel GPUs. Generated from a template.
//
// out[r, i] = silu(x[r, i]) * x[r, d + i]
//
// The halves are split, not interleaved: the gate is the first d columns and the
// up-projection the second d. A fused GEMM epilogue may hand you interleaved pairs
// instead -- same maths, incompatible layout, and no error if confused.
//
// Purely elementwise, so this is bandwidth-bound and the access width is what matters:
// $vec_size elements per load makes each access 16 bytes. Gate and up are read from
// positions d apart in the same row, so two loads feed one store.

#include <sycl/sycl.hpp>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace fib_sycl {
namespace {

using scalar_t = $scalar_t;

constexpr int kVecSize = $vec_size;
constexpr int kSubGroup = $sub_group;
constexpr size_t kMaxWorkGroup = 1024;

template <typename T, int N>
struct alignas(sizeof(T) * N) VecN {
  T val[N];
};

inline float silu(float v) { return v / (1.0f + sycl::exp(-v)); }

}  // namespace

void SiluAndMul(tvm::ffi::TensorView x, tvm::ffi::TensorView out) {
  TVM_FFI_ICHECK_EQ(x.ndim(), 2) << "x must be [batch, 2 * d]";
  TVM_FFI_ICHECK_EQ(out.ndim(), 2) << "out must be [batch, d]";
  const int64_t batch = x.size(0);
  const int64_t two_d = x.size(1);
  const int64_t d = out.size(1);
  TVM_FFI_ICHECK_EQ(two_d, 2 * d) << "x width must be twice out width";
  TVM_FFI_ICHECK_EQ(out.size(0), batch) << "batch mismatch";
  if (batch == 0 || d == 0) return;

  DLDevice dev = x.device();
  sycl::queue* q = static_cast<sycl::queue*>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";

  const scalar_t* __restrict__ xp = static_cast<const scalar_t*>(x.data_ptr());
  scalar_t* __restrict__ op = static_cast<scalar_t*>(out.data_ptr());

  constexpr size_t kAlign = sizeof(scalar_t) * kVecSize;
  const bool wide = (d % kVecSize == 0) &&
                    (reinterpret_cast<uintptr_t>(xp) % kAlign == 0) &&
                    (reinterpret_cast<uintptr_t>(op) % kAlign == 0);

  const int64_t total = batch * d;
  if (wide) {
    using Vec = VecN<scalar_t, kVecSize>;
    const int64_t items = total / kVecSize;
    const size_t wg = std::min(kMaxWorkGroup, static_cast<size_t>(kSubGroup) * 8);
    const size_t global = ((static_cast<size_t>(items) + wg - 1) / wg) * wg;
    q->parallel_for(sycl::nd_range<1>(global, wg),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
                      const int64_t i = static_cast<int64_t>(it.get_global_id(0));
                      if (i >= items) return;
                      // Flat index over [batch, d]; map back to find the paired half.
                      const int64_t base = i * kVecSize;
                      const int64_t row = base / d;
                      const int64_t col = base - row * d;
                      const scalar_t* g = xp + row * two_d + col;
                      const scalar_t* u = g + d;
                      const Vec gv = *reinterpret_cast<const Vec*>(g);
                      const Vec uv = *reinterpret_cast<const Vec*>(u);
                      Vec r;
#pragma unroll
                      for (int j = 0; j < kVecSize; ++j) {
                        r.val[j] = static_cast<scalar_t>(silu(static_cast<float>(gv.val[j])) *
                                                         static_cast<float>(uv.val[j]));
                      }
                      *reinterpret_cast<Vec*>(op + row * d + col) = r;
                    });
    return;
  }

  const size_t wg = std::min(kMaxWorkGroup, static_cast<size_t>(kSubGroup) * 8);
  const size_t global = ((static_cast<size_t>(total) + wg - 1) / wg) * wg;
  q->parallel_for(sycl::nd_range<1>(global, wg),
                  [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
                    const int64_t i = static_cast<int64_t>(it.get_global_id(0));
                    if (i >= total) return;
                    const int64_t row = i / d;
                    const int64_t col = i - row * d;
                    const float g = static_cast<float>(xp[row * two_d + col]);
                    const float u = static_cast<float>(xp[row * two_d + d + col]);
                    op[i] = static_cast<scalar_t>(silu(g) * u);
                  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(silu_and_mul_sycl, SiluAndMul);

}  // namespace fib_sycl
"""


_SILU_AND_MUL_TRITON = _TRITON_PREAMBLE + r"""

@triton.autotune(configs=_CONFIGS, key=["d"], restore_value=["o_ptr"])
@triton.jit
def _silu_and_mul_kernel(x_ptr, o_ptr, batch, d, two_d,
                         BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
    row0 = tl.program_id(0) * BLOCK_Y
    col0 = tl.program_id(1) * BLOCK_X
    cols = col0 + tl.arange(0, BLOCK_X)
    rows = row0 + tl.arange(0, BLOCK_Y)
    mask = (cols[None, :] < d) & (rows[:, None] < batch)

    gate_off = rows[:, None] * two_d + cols[None, :]
    up_off = gate_off + d
    g = tl.load(x_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + up_off, mask=mask, other=0.0).to(tl.float32)
    r = (g * tl.sigmoid(g)) * u
    tl.store(o_ptr + rows[:, None] * d + cols[None, :], r.to(o_ptr.dtype.element_ty), mask=mask)


def run(x):
    batch, two_d = x.shape
    d = two_d // 2
    out = torch.empty((batch, d), dtype=x.dtype, device=x.device)
    if batch == 0 or d == 0:
        return out
    # Elementwise over [batch, d], so tile it rather than giving each row a program:
    # a row here is the intermediate size, far wider than one work-group.
    block_x = min(triton.next_power_of_2(d), 1024)
    block_y = _rows_per_program(block_x)
    grid = (triton.cdiv(batch, block_y), triton.cdiv(d, block_x))
    _silu_and_mul_kernel[grid](x, out, batch, d, two_d, BLOCK_X=block_x, BLOCK_Y=block_y)
    return out
"""


REGISTRY: Tuple[InTreeKernel, ...] = (
    InTreeKernel(
        name="rmsnorm",
        op_type="rmsnorm",
        inputs=("hidden_states", "weight"),
        outputs=("output",),
        filename="rmsnorm_sycl.cpp",
        symbol="rmsnorm_sycl",
        source=_RMSNORM_SYCL,
        description=(
            "In-tree SYCL RMSNorm: 16-byte vectorized loads, sub-group pinned to 32, "
            "work-group sized to the row, float accumulation."
        ),
    ),
    InTreeKernel(
        name="fused_add_rmsnorm",
        op_type="rmsnorm",
        inputs=("hidden_states", "residual", "weight"),
        outputs=("output",),
        filename="fused_add_rmsnorm_sycl.cpp",
        symbol="fused_add_rmsnorm_sycl",
        source=_FUSED_ADD_RMSNORM_SYCL,
        description=(
            "In-tree SYCL fused residual-add + RMSNorm: one pass over memory, 16-byte "
            "vectorized loads, float accumulation."
        ),
    ),
    InTreeKernel(
        name="rmsnorm",
        op_type="rmsnorm",
        inputs=("hidden_states", "weight"),
        outputs=("output",),
        filename="main.py",
        symbol="run",
        source=_RMSNORM_TRITON,
        language=SupportedLanguages.TRITON,
        destination_passing_style=False,
        description=(
            "In-tree Triton RMSNorm: one program per row, single-block reduction, float "
            "accumulation. Device-agnostic -- no CUDA guards."
        ),
    ),
    InTreeKernel(
        name="fused_add_rmsnorm",
        op_type="rmsnorm",
        inputs=("hidden_states", "residual", "weight"),
        outputs=("output",),
        filename="main.py",
        symbol="run",
        source=_FUSED_ADD_RMSNORM_TRITON,
        language=SupportedLanguages.TRITON,
        destination_passing_style=False,
        description=(
            "In-tree Triton fused residual-add + RMSNorm. Device-agnostic -- no CUDA " "guards."
        ),
    ),
    InTreeKernel(
        name="silu_and_mul",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        filename="silu_and_mul_sycl.cpp",
        symbol="silu_and_mul_sycl",
        source=_SILU_AND_MUL_SYCL,
        fi_api="flashinfer.activation.silu_and_mul",
        description=(
            "In-tree SYCL SwiGLU: 16-byte vectorized loads of both halves, float "
            "activation, single store."
        ),
    ),
    InTreeKernel(
        name="silu_and_mul",
        op_type="activation",
        inputs=("x",),
        outputs=("out",),
        filename="main.py",
        symbol="run",
        source=_SILU_AND_MUL_TRITON,
        language=SupportedLanguages.TRITON,
        destination_passing_style=False,
        fi_api="flashinfer.activation.silu_and_mul",
        description=(
            "In-tree Triton SwiGLU: 2D tiling over [batch, d], autotuned over "
            "warp_size x num_warps. Device-agnostic."
        ),
    ),
)
"""Every in-tree kernel, matched against definitions by exact signature.

Both languages are carried for the same signature on purpose: a definition gets a SYCL and
a Triton solution, and the benchmark decides which is better on this hardware rather than
the registry deciding in advance.
"""

AUTHOR = "flashinfer-bench-intree"
"""Author recorded on generated solutions, so they are distinguishable in results."""


def find_kernels(definition: Definition) -> List[InTreeKernel]:
    """In-tree SYCL kernels implementing ``definition``."""
    return [k for k in REGISTRY if k.matches(definition)]


def make_intree_solution(definition: Definition, kernel: InTreeKernel) -> Solution:
    """Wrap an in-tree kernel as a Solution for ``definition``."""
    if not kernel.matches(definition):
        raise ValueError(
            f"SYCL kernel '{kernel.name}' does not implement '{definition.name}' "
            f"(expects inputs {kernel.inputs}, outputs {kernel.outputs})"
        )
    return Solution(
        name=f"{definition.name}__{kernel.language.value}_{kernel.name}",
        definition=definition.name,
        author=AUTHOR,
        spec=BuildSpec(
            language=kernel.language,
            target_hardware=["xpu"],
            entry_point=f"{kernel.filename}::{kernel.symbol}",
            destination_passing_style=kernel.destination_passing_style,
        ),
        sources=[SourceFile(path=kernel.filename, content=kernel.render(definition))],
        description=kernel.description,
    )


def make_intree_solutions(definition: Definition) -> List[Solution]:
    """Every in-tree SYCL solution for ``definition``."""
    return [make_intree_solution(definition, k) for k in find_kernels(definition)]


def explain_no_match(definition: Definition) -> List[str]:
    """Why no in-tree kernel covers ``definition``, as one line per near-miss."""
    if find_kernels(definition):
        return []
    reasons: List[str] = []
    for kernel in REGISTRY:
        if kernel.op_type != definition.op_type:
            continue
        got_in, got_out = tuple(definition.inputs), tuple(definition.outputs)
        if got_in != kernel.inputs:
            reasons.append(
                f"{kernel.language.value}/{kernel.name}: op_type matches, inputs differ "
                f"(kernel wants {kernel.inputs}, definition has {got_in})"
            )
        elif got_out != kernel.outputs:
            reasons.append(
                f"{kernel.language.value}/{kernel.name}: op_type and inputs match, outputs differ "
                f"(kernel wants {kernel.outputs}, definition has {got_out})"
            )
        elif kernel.fi_api is not None and f"fi_api:{kernel.fi_api}" not in definition.tags:
            # The common case for activations: identical signature, different operation.
            # Reporting a dtype problem here would send the reader somewhere useless.
            reasons.append(
                f"{kernel.language.value}/{kernel.name}: signature matches but this is a "
                f"different operation (kernel implements {kernel.fi_api})"
            )
        else:
            dtypes = sorted(str(spec.dtype) for spec in definition.inputs.values())
            reasons.append(
                f"{kernel.language.value}/{kernel.name}: signature matches but dtype(s) "
                f"{dtypes} are not supported"
            )
    return reasons
