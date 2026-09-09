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

Sixteen bytes is the widest single access the memory pipe takes (``Capabilities.vector_bytes``
on a known device; a generated solution is shared, so it cannot query one). A memory-bound
kernel reading one element per work-item leaves most of that idle, which measures as a
clear penalty at prefill batch sizes.
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
    """Sub-group width to size an elementwise Triton or SYCL launch against.

    32 on every Intel AOT target today (Battlemage reports {16, 32}); the per-device value
    is ``Capabilities.preferred_sub_group_size`` for code running on a known device. A
    generated solution is shared, so it cannot bake in one machine's answer.

    Only elementwise templates substitute this. Matrix kernels pin 16 -- see
    ``_GEMM_CONFIGS`` for why sweeping the wider width there measures nothing.
    """
    return "32"


def _num_stages(definition: Definition) -> str:
    """Pipeline depth. CUDA uses this for async-copy pipelining, which Intel lacks; a
    sweep on Intel found 2 at or near the optimum across shapes and never worse than 1,
    while 3 regressed badly at small batch. Re-sweep with ``scripts/kernel_trials.py``
    before carrying this to a new part.
    """
    return "2"


_DNNL_DTYPES = {
    "bfloat16": "memory::data_type::bf16",
    "float16": "memory::data_type::f16",
    "float32": "memory::data_type::f32",
}


def _dnnl_dtype(definition: Definition) -> str:
    dtype = next(iter(definition.inputs.values())).dtype
    try:
        return _DNNL_DTYPES[dtype]
    except KeyError:
        raise ValueError(f"No oneDNN data type for dtype '{dtype}'") from None


def _block_size(definition: Definition, axis: str, name_key: str) -> str:
    """Weight-quantisation block extent along one dimension.

    Read from the definition rather than assumed, because it is what the scale tensor's
    shape means: `B_scale_inv` is [N_blocks, K_blocks], so a kernel that tiles by the
    wrong extent reads the wrong scale for every block and is silently wrong rather than
    slow. Prefer the derived value (`N / N_blocks`), and fall back to the
    `quantization:block{n}x{k}` tag only when the axes are absent.
    """
    axes = definition.axes
    full, blocks = axes.get(axis), axes.get(f"{axis}_blocks")
    full_v = getattr(full, "value", None)
    blocks_v = getattr(blocks, "value", None)
    if full_v and blocks_v:
        # Ceiling division on the way in, so recover the extent that produced it.
        extent = -(-int(full_v) // int(blocks_v))
        return str(extent)
    for tag in definition.tags:
        if tag.startswith("quantization:block"):
            n, _, k = tag[len("quantization:block") :].partition("x")
            if n.isdigit() and k.isdigit():
                return n if name_key == "N" else k
    raise ValueError(
        f"Definition '{definition.name}' declares no {axis}_blocks axis and no "
        "quantization:block{n}x{k} tag, so the weight block size is unknown."
    )


_DNNL_OUT_DTYPES = {
    "bfloat16": "memory::data_type::bf16",
    "float16": "memory::data_type::f16",
    "float32": "memory::data_type::f32",
}


def _dnnl_out_dtype(definition: Definition) -> str:
    """oneDNN type for the definition's output, which for a quantized GEMM differs from
    its inputs -- `$dnnl_dtype` reads the first *input*, which here is int8."""
    dtype = next(iter(definition.outputs.values())).dtype
    try:
        return _DNNL_OUT_DTYPES[dtype]
    except KeyError:
        raise ValueError(f"No oneDNN output type for dtype '{dtype}'") from None


_SUBSTITUTIONS = {
    "eps": lambda d: repr(definition_eps(d)),
    "dnnl_out_dtype": _dnnl_out_dtype,
    "block_n": lambda d: _block_size(d, "N", "N"),
    "block_k": lambda d: _block_size(d, "K", "K"),
    "dnnl_dtype": _dnnl_dtype,
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
    dependencies: Tuple[str, ...] = ()
    """Build dependencies the solution declares, e.g. ("onednn",).

    Without this a kernel that includes oneDNN headers compiles against whatever happens to
    be on the include path and fails to link with `cannot find -ldnnl`. SyclBuilder supplies
    the include, library and rpath flags from the declared dependency.
    """

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


_GEMM_ONEDNN_SYCL = r"""// In-tree GEMM via oneDNN, with cached engine, stream and primitive descriptors.
//
// On Intel, `F.linear` and `torch.matmul` are already oneDNN matmuls -- you cannot opt out
// of oneDNN, only tune how it is called. So this is not an attempt to beat oneDNN at matrix
// multiply; it is the vendor baseline made explicit, so a solution has something named to
// be measured against and so the primitive-rebuild cost is paid once rather than per call.
//
// The definition is C[M,N] = A[M,K] @ B[N,K]^T. B is passed as [N, K] row-major, which is
// described to oneDNN as a [K, N] matrix with strides {1, K} -- no copy, no transpose.

#include <sycl/sycl.hpp>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <unordered_map>

namespace fib_sycl {
namespace {

using dnnl::memory;

struct OneDnnContext {
  dnnl::engine engine;
  dnnl::stream stream;
};

OneDnnContext& context_for(sycl::queue* q) {
  // Engine and stream creation is not cheap and depends only on the queue.
  static std::unordered_map<sycl::queue*, OneDnnContext> cache;
  auto it = cache.find(q);
  if (it != cache.end()) return it->second;
  dnnl::engine eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
  dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, *q);
  return cache.emplace(q, OneDnnContext{std::move(eng), std::move(strm)}).first->second;
}

struct Key {
  int64_t m, n, k;
  bool operator==(const Key& o) const { return m == o.m && n == o.n && k == o.k; }
};
struct KeyHash {
  size_t operator()(const Key& x) const {
    return std::hash<int64_t>()(x.m) ^ (std::hash<int64_t>()(x.n) << 1) ^
           (std::hash<int64_t>()(x.k) << 2);
  }
};

}  // namespace

void GemmOneDnn(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
  TVM_FFI_ICHECK_EQ(A.ndim(), 2) << "A must be [M, K]";
  TVM_FFI_ICHECK_EQ(B.ndim(), 2) << "B must be [N, K]";
  TVM_FFI_ICHECK_EQ(A.size(1), B.size(1)) << "A/B inner dimension mismatch";

  const int64_t M = A.size(0), K = A.size(1), N = B.size(0);
  if (M == 0 || N == 0 || K == 0) return;

  DLDevice dev = A.device();
  sycl::queue* q = static_cast<sycl::queue*>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";
  OneDnnContext& ctx = context_for(q);

  const memory::data_type dt = $dnnl_dtype;

  // Primitive descriptor creation is expensive and depends only on the shape, so it is
  // cached; rebuilding it per call shows up as launch overhead at small M.
  static std::unordered_map<Key, dnnl::matmul, KeyHash> prim_cache;
  Key key{M, N, K};
  auto cached = prim_cache.find(key);
  if (cached == prim_cache.end()) {
    const memory::desc a_md({M, K}, dt, memory::dims{K, 1});
    const memory::desc b_md({K, N}, dt, memory::dims{1, K});  // B is [N,K]; view as [K,N]
    const memory::desc c_md({M, N}, dt, memory::dims{N, 1});
    dnnl::matmul::primitive_desc pd(ctx.engine, a_md, b_md, c_md);
    cached = prim_cache.emplace(key, dnnl::matmul(pd)).first;
  }

  memory a_mem(memory::desc({M, K}, dt, memory::dims{K, 1}), ctx.engine, A.data_ptr());
  memory b_mem(memory::desc({K, N}, dt, memory::dims{1, K}), ctx.engine, B.data_ptr());
  memory c_mem(memory::desc({M, N}, dt, memory::dims{N, 1}), ctx.engine, C.data_ptr());

  cached->second.execute(ctx.stream, {{DNNL_ARG_SRC, a_mem},
                                      {DNNL_ARG_WEIGHTS, b_mem},
                                      {DNNL_ARG_DST, c_mem}});
  // No stream.wait() here. The oneDNN stream is built over PyTorch's own SYCL queue via
  // sycl_interop, so work is ordered against everything else on that queue and the caller
  // synchronizes when it needs the result. Blocking per call is what torch does not do,
  // and it is pure host-side latency on every launch.
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_onednn_sycl, GemmOneDnn);

}  // namespace fib_sycl
"""


_RMSNORM_SYCL = r"""// RMSNorm in SYCL for Intel GPUs. Generated from a template.
//
// One work-group per row: each work-item accumulates a partial sum of squares over its
// slice, the work-group reduces them, and the row is rescaled.
//
// The access pattern is what matters. Reading one element per work-item leaves most of
// the memory pipe idle, and this kernel is bandwidth-bound at any interesting batch size;
// loading $vec_size elements at a time makes each access 16 bytes, which measured clearly
// faster at prefill sizes. The row is deliberately re-read for the rescale rather than
// cached across the reduction -- caching it measured within noise, and the registers cost
// occupancy.

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


_FUSED_ADD_RMSNORM_RESIDUAL_SYCL = r"""// Fused residual-add + RMSNorm in SYCL for Intel GPUs. Generated from a template.
//
// Computes rmsnorm(hidden_states + residual) * weight in one pass, and writes the summed
// residual back as a second output.
//
// The single-output variant discards that sum, which reads like a saving and is the
// opposite. Every caller that fuses a residual add needs the new residual for its next
// block -- vLLM's own kernel updates it in place for exactly this reason -- so discarding
// it forces the caller to recompute `x + residual` in eager PyTorch: a whole extra pass
// over [batch, hidden], which at prefill sizes more than doubled the call and handed back
// the kernel's entire win over the provider. Storing a value already in registers costs
// one store.
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

void FusedAddRmsNormResidual(tvm::ffi::TensorView hidden_states, tvm::ffi::TensorView residual,
                             tvm::ffi::TensorView weight, tvm::ffi::TensorView out,
                             tvm::ffi::TensorView residual_out) {
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
  scalar_t* __restrict__ ro_res = static_cast<scalar_t*>(residual_out.data_ptr());
  const float inv_hidden = 1.0f / static_cast<float>(hidden);

  constexpr size_t kAlign = sizeof(scalar_t) * kVecSize;
  const bool wide = (hidden % kVecSize == 0) &&
                    (reinterpret_cast<uintptr_t>(x) % kAlign == 0) &&
                    (reinterpret_cast<uintptr_t>(r) % kAlign == 0) &&
                    (reinterpret_cast<uintptr_t>(w) % kAlign == 0) &&
                    (reinterpret_cast<uintptr_t>(o) % kAlign == 0) &&
                    (reinterpret_cast<uintptr_t>(ro_res) % kAlign == 0);

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
          Vec* __restrict__ vres = reinterpret_cast<Vec*>(ro_res + row * hidden);

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
            Vec sres;
#pragma unroll
            for (int j = 0; j < kVecSize; ++j) {
              const float v = static_cast<float>(a.val[j]) + static_cast<float>(b.val[j]);
              sres.val[j] = static_cast<scalar_t>(v);
              d.val[j] = static_cast<scalar_t>(v * scale * static_cast<float>(g.val[j]));
            }
            vout[i] = d;
            vres[i] = sres;
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
                    scalar_t* __restrict__ rres = ro_res + row * hidden;

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
                      rres[i] = static_cast<scalar_t>(v);
                      ro[i] = static_cast<scalar_t>(v * scale * static_cast<float>(w[i]));
                    }
                  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_add_rmsnorm_residual_sycl, FusedAddRmsNormResidual);

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


# Tiles for GEMM. warp_size is NOT swept here, unlike the elementwise configs above:
# Triton's Intel backend overrides it to 16 for any kernel it can lower to DPAS, which
# every tl.dot kernel is (TritonAnnotateModule.cpp, setThreadsPerWarp -- it sets
# ttg::AttrNumThreadsPerWarp to minSGSize and returns, ignoring the option value). The
# override is silent, so sweeping {16, 32} compiles two identical kernels and measures
# the same one twice, doubling autotune time for no coverage. Pinned to make that
# explicit rather than accidental.
_GEMM_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "warp_size": 16},
                  num_warps=nw, num_stages=$num_stages)
    for bm, bn, bk in ((64, 64, 32), (128, 64, 32), (64, 128, 32), (128, 128, 32))
    for nw in (4, 8)
]


def _rows_per_program(block_x):
    # Rows per program, so a row narrower than a work-group does not waste one.
    return max(1, min(MAX_WORK_GROUP_SIZE // max(block_x, 1), 16))
"""


_GEMM_TRITON = _TRITON_PREAMBLE + r"""

@triton.autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # C[M,N] = A[M,K] @ B[N,K]^T. B is read transposed by indexing, never materialised.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k = k0 * BLOCK_K + offs_k
        a = tl.load(a_ptr + offs_m[:, None] * K + k[None, :],
                    mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_n[:, None] * K + k[None, :],
                    mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0.0)
        acc += tl.dot(a, tl.trans(b), allow_tf32=False)

    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(A, B):
    M, K = A.shape
    N = B.shape[0]
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    if M == 0 or N == 0 or K == 0:
        return C
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _gemm_kernel[grid](A, B, C, M, N, K)
    return C
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


_FUSED_ADD_RMSNORM_RESIDUAL_TRITON = _TRITON_PREAMBLE + r"""
EPS = $eps


@triton.autotune(configs=_CONFIGS, key=["hidden"], restore_value=["o_ptr", "ro_ptr"])
@triton.jit
def _fused_add_rmsnorm_residual_kernel(x_ptr, r_ptr, w_ptr, o_ptr, ro_ptr, batch, hidden, eps,
                                       BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
    row0 = tl.program_id(0) * BLOCK_Y
    cols = tl.arange(0, BLOCK_X)
    rows = tl.arange(0, BLOCK_Y)
    mask = (cols[None, :] < hidden) & ((row0 + rows)[:, None] < batch)
    offs = (row0 + rows)[:, None] * hidden + cols[None, :]

    v = (tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
         + tl.load(r_ptr + offs, mask=mask, other=0.0).to(tl.float32))
    # The summed residual is what the caller's next block adds to, so it is stored rather
    # than recomputed there -- one store against a whole extra pass over [batch, hidden].
    tl.store(ro_ptr + offs, v.to(ro_ptr.dtype.element_ty), mask=mask)
    scale = tl.rsqrt(tl.sum(v * v, axis=1) / hidden + eps)[:, None]
    w = tl.load(w_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)[None, :]
    tl.store(o_ptr + offs, (v * scale * w).to(o_ptr.dtype.element_ty), mask=mask)


def run(hidden_states, residual, weight):
    batch, hidden = hidden_states.shape
    out = torch.empty_like(hidden_states)
    residual_out = torch.empty_like(residual)
    if batch == 0 or hidden == 0:
        return out, residual_out
    block_x = triton.next_power_of_2(hidden)
    block_y = _rows_per_program(block_x)
    grid = (triton.cdiv(batch, block_y),)
    _fused_add_rmsnorm_residual_kernel[grid](hidden_states, residual, weight, out, residual_out,
                                            batch, hidden, EPS, BLOCK_X=block_x, BLOCK_Y=block_y)
    return out, residual_out
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


_GEMM_FP8_BLOCK_PYTHON = r"""import torch

# Block-scaled FP8 W8A8 GEMM. Both operands arrive quantized:
#   A_fp8 [M, K] with A_scale [M, K/BLOCK_K] per token-group,
#   B_fp8 [N, K] with B_scale [N/BLOCK_N, K/BLOCK_K] per weight tile.
#
# The activation scale is per (token, K-group), so unlike the weight scale it cannot be
# folded into a single pre-scaled operand and reused across rows. Applying both scales to
# the float32 accumulator one K-block at a time keeps them out of the low-precision
# operands entirely, which is where the accuracy of the pre-scaled form went.
#
# This is a placeholder in plain PyTorch. oneDNN takes grouped scales natively --
# `set_scales(DNNL_ARG_WEIGHTS, mask, {BLOCK_K, BLOCK_N})` with f8_e4m3 operands
# dispatches to jit:gemm:any on Battlemage -- and that is where this kernel belongs.
BLOCK_N = $block_n
BLOCK_K = $block_k


def run(A_fp8, A_scale, B_fp8, B_scale):
    N, K = B_fp8.shape
    n_blk, k_blk = -(-N // BLOCK_N), -(-K // BLOCK_K)
    a = A_fp8.to(torch.bfloat16)
    b = B_fp8.to(torch.bfloat16)
    a_s = A_scale.to(torch.float32)
    b_s = B_scale.to(torch.float32)

    acc = None
    for kb in range(k_blk):
        sl = slice(kb * BLOCK_K, min((kb + 1) * BLOCK_K, K))
        part = torch.matmul(a[:, sl], b[:, sl].T).to(torch.float32)
        # Row scale for this K-group, and column scale for each N-block within it.
        part *= a_s[:, kb].unsqueeze(1)
        part *= b_s[:, kb].repeat_interleave(BLOCK_N)[:N].unsqueeze(0)
        acc = part if acc is None else acc + part
    return acc.to(torch.bfloat16)
"""


_GEMM_SWIGLU_ONEDNN_SYCL = r"""// Fused GEMM + SwiGLU via oneDNN post-ops.
//
// SwiGLU is a pairwise reduction -- two accumulator lanes combine into one output -- which
// oneDNN post-ops, being elementwise or binary on a single GEMM's output, cannot express
// directly. Splitting into two matmuls makes it expressible at identical total FLOPs to
// one [M, 2D] GEMM:
//
//     up  = x @ Wu
//     out = swish(x @ Wg) * up
//
// The second matmul carries the whole activation in its epilogue: eltwise_swish, then
// binary_mul by `up`. Post-ops apply in declaration order.
//
// This keeps oneDNN's matmul rather than replacing it. A CUTLASS-SYCL fused kernel lost to
// the unfused path here not because fusion is wrong but because its GEMM was markedly
// slower than oneDNN's on the part it was tried on -- what fusion saved on the activation
// kernel it paid back twice over on the matmul.

#include <sycl/sycl.hpp>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <unordered_map>

namespace fib_sycl {
namespace {

using dnnl::memory;

struct OneDnnContext {
  dnnl::engine engine;
  dnnl::stream stream;
};

OneDnnContext& context_for(sycl::queue* q) {
  static std::unordered_map<sycl::queue*, OneDnnContext> cache;
  auto it = cache.find(q);
  if (it != cache.end()) return it->second;
  dnnl::engine eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
  dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, *q);
  return cache.emplace(q, OneDnnContext{std::move(eng), std::move(strm)}).first->second;
}

memory::desc row_major(const memory::dims& dims, memory::data_type dt) {
  memory::dims strides(dims.size(), 1);
  for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return memory::desc(dims, dt, strides);
}

struct Key {
  int64_t m, k, d;
  bool operator==(const Key& o) const { return m == o.m && k == o.k && d == o.d; }
};
struct KeyHash {
  size_t operator()(const Key& x) const {
    return std::hash<int64_t>()(x.m) ^ (std::hash<int64_t>()(x.k) << 1) ^
           (std::hash<int64_t>()(x.d) << 2);
  }
};
struct Cached {
  dnnl::matmul up_mm;
  dnnl::matmul out_mm;
};

}  // namespace

void GemmSwiGLUOneDnn(tvm::ffi::TensorView x, tvm::ffi::TensorView wg, tvm::ffi::TensorView wu,
                      tvm::ffi::TensorView up, tvm::ffi::TensorView out) {
  TVM_FFI_ICHECK_EQ(x.ndim(), 2) << "x must be [M, K]";
  TVM_FFI_ICHECK_EQ(wg.ndim(), 2) << "gate weight must be [K, D]";
  TVM_FFI_ICHECK_EQ(wu.ndim(), 2) << "up weight must be [K, D]";

  const int64_t M = x.size(0), K = x.size(1), D = wg.size(1);
  TVM_FFI_ICHECK_EQ(wg.size(0), K) << "gate weight rows must match x's inner dim";
  TVM_FFI_ICHECK_EQ(wu.size(0), K) << "up weight rows must match x's inner dim";
  TVM_FFI_ICHECK_EQ(wu.size(1), D) << "gate and up must have the same width";
  if (M == 0 || D == 0 || K == 0) return;

  DLDevice dev = x.device();
  sycl::queue* q = static_cast<sycl::queue*>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";
  OneDnnContext& ctx = context_for(q);

  const auto dt = $dnnl_dtype;
  auto x_md = row_major({M, K}, dt);
  auto w_md = row_major({K, D}, dt);
  auto o_md = row_major({M, D}, dt);

  memory x_mem(x_md, ctx.engine, x.data_ptr());
  memory wg_mem(w_md, ctx.engine, wg.data_ptr());
  memory wu_mem(w_md, ctx.engine, wu.data_ptr());
  memory up_mem(o_md, ctx.engine, up.data_ptr());
  memory out_mem(o_md, ctx.engine, out.data_ptr());

  // Primitive descriptors depend only on the shape, so build them once. Rebuilding per
  // call shows up as launch overhead at small M, which is where decode lives.
  static std::unordered_map<Key, Cached, KeyHash> prim_cache;
  Key key{M, K, D};
  auto cached = prim_cache.find(key);
  if (cached == prim_cache.end()) {
    dnnl::matmul::primitive_desc pd_up(ctx.engine, x_md, w_md, o_md);

    dnnl::post_ops po;
    po.append_eltwise(dnnl::algorithm::eltwise_swish, 1.0f, 0.0f);
    po.append_binary(dnnl::algorithm::binary_mul, o_md);
    dnnl::primitive_attr attr;
    attr.set_post_ops(po);
    dnnl::matmul::primitive_desc pd_out(ctx.engine, x_md, w_md, o_md, attr);

    cached = prim_cache.emplace(key, Cached{dnnl::matmul(pd_up), dnnl::matmul(pd_out)}).first;
  }

  // `up` must be complete before the second matmul multiplies by it. Both run on the
  // framework's queue, which is in-order, so that ordering is already guaranteed.
  cached->second.up_mm.execute(ctx.stream, {{DNNL_ARG_SRC, x_mem},
                                            {DNNL_ARG_WEIGHTS, wu_mem},
                                            {DNNL_ARG_DST, up_mem}});

  cached->second.out_mm.execute(
      ctx.stream, {{DNNL_ARG_SRC, x_mem},
                   {DNNL_ARG_WEIGHTS, wg_mem},
                   {DNNL_ARG_DST, out_mem},
                   {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, up_mem}});

  // No stream.wait(): the oneDNN stream is built over PyTorch's own queue, so this work is
  // ordered against everything else on it and the caller synchronizes when it needs the
  // result. Blocking here is pure host-side latency on every launch -- removing it took
  // this kernel from losing to the reference at M=1 to beating it several-fold.
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_swiglu_onednn_sycl, GemmSwiGLUOneDnn);

}  // namespace fib_sycl
"""


_GEMM_SWIGLU_TRITON = _TRITON_PREAMBLE + r"""

@triton.autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _gemm_swiglu_kernel(x_ptr, wg_ptr, wu_ptr, up_ptr, out_ptr, M, N, K,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # up  = x @ Wu ; out = silu(x @ Wg) * up, with x [M, K] and both weights [K, N].
    #
    # Both accumulators are built in one K loop so the x tile is loaded once and serves
    # both projections. That is the advantage over the oneDNN post-op route, which runs
    # two matmuls and therefore reads x twice.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k = k0 * BLOCK_K + offs_k
        x = tl.load(x_ptr + offs_m[:, None] * K + k[None, :],
                    mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        wmask = (k[:, None] < K) & (offs_n[None, :] < N)
        wg = tl.load(wg_ptr + k[:, None] * N + offs_n[None, :], mask=wmask, other=0.0)
        wu = tl.load(wu_ptr + k[:, None] * N + offs_n[None, :], mask=wmask, other=0.0)
        acc_g += tl.dot(x, wg, allow_tf32=False)
        acc_u += tl.dot(x, wu, allow_tf32=False)

    # Round both projections to the activation dtype *before* the activation. The
    # definition specifies `silu(x @ wg) * (x @ wu)` where each matmul produces the input
    # dtype, so the rounding is part of the operation, not an implementation detail.
    # Activating the unrounded float32 accumulator is more precise and disagrees with the
    # reference by more than tolerance once M is large enough for cancellation to show --
    # it passed at M=1 and failed from M=1102 up. oneDNN's post-op route matches by
    # construction, since the post-op reads the bfloat16 destination.
    g = acc_g.to(out_ptr.dtype.element_ty).to(tl.float32)
    u = acc_u.to(up_ptr.dtype.element_ty).to(tl.float32)
    out = (g * tl.sigmoid(g)) * u
    store_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(up_ptr + offs_m[:, None] * N + offs_n[None, :],
             u.to(up_ptr.dtype.element_ty), mask=store_mask)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
             out.to(out_ptr.dtype.element_ty), mask=store_mask)


def run(x, wg, wu):
    M, K = x.shape
    N = wg.shape[1]
    up = torch.empty((M, N), dtype=x.dtype, device=x.device)
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M == 0 or N == 0 or K == 0:
        return up, out
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _gemm_swiglu_kernel[grid](x, wg, wu, up, out, M, N, K)
    return up, out
"""


_GEMM_INT8_W8A8_ONEDNN_SYCL = r"""// W8A8 int8 GEMM via oneDNN, with both scalings in the epilogue.
//
// Unlike a block-scaled FP8 checkpoint, the scales here do not vary along K: the weight
// carries one scale per output channel and the activation one per token. So this is a
// single matmul, not a per-K-block accumulation, and it avoids the accumulator traffic
// that makes the block-scaled decomposition bandwidth-bound.
//
// Both scalings use configurations verified exact on this hardware (see /optimize-onednn):
// an ungrouped per-column weight scale, and the per-row activation scale as a binary_mul
// post-op. oneDNN rejects source scales outright, which is why the activation scale is a
// post-op rather than an attribute.

#include <sycl/sycl.hpp>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <unordered_map>
#include <vector>

namespace fib_sycl {
namespace {

using dnnl::memory;

struct OneDnnContext {
  dnnl::engine engine;
  dnnl::stream stream;
};

OneDnnContext& context_for(sycl::queue* q) {
  static std::unordered_map<sycl::queue*, OneDnnContext> cache;
  auto it = cache.find(q);
  if (it != cache.end()) return it->second;
  dnnl::engine eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
  dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, *q);
  return cache.emplace(q, OneDnnContext{std::move(eng), std::move(strm)}).first->second;
}

struct Key {
  int64_t m, n, k;
  bool operator==(const Key& o) const { return m == o.m && n == o.n && k == o.k; }
};
struct KeyHash {
  size_t operator()(const Key& x) const {
    return std::hash<int64_t>()(x.m) ^ (std::hash<int64_t>()(x.n) << 1) ^
           (std::hash<int64_t>()(x.k) << 2);
  }
};

}  // namespace

void GemmInt8W8A8OneDnn(tvm::ffi::TensorView A, tvm::ffi::TensorView A_scale,
                        tvm::ffi::TensorView B, tvm::ffi::TensorView B_scale,
                        tvm::ffi::TensorView C) {
  TVM_FFI_ICHECK_EQ(A.ndim(), 2) << "A must be [M, K]";
  TVM_FFI_ICHECK_EQ(B.ndim(), 2) << "B must be [N, K]";
  TVM_FFI_ICHECK_EQ(A.size(1), B.size(1)) << "A/B inner dimension mismatch";

  const int64_t M = A.size(0), K = A.size(1), N = B.size(0);
  if (M == 0 || N == 0 || K == 0) return;

  DLDevice dev = A.device();
  sycl::queue* q = static_cast<sycl::queue*>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";
  OneDnnContext& ctx = context_for(q);

  const auto s8 = memory::data_type::s8;
  const auto f32 = memory::data_type::f32;
  const auto out_dt = $dnnl_out_dtype;

  const memory::desc a_md({M, K}, s8, memory::dims{K, 1});
  const memory::desc b_md({K, N}, s8, memory::dims{1, K});  // B is [N,K]; view as [K,N]
  const memory::desc c_md({M, N}, out_dt, memory::dims{N, 1});
  const memory::desc arow_md({M, 1}, f32, memory::dims{1, 1});  // broadcasts over N

  static std::unordered_map<Key, dnnl::matmul, KeyHash> prim_cache;
  Key key{M, N, K};
  auto cached = prim_cache.find(key);
  if (cached == prim_cache.end()) {
    dnnl::primitive_attr attr;
    // Per-column (per-output-channel) weight scale: ungrouped, which is the form verified
    // exact here. The grouped form is silently wrong on this hardware.
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 1);
    dnnl::post_ops po;
    po.append_binary(dnnl::algorithm::binary_mul, arow_md);
    attr.set_post_ops(po);
    dnnl::matmul::primitive_desc pd(ctx.engine, a_md, b_md, c_md, attr);
    cached = prim_cache.emplace(key, dnnl::matmul(pd)).first;
  }

  memory a_mem(a_md, ctx.engine, A.data_ptr());
  memory b_mem(b_md, ctx.engine, B.data_ptr());
  memory c_mem(c_md, ctx.engine, C.data_ptr());
  // The weight scale is declared [N, 1] but oneDNN reads scale memory densely; N
  // contiguous floats is the same bytes either way.
  memory bs_mem(memory::desc({N}, f32, memory::dims{1}), ctx.engine, B_scale.data_ptr());
  memory as_mem(arow_md, ctx.engine, A_scale.data_ptr());

  cached->second.execute(ctx.stream,
                         {{DNNL_ARG_SRC, a_mem},
                          {DNNL_ARG_WEIGHTS, b_mem},
                          {DNNL_ARG_DST, c_mem},
                          {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, bs_mem},
                          {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, as_mem}});
  // No stream.wait(): the oneDNN stream is built over PyTorch's own queue, so this is
  // ordered against everything else on it and the caller synchronizes when it needs the
  // result.
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_int8_w8a8_onednn_sycl, GemmInt8W8A8OneDnn);

}  // namespace fib_sycl
"""


_GEMM_INT8_W8A8_TRITON = _TRITON_PREAMBLE + r"""

@triton.autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _gemm_int8_w8a8_kernel(a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, M, N, K,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_K: tl.constexpr):
    # C = (A_int8 * a_scale[m]) @ (B_int8 * b_scale[n])^T, with A [M,K] and B [N,K].
    #
    # Both scales are outer -- one per row of A, one per row of B -- so they factor out of
    # the reduction entirely: accumulate the integer product exactly in int32, then apply
    # both scales once. Scaling the operands first would round K times instead.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k = k0 * BLOCK_K + offs_k
        a = tl.load(a_ptr + offs_m[:, None] * K + k[None, :],
                    mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(b_ptr + offs_n[:, None] * K + k[None, :],
                    mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0)
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.int32)

    a_scale = tl.load(as_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    b_scale = tl.load(bs_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    out = acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], out.to(c_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(A_int8, A_scale, B_int8, B_scale):
    M, K = A_int8.shape
    N = B_int8.shape[0]
    C = torch.empty((M, N), dtype=torch.bfloat16, device=A_int8.device)
    if M == 0 or N == 0 or K == 0:
        return C
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _gemm_int8_w8a8_kernel[grid](A_int8, A_scale.reshape(-1).contiguous(), B_int8,
                                 B_scale.reshape(-1).contiguous(), C, M, N, K)
    return C
"""


REGISTRY: Tuple[InTreeKernel, ...] = (
    InTreeKernel(
        name="gemm_int8_w8a8_onednn",
        op_type="gemm",
        inputs=("A_int8", "A_scale", "B_int8", "B_scale"),
        outputs=("C",),
        filename="main.py",
        symbol="run",
        source=_GEMM_INT8_W8A8_TRITON,
        language=SupportedLanguages.TRITON,
        destination_passing_style=False,
        dtypes=("int8", "float32", "bfloat16", "float16"),
        description=(
            "In-tree W8A8 int8 GEMM in Triton: int32 accumulation, both outer scales "
            "applied once at the end rather than K times on the operands."
        ),
    ),
    InTreeKernel(
        name="gemm_int8_w8a8_onednn",
        op_type="gemm",
        inputs=("A_int8", "A_scale", "B_int8", "B_scale"),
        outputs=("C",),
        filename="gemm_int8_w8a8_onednn_sycl.cpp",
        symbol="gemm_int8_w8a8_onednn_sycl",
        source=_GEMM_INT8_W8A8_ONEDNN_SYCL,
        dependencies=("onednn",),
        dtypes=("int8", "float32", "bfloat16", "float16"),
        description=(
            "In-tree W8A8 int8 GEMM via oneDNN: per-channel weight scale as an ungrouped "
            "column scale, per-token activation scale as a binary_mul post-op. One matmul "
            "-- the scales do not vary along K."
        ),
    ),
    InTreeKernel(
        name="gemm_swiglu_onednn",
        op_type="gemm",
        inputs=("x", "wg", "wu"),
        outputs=("up", "out"),
        filename="main.py",
        symbol="run",
        source=_GEMM_SWIGLU_TRITON,
        language=SupportedLanguages.TRITON,
        destination_passing_style=False,
        description=(
            "In-tree fused GEMM + SwiGLU in Triton. Both projections share one K loop, so "
            "the activation tile is read once rather than twice."
        ),
    ),
    InTreeKernel(
        name="gemm_swiglu_onednn",
        op_type="gemm",
        inputs=("x", "wg", "wu"),
        outputs=("up", "out"),
        filename="gemm_swiglu_onednn_sycl.cpp",
        symbol="gemm_swiglu_onednn_sycl",
        source=_GEMM_SWIGLU_ONEDNN_SYCL,
        dependencies=("onednn",),
        description=(
            "In-tree fused GEMM + SwiGLU via oneDNN post-ops. Two matmuls at the same total "
            "FLOPs as one [M, 2D] GEMM, with the activation in the second one's epilogue."
        ),
    ),
    InTreeKernel(
        name="gemm_fp8_w8a8_block_dequant",
        op_type="gemm",
        inputs=("A_fp8", "A_scale", "B_fp8", "B_scale"),
        outputs=("C",),
        filename="main.py",
        symbol="run",
        source=_GEMM_FP8_BLOCK_PYTHON,
        language=SupportedLanguages.PYTHON,
        # Declared explicitly: the default set is the SYCL scalar types, which have no
        # fp8, so without this the kernel is refused for the only definitions it exists
        # to serve. Battlemage has no FP8 DPAS -- the packed weight is dequantised to
        # bf16 here and the GEMM runs in bf16, so what the device must support is
        # bfloat16, plus enough fp8 handling to cast. It has both.
        dtypes=("bfloat16", "float32", "float8_e4m3fn", "float8_e5m2"),
        destination_passing_style=False,
        description=(
            "In-tree block-scaled FP8 W8A8 GEMM in plain PyTorch: both scales are applied "
            "to the float32 accumulator per K-block, so neither rounds through bfloat16. "
            "A placeholder -- oneDNN takes grouped scales natively on this hardware and is "
            "where this belongs."
        ),
    ),
    InTreeKernel(
        name="gemm_onednn",
        op_type="gemm",
        inputs=("A", "B"),
        outputs=("C",),
        filename="gemm_onednn_sycl.cpp",
        symbol="gemm_onednn_sycl",
        source=_GEMM_ONEDNN_SYCL,
        dependencies=("onednn",),
        description=(
            "In-tree GEMM via oneDNN with cached engine, stream and primitive descriptors. "
            "oneDNN is what F.linear already calls on Intel, so this is the vendor baseline "
            "made explicit, with the per-call primitive rebuild removed."
        ),
    ),
    InTreeKernel(
        name="gemm_onednn",
        op_type="gemm",
        inputs=("A", "B"),
        outputs=("C",),
        filename="main.py",
        symbol="run",
        source=_GEMM_TRITON,
        language=SupportedLanguages.TRITON,
        destination_passing_style=False,
        description=(
            "In-tree Triton GEMM, tiles swept together with Intel's warp_size knob. "
            "Device-agnostic -- no CUDA guards."
        ),
    ),
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
        name="fused_add_rmsnorm_residual",
        op_type="rmsnorm",
        inputs=("hidden_states", "residual", "weight"),
        outputs=("output", "residual_out"),
        filename="fused_add_rmsnorm_residual_sycl.cpp",
        symbol="fused_add_rmsnorm_residual_sycl",
        source=_FUSED_ADD_RMSNORM_RESIDUAL_SYCL,
        description=(
            "In-tree SYCL fused residual-add + RMSNorm returning the summed residual as a "
            "second output, so the caller need not recompute it."
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
        name="fused_add_rmsnorm_residual",
        op_type="rmsnorm",
        inputs=("hidden_states", "residual", "weight"),
        outputs=("output", "residual_out"),
        filename="main.py",
        symbol="run",
        source=_FUSED_ADD_RMSNORM_RESIDUAL_TRITON,
        language=SupportedLanguages.TRITON,
        destination_passing_style=False,
        description=(
            "In-tree Triton fused residual-add + RMSNorm returning the summed residual as "
            "a second output. Device-agnostic -- no CUDA guards."
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
            dependencies=list(kernel.dependencies),
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
