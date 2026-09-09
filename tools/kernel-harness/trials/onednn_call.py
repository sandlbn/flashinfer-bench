"""The production GEMM's own library, called without the framework in between.

`aten.linear` on XPU reaches oneDNN through the dispatcher, autocast, `matmul`'s shape
handling and torch-xpu-ops' per-call descriptor construction before a primitive runs. At
decode sizes that host path is of the same order as the kernel, so it is one of the levers
a caller holds over a library call: keep the engine, stream and primitive alive across
calls and hand oneDNN the tensors directly on PyTorch's own queue.

Two flavours, chosen at build time:

    ba    the weight exactly as the model stores it, ``[N, K]`` row-major (oneDNN tag `ba`)
    any   ``format_tag::any`` for the weight, so the library picks its packed layout, with
          the reorder done once per weight and cached -- the layout lever the dispatch
          chain names

The kernel is the same shape of thing the routed harness runs: ``out = x @ w^T`` in the
activation dtype, nothing fused. Built through the repo's SYCL builder with the ``onednn``
dependency, so it links the oneAPI oneDNN rather than the copy inside torch; the verbose
banner names the version each arm ran, and the strategy line says whether they chose the
same kernel.

    from onednn_call import make
    Model, get_inputs, get_init_inputs = make("ba")          # library_call candidate
    Model, get_inputs, get_init_inputs = make_pair_control("ba")  # fusion control arm
"""

from __future__ import annotations

import importlib.util
import math
import os
import pathlib
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

BASE_ENV = "FIB_HARNESS_BASE"
POOL_ENV = "FIB_WEIGHT_POOL_MB"

SOURCE = r"""
#include <sycl/sycl.hpp>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <unordered_map>

namespace fib_trials {

using dnnl::memory;

constexpr int ANYWEI = @ANYWEI@;

struct Ctx {
  dnnl::engine engine;
  dnnl::stream stream;
};

Ctx& ctx_for(sycl::queue* q) {
  static std::unordered_map<sycl::queue*, Ctx> cache;
  auto it = cache.find(q);
  if (it != cache.end()) return it->second;
  dnnl::engine eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
  dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, *q);
  return cache.emplace(q, Ctx{std::move(eng), std::move(strm)}).first->second;
}

struct Key {
  int64_t m, n, k;
  bool operator==(const Key& o) const { return m == o.m && n == o.n && k == o.k; }
};
struct KeyHash {
  size_t operator()(const Key& x) const {
    return std::hash<int64_t>{}(x.m) ^ (std::hash<int64_t>{}(x.n) << 1) ^
           (std::hash<int64_t>{}(x.k) << 2);
  }
};

struct Prim {
  dnnl::matmul mm;
  memory::desc src_md, wei_md, dst_md;
};

// One packed copy of each distinct weight, made the first time that weight is seen.
struct Packed {
  memory mem;
};

void OneDnnLinear(tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView out) {
  TVM_FFI_ICHECK_EQ(x.ndim(), 2) << "x must be [M, K]";
  TVM_FFI_ICHECK_EQ(w.ndim(), 2) << "w must be [N, K]";
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  TVM_FFI_ICHECK_EQ(w.size(1), K) << "K mismatch";

  DLDevice dev = x.device();
  sycl::queue* q = static_cast<sycl::queue*>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";
  Ctx& ctx = ctx_for(q);

  static std::unordered_map<Key, Prim, KeyHash> prims;
  static std::unordered_map<const void*, Packed> packed;

  const auto bf16 = memory::data_type::bf16;
  Key key{M, N, K};
  auto it = prims.find(key);
  if (it == prims.end()) {
    memory::desc src_md({M, K}, bf16, memory::dims{K, 1});
    memory::desc dst_md({M, N}, bf16, memory::dims{N, 1});
    memory::desc wei_md = ANYWEI ? memory::desc({K, N}, bf16, memory::format_tag::any)
                                 : memory::desc({K, N}, bf16, memory::dims{1, K});
    dnnl::matmul::primitive_desc pd(ctx.engine, src_md, wei_md, dst_md);
    it = prims.emplace(key, Prim{dnnl::matmul(pd), pd.src_desc(), pd.weights_desc(), pd.dst_desc()}).first;
  }
  Prim& p = it->second;

  memory src(p.src_md, ctx.engine, x.data_ptr());
  memory dst(p.dst_md, ctx.engine, out.data_ptr());
  memory wei;
  if (ANYWEI) {
    auto pk = packed.find(w.data_ptr());
    if (pk == packed.end()) {
      memory::desc user_md({K, N}, bf16, memory::dims{1, K});
      memory user(user_md, ctx.engine, w.data_ptr());
      memory packed_mem(p.wei_md, ctx.engine);
      dnnl::reorder(user, packed_mem).execute(ctx.stream, user, packed_mem);
      pk = packed.emplace(w.data_ptr(), Packed{packed_mem}).first;
    }
    wei = pk->second.mem;
  } else {
    wei = memory(p.wei_md, ctx.engine, w.data_ptr());
  }
  p.mm.execute(ctx.stream, {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei}, {DNNL_ARG_DST, dst}});
  // No wait: the stream wraps PyTorch's queue, so ordering already holds.
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(onednn_linear, OneDnnLinear);

}  // namespace fib_trials
"""


def _definition():
    from flashinfer_bench.data import Definition

    return Definition.model_validate(
        {
            "name": "onednn_linear_direct",
            "op_type": "gemm",
            "description": "oneDNN matmul called directly on the framework queue (trial)",
            "axes": {k: {"type": "var"} for k in ("M", "K", "N")},
            "inputs": {
                "x": {"shape": ["M", "K"], "dtype": "bfloat16"},
                "w": {"shape": ["N", "K"], "dtype": "bfloat16"},
            },
            "outputs": {"out": {"shape": ["M", "N"], "dtype": "bfloat16"}},
            "reference": "import torch\n\ndef run(x, w):\n    return x @ w.t()\n",
        }
    )


_BUILT: Dict[str, Any] = {}


def build_kernel(variant: str):
    fn = _BUILT.get(variant)
    if fn is None:
        import sys

        sys.path.insert(0, "tools/kernel-harness")
        from sycl_harness import build

        src = SOURCE.replace("@ANYWEI@", "1" if variant == "any" else "0")
        fn = _BUILT[variant] = build(_definition(), src, dependencies=["onednn"])
    return fn


def _base_module():
    path = os.environ.get(BASE_ENV)
    if not path:
        raise SystemExit(f"{BASE_ENV} must name the harness this candidate is measured against.")
    spec = importlib.util.spec_from_file_location("onednn_call_base", pathlib.Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pool_copies(w: torch.Tensor) -> int:
    pool_mb = int(os.environ.get(POOL_ENV, "0"))
    if pool_mb <= 0:
        return 1
    return max(1, math.ceil(pool_mb * 2**20 / (w.numel() * w.element_size())))


class _Direct(nn.Module):
    def __init__(self, variant: str, consumer=None, replaces: int = 0):
        super().__init__()
        self.variant, self.consumer, self.replaces = variant, consumer, replaces
        self._ptr, self._pool, self._calls = None, [], 0

    def _weight(self, w: torch.Tensor) -> torch.Tensor:
        if w.data_ptr() != self._ptr:
            self._ptr = w.data_ptr()
            self._pool = [w] + [w.clone() for _ in range(_pool_copies(w) - 1)]
            self._calls = 0
        if len(self._pool) == 1:
            return w
        w_i = self._pool[self._calls % len(self._pool)]
        self._calls += 1
        return w_i

    def forward(self, x: torch.Tensor, w: torch.Tensor, *rest: torch.Tensor):
        w = self._weight(w)
        out = torch.empty(x.shape[0], w.shape[0], dtype=x.dtype, device=x.device)
        build_kernel(self.variant)(x, w, out)
        if self.consumer is None:
            return out
        args = list(rest)
        args.insert(self.replaces, out)
        return self.consumer(*args)


def make(variant: str):
    base = _base_module()

    class Model(_Direct):
        def __init__(self):
            super().__init__(variant)

    return Model, base.get_inputs, base.get_init_inputs


def make_pair_control(variant: str):
    base = _base_module()

    class Model(_Direct):
        def __init__(self):
            super().__init__(variant, consumer=base._Pair().consumer, replaces=int(base.REPLACES))

    return Model, base.get_inputs, base.get_init_inputs
