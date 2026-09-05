// Fused GEMM + RMSNorm scale + SwiGLU using oneDNN post-ops.
//
// Why this exists: profiling showed the CUTLASS-SYCL fused kernel losing to vLLM's
// unfused path not because fusion is wrong, but because the GEMM underneath it is ~45%
// slower than oneDNN on this device. Fusion saved a 1.17 ms activation kernel while
// paying ~2.4 ms extra on the matmul.
//
// So: keep oneDNN's matmul, and fuse the epilogue with oneDNN's own post-op mechanism.
//
// oneDNN post-ops are elementwise or binary on the GEMM output, so they cannot express
// SwiGLU's pairwise reduction (two accumulator lanes -> one output) directly. Splitting
// into two matmuls makes it expressible, at identical total FLOPs to one [M, 2d] GEMM:
//
//     up  = (x @ Wu) * r[m]
//     out = swish((x @ Wg) * r[m]) * up
//
// The second matmul carries three post-ops in order: binary_mul by the row scale,
// eltwise_swish, binary_mul by `up`. That is the whole activation, fused.
//
// A side benefit over the CUTLASS route: no interleaving. Gate and up stay as the
// separate matrices a model already ships, so this needs no weight re-layout at all.

#include <sycl/sycl.hpp>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <unordered_map>

namespace fib_examples {

namespace {

using dnnl::memory;

// oneDNN engine/stream creation is not free, and the queue is stable for the life of the
// process, so both are cached. Rebuilding them per call would show up as launch overhead
// and swamp what this kernel is trying to save.
struct OneDnnContext {
  dnnl::engine engine;
  dnnl::stream stream;
};

// Primitive descriptors are built by querying the library for an implementation, which is
// not cheap and depends only on the problem shape. Rebuilding them per call showed up as
// fixed overhead that dominated at small M (0.90x at M=512 against vLLM, versus 1.16x at
// M=2048). Keyed on the shape so a changing batch size still gets a correct primitive.
struct MatmulKey {
  int64_t m, k, d;
  bool operator==(const MatmulKey& o) const { return m == o.m && k == o.k && d == o.d; }
};

struct MatmulKeyHash {
  size_t operator()(const MatmulKey& x) const {
    return std::hash<int64_t>{}(x.m) ^ (std::hash<int64_t>{}(x.k) << 1) ^
           (std::hash<int64_t>{}(x.d) << 2);
  }
};

struct CachedPrimitives {
  dnnl::matmul up_mm;
  dnnl::matmul out_mm;
};

OneDnnContext& context_for(sycl::queue* q) {
  static std::unordered_map<sycl::queue*, OneDnnContext> cache;
  auto it = cache.find(q);
  if (it != cache.end()) return it->second;
  dnnl::engine eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
  dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, *q);
  return cache.emplace(q, OneDnnContext{std::move(eng), std::move(strm)}).first->second;
}

memory::desc row_major(memory::dims dims, memory::data_type dt) {
  memory::dims strides(dims.size(), 1);
  for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return memory::desc(dims, dt, strides);
}

}  // namespace

void GemmSwiGLUOneDnn(tvm::ffi::TensorView x, tvm::ffi::TensorView wg, tvm::ffi::TensorView wu,
                      tvm::ffi::TensorView scale, tvm::ffi::TensorView up,
                      tvm::ffi::TensorView out) {
  TVM_FFI_ICHECK_EQ(x.ndim(), 2) << "x must be [m, k]";
  TVM_FFI_ICHECK_EQ(wg.ndim(), 2) << "gate weight must be [k, d]";
  TVM_FFI_ICHECK_EQ(wu.ndim(), 2) << "up weight must be [k, d]";

  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t D = wg.size(1);

  TVM_FFI_ICHECK_EQ(wg.size(0), K) << "gate weight rows must match x's inner dim";
  TVM_FFI_ICHECK_EQ(wu.size(0), K) << "up weight rows must match x's inner dim";
  TVM_FFI_ICHECK_EQ(wu.size(1), D) << "gate and up must have the same width";
  TVM_FFI_ICHECK_EQ(scale.size(0), M) << "scale must have one entry per row";
  TVM_FFI_ICHECK_EQ(out.size(0), M) << "out rows must match x";
  TVM_FFI_ICHECK_EQ(out.size(1), D) << "out width must match the projections";
  if (M == 0 || D == 0 || K == 0) return;

  DLDevice dev = x.device();
  sycl::queue* q = static_cast<sycl::queue*>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";
  OneDnnContext& ctx = context_for(q);

  static std::unordered_map<MatmulKey, CachedPrimitives, MatmulKeyHash> prim_cache;

  const auto bf16 = memory::data_type::bf16;
  const auto f32 = memory::data_type::f32;

  auto x_md = row_major({M, K}, bf16);
  auto wg_md = row_major({K, D}, bf16);
  auto wu_md = row_major({K, D}, bf16);
  auto out_md = row_major({M, D}, bf16);
  auto up_md = row_major({M, D}, bf16);
  auto scale_md = row_major({M, 1}, f32);  // broadcast across columns

  memory x_mem(x_md, ctx.engine, x.data_ptr());
  memory wg_mem(wg_md, ctx.engine, wg.data_ptr());
  memory wu_mem(wu_md, ctx.engine, wu.data_ptr());
  memory up_mem(up_md, ctx.engine, up.data_ptr());
  memory out_mem(out_md, ctx.engine, out.data_ptr());
  memory scale_mem(scale_md, ctx.engine, scale.data_ptr());

  MatmulKey key{M, K, D};
  auto cached = prim_cache.find(key);
  if (cached == prim_cache.end()) {
    // matmul 1: up = (x @ Wu) * r[m]
    dnnl::post_ops po_up;
    po_up.append_binary(dnnl::algorithm::binary_mul, scale_md);
    dnnl::primitive_attr attr_up;
    attr_up.set_post_ops(po_up);
    dnnl::matmul::primitive_desc pd_up(ctx.engine, x_md, wu_md, up_md, attr_up);

    // matmul 2: out = swish((x @ Wg) * r[m]) * up
    // Post-ops apply in declaration order, so this is the whole activation chain fused
    // into the epilogue of a oneDNN GEMM.
    dnnl::post_ops po_out;
    po_out.append_binary(dnnl::algorithm::binary_mul, scale_md);
    po_out.append_eltwise(dnnl::algorithm::eltwise_swish, 1.0f, 0.0f);
    po_out.append_binary(dnnl::algorithm::binary_mul, up_md);
    dnnl::primitive_attr attr_out;
    attr_out.set_post_ops(po_out);
    dnnl::matmul::primitive_desc pd_out(ctx.engine, x_md, wg_md, out_md, attr_out);

    cached = prim_cache.emplace(key, CachedPrimitives{dnnl::matmul(pd_up),
                                                      dnnl::matmul(pd_out)}).first;
  }

  cached->second.up_mm.execute(ctx.stream, {
      {DNNL_ARG_SRC, x_mem},
      {DNNL_ARG_WEIGHTS, wu_mem},
      {DNNL_ARG_DST, up_mem},
      {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, scale_mem}});

  cached->second.out_mm.execute(ctx.stream, {
      {DNNL_ARG_SRC, x_mem},
      {DNNL_ARG_WEIGHTS, wg_mem},
      {DNNL_ARG_DST, out_mem},
      {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, scale_mem},
      {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1, up_mem}});

  // oneDNN was handed the framework's queue, so work is already ordered against it. The
  // wait is what makes the result visible to the caller's next torch op.
  ctx.stream.wait();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_swiglu_onednn, GemmSwiGLUOneDnn);

}  // namespace fib_examples
