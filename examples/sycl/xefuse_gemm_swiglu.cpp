// Fused GEMM + RMSNorm row-scale + SwiGLU for Intel GPUs, via Xe-Fuse.
//
// This is the payoff of epilogue fusion. GEMM dominates decoder time (~70-85%), it runs
// through oneDNN, and a hand-written matrix multiply will not beat it. What can beat the
// unfused sequence is doing the norm scaling and the gated activation *inside* the GEMM's
// epilogue, on data still in registers -- no extra launches, no round trip through memory.
//
//     D = SwiGLU(acc * R[m])
//
// The kernel body is Xe-Fuse's; generated with:
//     python autotune/generate_kernel.py --preset k2 --m M --n N --k K
//
// OPERAND LAYOUT: A is [M, K] and B is [K, N], both row-major. The CUTLASS stride is
// built with make_shape(N, K, L), which reads as though B were [N, K] -- it is not.
// Established by comparing the kernel's output against candidate references rather than
// by reading the stride code.
// What is written here is the binding: its standalone `main` replaced by a TVM-FFI entry
// point that takes the Definition's tensors.
//
// Build: declare "xe-fuse" in the solution's dependencies and set FIB_XE_FUSE_DIR. The
// builder supplies the CUTLASS-SYCL include paths and the two defines that select the
// SYCL backend; without those CUTLASS assumes CUDA and the compile fails looking for
// cuda_runtime_api.h.
//
// NOTE ON LAYOUT: Xe-Fuse's SwiGLU takes *interleaved* (gate, up) pairs -- gate on even
// lanes, up on odd -- and both lanes of a pair receive the same result. That is a
// different layout from vLLM's silu_and_mul, which splits the tensor in half. The
// Definition's reference must match this one, or the correctness gate will (correctly)
// reject the kernel.

#include "xe-fuse/builder/epilogue_builder.hpp"

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include <cstdint>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

using namespace cute;
namespace b = xe_fuse::builder;
using bf16 = cutlass::bfloat16_t;

namespace fib_examples {

using TileShape = Shape<_256, _256, _32>;

// The Epilogue Visitor Tree: scale each row by R[m], then SwiGLU over adjacent pairs.
// This is the composable part -- swapping SwiGLU for GeGLU, or adding a residual, is a
// change to this one line.
using EVT = b::SwiGLU<b::ScaleRows<b::Acc, TileShape, float>>;
using KernelConfig = b::MakeGemm<EVT, bf16, bf16, bf16, float, float, TileShape>;
using GemmOp = typename KernelConfig::Gemm;

void GemmSwiGLUXeFuse(tvm::ffi::TensorView a, tvm::ffi::TensorView bt,
                      tvm::ffi::TensorView scale, tvm::ffi::TensorView out) {
  TVM_FFI_ICHECK_EQ(a.ndim(), 2) << "a must be [m, k]";
  TVM_FFI_ICHECK_EQ(bt.ndim(), 2) << "b must be [k, n]";
  TVM_FFI_ICHECK_EQ(scale.ndim(), 1) << "scale must be [m]";
  TVM_FFI_ICHECK_EQ(out.ndim(), 2) << "out must be [m, n]";

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  // B is [K, N] row-major -- verified empirically against the kernel's output, not
  // assumed from the CUTLASS stride construction, which reads as if it were [N, K].
  const int N = static_cast<int>(bt.size(1));
  const int L = 1;

  TVM_FFI_ICHECK_EQ(bt.size(0), K) << "b leading dimension must match a's inner";
  TVM_FFI_ICHECK_EQ(scale.size(0), M) << "scale must have one entry per row";
  TVM_FFI_ICHECK_EQ(out.size(0), M) << "out rows must match a";
  TVM_FFI_ICHECK_EQ(out.size(1), N) << "out cols must match b";
  if (M == 0 || N == 0 || K == 0) return;

  using StrideA = typename GemmOp::GemmKernel::StrideA;
  using StrideB = typename GemmOp::GemmKernel::StrideB;
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(KernelConfig::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(KernelConfig::StrideD{}, make_shape(M, N, L));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  // EVT arguments, inner to outer: accumulator -> per-row scale -> multiply -> SwiGLU.
  typename b::Acc::Arguments accum_args{};
  typename b::ColBroadcast<0, TileShape, float>::Arguments scale_args;
  scale_args.ptr_col = static_cast<const float*>(scale.data_ptr());
  scale_args.null_default = float(1);
  scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};
  typename b::MulOp<float, float>::Arguments mul_args{};
  typename b::ScaleRows<b::Acc, TileShape, float>::Arguments rms_args{
      accum_args, scale_args, mul_args};
  typename xe_fuse::XePairwiseCompute<xe_fuse::SwiGLUFn>::Arguments swiglu_args{};
  typename EVT::Arguments evt_args{rms_args, swiglu_args};

  typename GemmOp::GemmKernel::EpilogueArguments epilogue_args{
      evt_args, nullptr, stride_C, static_cast<bf16*>(out.data_ptr()), stride_D};
  typename GemmOp::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, L},
      {static_cast<const bf16*>(a.data_ptr()), stride_A,
       static_cast<const bf16*>(bt.data_ptr()), stride_B},
      epilogue_args,
      hw_info};

  GemmOp gemm_op;
  const size_t workspace_size = GemmOp::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  TVM_FFI_ICHECK(gemm_op.can_implement(arguments) == cutlass::Status::kSuccess)
      << "xe-fuse GEMM cannot implement this problem shape";
  TVM_FFI_ICHECK(gemm_op.initialize(arguments, workspace.get()) == cutlass::Status::kSuccess)
      << "xe-fuse GEMM initialize failed";
  TVM_FFI_ICHECK(gemm_op.run() == cutlass::Status::kSuccess) << "xe-fuse GEMM run failed";

  // CUTLASS-SYCL drives its own queue rather than the framework's, so the completion
  // barrier has to be explicit here. The benchmark harness synchronises around
  // measurements anyway, so this costs ordering, not throughput.
  compat::wait();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_swiglu_xefuse, GemmSwiGLUXeFuse);

}  // namespace fib_examples
