// Fused residual-add + RMSNorm in SYCL for Intel GPUs.
//
// This is the shape of kernel that matters on Intel: both sgl-kernel-xpu and
// vllm-xpu-kernels ship a `fused_add_rms_norm`, because every transformer block runs one
// per sublayer. Fusing the residual add into the norm halves the memory traffic --
// the residual is read once and written once, instead of round-tripping through a
// separate elementwise kernel.
//
//   residual_out = x + residual
//   out          = rmsnorm(residual_out) * weight
//
// One work-group per row; a single group reduction for the sum of squares.

#include <sycl/sycl.hpp>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace fib_examples {

namespace {
constexpr size_t kWorkGroupSize = 256;
}  // namespace

void FusedAddRMSNormSycl(tvm::ffi::TensorView x, tvm::ffi::TensorView residual,
                         tvm::ffi::TensorView weight, double eps,
                         tvm::ffi::TensorView out, tvm::ffi::TensorView residual_out) {
  TVM_FFI_ICHECK_EQ(x.ndim(), 2) << "x must be [tokens, hidden]";
  TVM_FFI_ICHECK_EQ(residual.ndim(), 2) << "residual must be [tokens, hidden]";
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1) << "weight must be [hidden]";
  TVM_FFI_ICHECK_EQ(weight.size(0), x.size(1)) << "weight/hidden mismatch";

  const int64_t tokens = x.size(0);
  const int64_t hidden = x.size(1);
  if (tokens == 0 || hidden == 0) return;

  DLDevice dev = x.device();
  sycl::queue* q = static_cast<sycl::queue*>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";

  const float* x_data = static_cast<const float*>(x.data_ptr());
  const float* res_data = static_cast<const float*>(residual.data_ptr());
  const float* w_data = static_cast<const float*>(weight.data_ptr());
  float* out_data = static_cast<float*>(out.data_ptr());
  float* res_out_data = static_cast<float*>(residual_out.data_ptr());
  const float eps_f = static_cast<float>(eps);
  const float inv_hidden = 1.0f / static_cast<float>(hidden);

  q->parallel_for(
      sycl::nd_range<1>(static_cast<size_t>(tokens) * kWorkGroupSize, kWorkGroupSize),
      [=](sycl::nd_item<1> item) {
        const int64_t row = static_cast<int64_t>(item.get_group(0));
        const size_t lid = item.get_local_id(0);
        const size_t lsize = item.get_local_range(0);

        const float* x_row = x_data + row * hidden;
        const float* res_row = res_data + row * hidden;
        float* out_row = out_data + row * hidden;
        float* res_out_row = res_out_data + row * hidden;

        // Fused add: compute the new residual once, keep it for the rescale pass.
        float partial = 0.0f;
        for (int64_t i = static_cast<int64_t>(lid); i < hidden;
             i += static_cast<int64_t>(lsize)) {
          const float v = x_row[i] + res_row[i];
          res_out_row[i] = v;
          partial += v * v;
        }

        const float sum_sq =
            sycl::reduce_over_group(item.get_group(), partial, sycl::plus<float>());
        const float scale = sycl::rsqrt(sum_sq * inv_hidden + eps_f);

        for (int64_t i = static_cast<int64_t>(lid); i < hidden;
             i += static_cast<int64_t>(lsize)) {
          out_row[i] = res_out_row[i] * scale * w_data[i];
        }
      });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_add_rmsnorm_sycl, FusedAddRMSNormSycl);

}  // namespace fib_examples
