// RMSNorm in SYCL for Intel GPUs.
//
// One work-group per row. Each work-group reduces the row's sum of squares with a group
// collective, then rescales. Demonstrates the parts of SYCL that matter for real kernels:
// nd_range, group reductions, and running on the framework's own queue.
//
// Dispatches on dtype: real models run fp16 or bf16, and a kernel that only handles fp32
// is not usable for serving. Accumulation is always fp32 regardless of storage type --
// summing squares of half-precision values in half precision loses the norm.

#include <sycl/sycl.hpp>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace fib_examples {

namespace {

constexpr size_t kWorkGroupSize = 256;

template <typename T>
void LaunchRMSNorm(sycl::queue* q, const T* x_data, const T* w_data, T* out_data,
                   int64_t batch, int64_t hidden, float eps) {
  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  q->parallel_for(
      sycl::nd_range<1>(static_cast<size_t>(batch) * kWorkGroupSize, kWorkGroupSize),
      [=](sycl::nd_item<1> item) {
        const int64_t row = static_cast<int64_t>(item.get_group(0));
        const size_t lid = item.get_local_id(0);
        const size_t lsize = item.get_local_range(0);

        const T* row_in = x_data + row * hidden;
        T* row_out = out_data + row * hidden;

        float partial = 0.0f;
        for (int64_t i = static_cast<int64_t>(lid); i < hidden;
             i += static_cast<int64_t>(lsize)) {
          const float v = static_cast<float>(row_in[i]);
          partial += v * v;
        }

        const float sum_sq =
            sycl::reduce_over_group(item.get_group(), partial, sycl::plus<float>());
        const float scale = sycl::rsqrt(sum_sq * inv_hidden + eps);

        for (int64_t i = static_cast<int64_t>(lid); i < hidden;
             i += static_cast<int64_t>(lsize)) {
          const float v = static_cast<float>(row_in[i]) * scale * static_cast<float>(w_data[i]);
          row_out[i] = static_cast<T>(v);
        }
      });
}

}  // namespace

// Argument order follows the Definition: inputs in declaration order, then outputs.
void RMSNormSycl(tvm::ffi::TensorView x, tvm::ffi::TensorView weight, double eps,
                 tvm::ffi::TensorView out) {
  TVM_FFI_ICHECK_EQ(x.ndim(), 2) << "x must be [batch, hidden]";
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1) << "weight must be [hidden]";
  TVM_FFI_ICHECK_EQ(weight.size(0), x.size(1)) << "weight/hidden mismatch";
  TVM_FFI_ICHECK_EQ(out.size(0), x.size(0)) << "out/batch mismatch";
  TVM_FFI_ICHECK_EQ(out.size(1), x.size(1)) << "out/hidden mismatch";

  const int64_t batch = x.size(0);
  const int64_t hidden = x.size(1);
  if (batch == 0 || hidden == 0) return;

  // PyTorch's own queue: already on the right device, and in the context that owns these
  // pointers. Creating a queue here instead would be undefined behaviour.
  DLDevice dev = x.device();
  sycl::queue* q = static_cast<sycl::queue*>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";

  const DLDataType dtype = x.dtype();
  const float eps_f = static_cast<float>(eps);

  if (dtype.code == kDLFloat && dtype.bits == 32) {
    LaunchRMSNorm<float>(q, static_cast<const float*>(x.data_ptr()),
                         static_cast<const float*>(weight.data_ptr()),
                         static_cast<float*>(out.data_ptr()), batch, hidden, eps_f);
  } else if (dtype.code == kDLFloat && dtype.bits == 16) {
    LaunchRMSNorm<sycl::half>(q, static_cast<const sycl::half*>(x.data_ptr()),
                              static_cast<const sycl::half*>(weight.data_ptr()),
                              static_cast<sycl::half*>(out.data_ptr()), batch, hidden, eps_f);
  } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
    using bf16 = sycl::ext::oneapi::bfloat16;
    LaunchRMSNorm<bf16>(q, static_cast<const bf16*>(x.data_ptr()),
                        static_cast<const bf16*>(weight.data_ptr()),
                        static_cast<bf16*>(out.data_ptr()), batch, hidden, eps_f);
  } else {
    TVM_FFI_LOG_AND_THROW(RuntimeError)
        << "rmsnorm_sycl: unsupported dtype (code=" << int(dtype.code)
        << " bits=" << int(dtype.bits) << ")";
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm_sycl, RMSNormSycl);

}  // namespace fib_examples
