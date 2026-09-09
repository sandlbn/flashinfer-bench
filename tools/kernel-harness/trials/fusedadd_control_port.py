"""Control: a faithful port of vLLM's fused_add_rms_norm kernel on the trial's call path.

Puts back everything the candidate removes -- a 128-item work-group per row (four
sub-groups at width 32), `reduce_over_group` over the whole work-group, the scale parked in
SLM, the barrier, and the phase-2 re-read of the residual it just wrote.  Isolates how much
of the candidate's ratio is the kernel and how much is TVM-FFI vs torch.ops on the host.
"""

import sys

import torch
import torch.nn as nn

sys.path.insert(0, "tools/kernel-harness")
from sycl_harness import build  # noqa: E402

SOURCE = r"""
#include <sycl/sycl.hpp>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace fib_trials {

using bf16 = sycl::ext::oneapi::bfloat16;
constexpr int kVec = 8;

struct alignas(16) bf16x8 {
  bf16 v[kVec];
};

void FusedAddPort(tvm::ffi::TensorView input, tvm::ffi::TensorView residual,
                  tvm::ffi::TensorView weight, double eps) {
  const int64_t hidden = input.size(input.ndim() - 1);
  const int64_t rows = input.numel() / hidden;

  DLDevice dev = input.device();
  sycl::queue* q = static_cast<sycl::queue*>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";

  auto* xp = static_cast<bf16x8*>(input.data_ptr());
  auto* rp = static_cast<bf16x8*>(residual.data_ptr());
  const auto* wp = static_cast<const bf16x8*>(weight.data_ptr());

  const int vec_hidden = static_cast<int>(hidden / kVec);
  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  const float eps_f = static_cast<float>(eps);
  const size_t local = static_cast<size_t>(vec_hidden < 1024 ? vec_hidden : 1024);

  q->submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(rows) * local, local),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          float* s = s_variance.get_multi_ptr<sycl::access::decorated::no>().get();
          const int64_t base = static_cast<int64_t>(it.get_group(0)) * vec_hidden;
          float variance = 0.0f;
          for (int idx = static_cast<int>(it.get_local_id(0)); idx < vec_hidden;
               idx += static_cast<int>(it.get_local_range(0))) {
            bf16x8 a = xp[base + idx];
            const bf16x8 b = rp[base + idx];
#pragma unroll
            for (int j = 0; j < kVec; ++j) {
              a.v[j] = static_cast<bf16>(static_cast<float>(a.v[j]) +
                                         static_cast<float>(b.v[j]));
              const float f = static_cast<float>(a.v[j]);
              variance += f * f;
            }
            rp[base + idx] = a;
          }
          variance = sycl::reduce_over_group(it.get_group(), variance, sycl::plus<float>());
          if (it.get_local_id(0) == 0) *s = sycl::rsqrt(variance * inv_hidden + eps_f);
          sycl::group_barrier(it.get_group());
          const float scale = *s;
          for (int idx = static_cast<int>(it.get_local_id(0)); idx < vec_hidden;
               idx += static_cast<int>(it.get_local_range(0))) {
            const bf16x8 res = rp[base + idx];  // re-read, as the production kernel does
            const bf16x8 w = wp[idx];
            bf16x8 dst;
#pragma unroll
            for (int j = 0; j < kVec; ++j) {
              dst.v[j] = static_cast<bf16>(static_cast<float>(res.v[j]) * scale *
                                           static_cast<float>(w.v[j]));
            }
            xp[base + idx] = dst;
          }
        });
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_add_port, FusedAddPort);

}  // namespace fib_trials
"""


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.run = build("fused_add_rmsnorm_residual_h1024", SOURCE)

    def forward(self, t0, t1, t2):
        self.run(t0, t1, t2, 1e-06)
        return t0


def get_inputs():
    device = "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    return [
        torch.randn([4, 1024], dtype=torch.bfloat16, device=device),
        torch.randn([4, 1024], dtype=torch.bfloat16, device=device),
        torch.randn([1024], dtype=torch.bfloat16, device=device),
    ]


def get_init_inputs():
    return []
