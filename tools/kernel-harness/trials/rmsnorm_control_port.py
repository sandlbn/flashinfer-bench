"""Control: a faithful port of vLLM's own multi_row kernel, run through the trial's own
build and call path.

Everything the candidates change is put back: sub_group_size 32 with a row spanning only
half a sub-group, the masked shift-tree reduction, the result parked in shared local
memory, the whole-work-group barrier, and the phase-2 re-read of the input.  Only the host
call path differs from the baseline (TVM-FFI instead of torch.ops).

If this control measures like the production baseline, the host paths cost the same and the
candidates' ratio is a property of the kernel.  If it does not, the ratio is partly a
property of the harness and has to be discounted by the difference.
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
constexpr int kRowsPerWg = 16;

struct alignas(16) bf16x8 {
  bf16 v[kVec];
};

void RmsNormPort(tvm::ffi::TensorView x, tvm::ffi::TensorView weight, double eps,
                 tvm::ffi::TensorView out) {
  const int64_t hidden = x.size(x.ndim() - 1);
  const int64_t rows = x.numel() / hidden;

  DLDevice dev = x.device();
  sycl::queue* q = static_cast<sycl::queue*>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";

  const auto* xp = static_cast<const bf16x8*>(x.data_ptr());
  const auto* wp = static_cast<const bf16x8*>(weight.data_ptr());
  auto* op = static_cast<bf16x8*>(out.data_ptr());

  const int items_per_row = static_cast<int>(hidden / kVec);
  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  const float eps_f = static_cast<float>(eps);
  const int nrows = static_cast<int>(rows);
  const int ngroups = (nrows + kRowsPerWg - 1) / kRowsPerWg;
  const size_t local = static_cast<size_t>(kRowsPerWg) * items_per_row;

  q->submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> s_variance(sycl::range<1>(kRowsPerWg), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(ngroups) * local, local),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          float* s = s_variance.get_multi_ptr<sycl::access::decorated::no>().get();
          const int lid = static_cast<int>(it.get_local_id(0));
          const int row_in_wg = lid / items_per_row;
          const int col = lid % items_per_row;
          const int row = static_cast<int>(it.get_group(0)) * kRowsPerWg + row_in_wg;

          const bf16x8* in = xp + static_cast<int64_t>(row) * items_per_row;
          float variance = 0.0f;
          {
            const bf16x8 t = in[col];
#pragma unroll
            for (int j = 0; j < kVec; ++j) {
              const float f = static_cast<float>(t.v[j]);
              variance += f * f;
            }
          }

          auto sg = it.get_sub_group();
          const int lane = static_cast<int>(sg.get_local_linear_id());
          const int row_lane_offset = lane % items_per_row;
          for (int offset = items_per_row / 2; offset > 0; offset >>= 1) {
            const float other = sycl::shift_group_left(sg, variance, offset);
            if (row_lane_offset < offset) variance += other;
          }
          if (row_lane_offset == 0) {
            s[row_in_wg] = sycl::rsqrt(variance * inv_hidden + eps_f);
          }
          sycl::group_barrier(it.get_group());

          const float scale = s[row_in_wg];
          bf16x8* o = op + static_cast<int64_t>(row) * items_per_row;
          const bf16x8 t = in[col];  // re-read, as the production kernel does
          const bf16x8 w = wp[col];
          bf16x8 dst;
#pragma unroll
          for (int j = 0; j < kVec; ++j) {
            dst.v[j] = static_cast<bf16>(static_cast<float>(t.v[j]) * scale *
                                         static_cast<float>(w.v[j]));
          }
          o[col] = dst;
        });
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(rms_norm_port, RmsNormPort);

}  // namespace fib_trials
"""


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.run = build("rmsnorm_h128", SOURCE)

    def forward(self, t0, t1, t2):
        self.run(t1, t2, 1e-06, t0)
        return t0


def get_inputs():
    device = "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    return [
        torch.empty([4, 16, 128], dtype=torch.bfloat16, device=device),
        torch.randn([4, 16, 128], dtype=torch.bfloat16, device=device),
        torch.randn([128], dtype=torch.bfloat16, device=device),
    ]


def get_init_inputs():
    return []
