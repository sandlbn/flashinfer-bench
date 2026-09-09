"""t1: one sub-group per row, sub_group_size=16, no SLM and no work-group barrier.

The production kernel (rms_norm_multi_row_kernel<bf16,3,8,16>) pins
[[reqd_sub_group_size(32)]] while giving each row only 16 work-items, so a sub-group
straddles two rows.  That forces the row reduction to be a hand-rolled masked shift tree,
the result to go through shared local memory, and the whole 256-item work-group to hit a
barrier before phase 2.  This device reports sub_group_sizes == [16, 32]; at 16 each row is
exactly one sub-group, the reduction is a single sub-group collective, and both the SLM
round trip and the barrier disappear.  The loaded vector is also kept in registers instead
of being re-read in phase 2.
"""

import sys

import torch
import torch.nn as nn

sys.path.insert(0, "tools/kernel-harness")
from sycl_harness import build  # noqa: E402

ROWS_PER_WG = 32

SOURCE = r"""
#include <sycl/sycl.hpp>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace fib_trials {

using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int kVec = 8;
constexpr int kSg = 16;
constexpr int kRowsPerWg = ROWS_PER_WG_TOKEN;

struct alignas(16) bf16x8 {
  bf16 v[kVec];
};

void RmsNorm(tvm::ffi::TensorView x, tvm::ffi::TensorView weight, double eps,
             tvm::ffi::TensorView out) {
  const int64_t hidden = x.size(x.ndim() - 1);
  int64_t rows = 1;
  for (int i = 0; i < x.ndim() - 1; ++i) rows *= x.size(i);

  TVM_FFI_ICHECK_EQ(hidden % kVec, 0);
  TVM_FFI_ICHECK_LE(hidden / kVec, kSg) << "one sub-group must cover a row";
  TVM_FFI_ICHECK_EQ(kSg % (hidden / kVec), 0);

  DLDevice dev = x.device();
  sycl::queue* q = static_cast<sycl::queue*>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";

  const auto* xp = static_cast<const bf16x8*>(x.data_ptr());
  const auto* wp = static_cast<const bf16x8*>(weight.data_ptr());
  auto* op = static_cast<bf16x8*>(out.data_ptr());

  const int vecs_per_row = static_cast<int>(hidden / kVec);
  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  const float eps_f = static_cast<float>(eps);
  const int nrows = static_cast<int>(rows);
  const int ngroups = (nrows + kRowsPerWg - 1) / kRowsPerWg;
  const size_t local = static_cast<size_t>(kRowsPerWg) * kSg;

  q->parallel_for(
      sycl::nd_range<1>(static_cast<size_t>(ngroups) * local, local),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
        const int lid = static_cast<int>(it.get_local_id(0));
        const int row_in_wg = lid / kSg;
        const int lane = lid % kSg;
        const int row = static_cast<int>(it.get_group(0)) * kRowsPerWg + row_in_wg;
        if (row >= nrows) return;  // sub-group uniform: a sub-group is one row

        const bf16x8* in = xp + static_cast<int64_t>(row) * vecs_per_row;
        bf16x8* o = op + static_cast<int64_t>(row) * vecs_per_row;

        // A row narrower than the sub-group leaves the tail lanes idle; they must still
        // enter the collective, so give them a zero partial rather than an early exit.
        bf16x8 t;
        float acc = 0.0f;
        if (lane < vecs_per_row) {
          t = in[lane];
#pragma unroll
          for (int j = 0; j < kVec; ++j) {
            const float f = static_cast<float>(t.v[j]);
            acc += f * f;
          }
        }

        const float sum =
            sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
        const float scale = sycl::rsqrt(sum * inv_hidden + eps_f);

        if (lane < vecs_per_row) {
          const bf16x8 w = wp[lane];
          bf16x8 dst;
#pragma unroll
          for (int j = 0; j < kVec; ++j) {
            dst.v[j] = static_cast<bf16>(static_cast<float>(t.v[j]) * scale *
                                         static_cast<float>(w.v[j]));
          }
          o[lane] = dst;
        }
      });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(rms_norm_sg16, RmsNorm);

}  // namespace fib_trials
""".replace("ROWS_PER_WG_TOKEN", str(ROWS_PER_WG))


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
