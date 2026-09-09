"""rows-per-work-group variant: ROWS sub-groups per work-group, still one row each, row held in registers, no SLM and no barrier.

The production kernel (fused_add_rms_norm_kernel<bf16,8,true>) gives each row a 128-item
work-group -- four sub-groups at the pinned width of 32 -- so `reduce_over_group` over the
whole group has to go through shared local memory, and there is a second `group_barrier`
before the normalize pass.  It also writes the summed residual out and then reads all of it
back in phase 2, a full extra round trip over hidden_size.

Here one row is one sub-group.  Each lane keeps its slice of the summed residual in
registers across the reduction, so phase 2 re-reads nothing; the reduction is a single
sub-group collective; there is no SLM and no barrier at all.
"""

import sys

import torch
import torch.nn as nn

sys.path.insert(0, "tools/kernel-harness")
from sycl_harness import build  # noqa: E402

SG = 32
ROWS_PER_WG = 2

SOURCE = r"""
#include <sycl/sycl.hpp>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace fib_trials {

using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int kVec = 8;
constexpr int kSg = SG_TOKEN;
constexpr int kRowsPerWg = ROWS_WG_TOKEN;
constexpr int kMaxVecsPerLane = 16;

struct alignas(16) bf16x8 {
  bf16 v[kVec];
};

void FusedAddRmsNorm(tvm::ffi::TensorView input, tvm::ffi::TensorView residual,
                     tvm::ffi::TensorView weight, double eps) {
  TVM_FFI_ICHECK(input.IsContiguous()) << "input must be contiguous";
  TVM_FFI_ICHECK(residual.IsContiguous()) << "residual must be contiguous";

  const int64_t hidden = input.size(input.ndim() - 1);
  const int64_t rows = input.numel() / hidden;

  TVM_FFI_ICHECK_EQ(hidden % (kVec * kSg), 0) << "hidden must tile the sub-group";
  const int vecs_per_lane = static_cast<int>(hidden / (kVec * kSg));
  TVM_FFI_ICHECK_LE(vecs_per_lane, kMaxVecsPerLane);

  DLDevice dev = input.device();
  sycl::queue* q = static_cast<sycl::queue*>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";

  auto* xp = static_cast<bf16x8*>(input.data_ptr());
  auto* rp = static_cast<bf16x8*>(residual.data_ptr());
  const auto* wp = static_cast<const bf16x8*>(weight.data_ptr());

  const int vecs_per_row = static_cast<int>(hidden / kVec);
  const int64_t nrows = rows;
  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  const float eps_f = static_cast<float>(eps);

  q->parallel_for(
      sycl::nd_range<1>(
          static_cast<size_t>((rows + kRowsPerWg - 1) / kRowsPerWg) * kRowsPerWg * kSg,
          static_cast<size_t>(kRowsPerWg) * kSg),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
        const int lid = static_cast<int>(it.get_local_id(0));
        const int lane = lid % kSg;
        const int64_t row = static_cast<int64_t>(it.get_group(0)) * kRowsPerWg + lid / kSg;
        if (row >= nrows) return;  // sub-group uniform
        const int64_t base = row * vecs_per_row;

        bf16x8 keep[kMaxVecsPerLane];
        float acc = 0.0f;
        for (int k = 0; k < vecs_per_lane; ++k) {
          const int idx = lane + k * kSg;  // coalesced across the sub-group
          bf16x8 a = xp[base + idx];
          const bf16x8 b = rp[base + idx];
#pragma unroll
          for (int j = 0; j < kVec; ++j) {
            const float f = static_cast<float>(a.v[j]) + static_cast<float>(b.v[j]);
            a.v[j] = static_cast<bf16>(f);
            acc += f * f;
          }
          rp[base + idx] = a;
          keep[k] = a;
        }

        const float sum =
            sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
        const float scale = sycl::rsqrt(sum * inv_hidden + eps_f);

        for (int k = 0; k < vecs_per_lane; ++k) {
          const int idx = lane + k * kSg;
          const bf16x8 w = wp[idx];
          bf16x8 dst;
#pragma unroll
          for (int j = 0; j < kVec; ++j) {
            dst.v[j] = static_cast<bf16>(static_cast<float>(keep[k].v[j]) * scale *
                                         static_cast<float>(w.v[j]));
          }
          xp[base + idx] = dst;
        }
      });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_add_rms_norm_sg, FusedAddRmsNorm);

}  // namespace fib_trials
""".replace("SG_TOKEN", str(SG)).replace("ROWS_WG_TOKEN", str(ROWS_PER_WG))


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
