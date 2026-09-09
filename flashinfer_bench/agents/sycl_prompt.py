"""Prompt for agents writing SYCL kernels for Intel GPUs.

Mirrors :mod:`flashinfer_bench.agents.ffi_prompt`, which covers CUDA. The two share a
binding layer (TVM-FFI over DLPack) and differ only in the kernel language and how the
framework's stream is obtained, so an agent that knows one needs little to write the
other.
"""

SYCL_PROMPT_SIMPLE = """
# Writing SYCL kernels for Intel GPUs with TVM-FFI

You are writing a SYCL kernel that FlashInfer-Bench will compile with oneAPI DPC++
(`icpx -fsycl`) and load through TVM-FFI. The kernel runs on Intel GPUs.

## The one thing to get right: use the framework's queue

Do NOT create your own `sycl::queue`. SYCL pointers are bound to a `sycl::context`, and
the tensors you receive belong to PyTorch's context. A queue you create yourself will
generally have a different context, and using framework pointers with it is undefined
behaviour.

Ask the environment for the queue instead. This is the exact counterpart of the CUDA path
(`cudaStream_t stream = TVMFFIEnvGetStream(...)`):

```cpp
DLDevice dev = x.device();
sycl::queue* q = static_cast<sycl::queue*>(
    TVMFFIEnvGetStream(dev.device_type, dev.device_id));
```

That pointer is PyTorch's own `sycl::queue`, already on the right device and context.

## Do not synchronize

Submit work and return. Do not call `q->wait()`, and do not use
`sycl::buffer`/`sycl::accessor` (which synchronize on destruction). The benchmark harness
synchronizes around measurements; a wait inside the kernel serializes execution and makes
the measurement worse than it should be.

## Complete example

```cpp
// File: add_one.cpp   (SYCL is C++; use a .cpp extension)
#include <sycl/sycl.hpp>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace my_kernels {

void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  TVM_FFI_ICHECK_EQ(x.ndim(), 1) << "x must be 1D";
  TVM_FFI_ICHECK_EQ(x.size(0), y.size(0)) << "shape mismatch";

  const int64_t n = x.size(0);

  DLDevice dev = x.device();
  sycl::queue* q = static_cast<sycl::queue*>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";

  const float* x_data = static_cast<const float*>(x.data_ptr());
  float* y_data = static_cast<float*>(y.data_ptr());

  q->parallel_for(sycl::range<1>(static_cast<size_t>(n)),
                  [=](sycl::id<1> i) { y_data[i] = x_data[i] + 1.0f; });
}

// Export under the name the Solution's entry_point refers to.
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_sycl, AddOne);

}  // namespace my_kernels
```

The matching Solution spec:

```json
{
  "spec": {
    "language": "sycl",
    "target_hardware": ["xpu"],
    "entry_point": "add_one.cpp::add_one_sycl",
    "destination_passing_style": true
  }
}
```
"""

SYCL_PROMPT = SYCL_PROMPT_SIMPLE + """
## Writing fast SYCL for Intel GPUs

### Work-group and sub-group sizing

Intel GPUs execute in sub-groups (the warp equivalent). Supported widths are reported per
device; 16 and 32 are the common ones. Pin the sub-group size when your kernel depends on
it, rather than letting the compiler choose:

```cpp
q->parallel_for(
    sycl::nd_range<1>(global_size, local_size),
    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
      // ...
    });
```

Use `nd_range` (not plain `range`) whenever you need work-group cooperation, shared local
memory, or barriers. Choose `local_size` as a multiple of the sub-group size, and keep it
within the device's `max_work_group_size`.

### Sub-group collectives instead of manual reduction

Sub-group primitives are considerably faster than shared-memory tree reductions and much
less error-prone:

```cpp
auto sg = item.get_sub_group();
float total = sycl::reduce_over_group(sg, value, sycl::plus<float>());
float shifted = sycl::shift_group_left(sg, value, 1);
float bcast = sycl::group_broadcast(sg, value, 0);
```

`sycl::reduce_over_group` also works over the whole work-group with `item.get_group()`.

### Shared local memory

SLM is the counterpart of CUDA shared memory. Allocate it with a local accessor:

```cpp
q->submit([&](sycl::handler& h) {
  sycl::local_accessor<float, 1> tile(sycl::range<1>(tile_size), h);
  h.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> item) {
    tile[item.get_local_id(0)] = in[item.get_global_id(0)];
    sycl::group_barrier(item.get_group());
    // ...
  });
});
```

The per-device SLM budget is reported as `local_mem_size` (commonly 64-128 KiB). Exceeding
it fails at launch.

### Vectorized access

Memory-bound kernels usually want wider accesses. `sycl::vec` loads move 4 floats at a
time:

```cpp
sycl::vec<float, 4> v;
v.load(0, sycl::global_ptr<const float>(x_data + i * 4));
```

Prefer contiguous, coalesced access across the sub-group; strided access costs the same
here as anywhere else.

### Matrix units

Recent Intel GPUs have XMX/DPAS matrix engines, exposed through
`sycl::ext::oneapi::experimental::matrix`. A device reports whether it has them
(`has_subgroup_matrix_multiply_accumulate`). Use them for GEMM-shaped work; do not assume
they exist without checking, and provide a fallback path.

### Prefer a tuned library where one exists

For standard GEMM, convolution and similar primitives, oneMKL and oneDNN are usually
faster than hand-written kernels and are already tuned per architecture. Declare them in
the Solution's `dependencies` (`"onemkl"`, `"onednn"`) and call them on the same queue.
Hand-write a kernel when you are fusing operations or doing something the libraries do not
cover.

## Know what you are competing with, and what winning would take

A kernel is not judged against the Definition's `reference`. The reference is plain PyTorch
and exists to decide correctness; it routinely does several passes over memory where the
serving stack does one, so a ratio against it can overstate the win by the whole factor that
matters. Kernels recorded several-fold faster than a reference have measured at parity or worse
against the kernel they would actually replace. Compare against the provider baseline
(`vllm-xpu`, `sgl-kernel-xpu`, oneDNN) -- that is what runs if yours is declined.

**Substitution is not free, so a faster kernel is not automatically a win.** Resolving the
definition, building a lookup key, checking dtypes and invoking the built kernel cost real
time per call and do not shrink with the kernel. That cost is a property of the part, so
read it rather than assuming it: `flashinfer_bench.device.calibration.get().dispatch_us`
(`scripts/calibrate_part.py` prints it). A kernel whose total runtime is comparable to that
cost cannot pay for its own replacement however large its ratio -- what it saves is
`provider_us - ours_us`, and that has to exceed `dispatch_us`. Before optimizing, ask what
the kernel costs in absolute terms; below a few multiples of the dispatch cost, the win has
to come from removing the call (fusion) rather than from making it faster.

**Compare against what the access pattern allows, not against peak bandwidth.** Read the
same bytes in the same shape the kernel is obliged to touch, and use that as the ceiling.
A paged-attention kernel that sustains a fraction of the part's contiguous bandwidth
(`calibration.get().bandwidth_gbs`) reads as a large rewrite opportunity -- until a bare
read of one KV-head slice of its `[pages, page_size, kv_heads, head_dim]` cache sustains
the same fraction. Then the kernel is at its ceiling and the gap is in the data layout,
which no rewrite of the kernel recovers. Measure the ceiling first, or you will spend
rounds on a kernel that is already done.

**Fusion is where the wins are, because it removes work rather than accelerating it.** You
will not beat oneDNN at a plain matmul -- it is tuned per architecture and is already the
path `F.linear` takes. What oneDNN cannot express is an epilogue that spans two accumulator
lanes, so fusing an activation or a norm into the GEMM's epilogue removes a kernel launch
and a full round trip through memory. Reach for oneDNN post-ops before writing a matmul.

## Measuring your own kernel

Getting this wrong will send you optimizing the wrong thing, and it is easy to get wrong.

- **Interleave the alternatives; never time them in sequence.** A GPU that has been idle
  ramps its clocks, so whichever case runs first absorbs the ramp. This has inverted a
  comparison outright: a kernel read as slower than its competitor when timed first and as
  faster when the rounds were interleaved and medians taken. Warm every case, then
  alternate them.
- **Put many calls inside one timed region.** Wrapping a single call in an event pair
  charges the event and launch overhead to the measurement -- the part's
  `calibration.get().timing_floor_us`, which is most of the number for a short kernel and
  puts every decode-sized kernel on the same floor. Batch until the region is milliseconds,
  then divide.
- **Never block the host inside the kernel.** A `wait()` after every submit turned a oneDNN
  kernel that beat the reference into one that lost to it, and the structure was blamed
  before the block was found. The framework's queue already orders work and the caller
  synchronizes when it needs the result.

## Correctness requirements

- Validate shapes and dtypes with `TVM_FFI_ICHECK*` before touching data.
- Guard against out-of-range indices; the last work-group is usually partial.
- Match the dtypes in the Definition exactly. `sycl::half` is `float16`;
  `sycl::ext::oneapi::bfloat16` is `bfloat16`.
- Accumulate in `float` even when inputs and outputs are half precision, unless the
  Definition says otherwise.
- **Assume your output buffers may alias your inputs.** Destination-passing callers pass
  `output is hidden_states` and `residual_out is residual` deliberately: the serving
  adapter does this because allocating a fresh pair per call cost more than the kernel
  saved. Aliasing is safe only if each work-item reads every index it needs before writing
  any index another work-item might still read -- in practice, read index i of all inputs,
  then write index i of all outputs, with the group reduction as the barrier between the
  passes. If your kernel cannot honour that, say so in the Solution description; nothing in
  the interface enforces it.

## Portability

Leaving the AOT target unset compiles to SPIR-V, which is JIT-compiled at load time and
runs on any supported Intel GPU, including parts released after the kernel was written.
Write portable code by default: query device properties rather than hard-coding a
sub-group size or SLM budget for one part.

## Common mistakes

1. Creating a local `sycl::queue` instead of using `TVMFFIEnvGetStream` -- wrong context,
   undefined behaviour with framework pointers.
2. Calling `q->wait()` inside the kernel -- serializes and distorts measurement.
3. Using `sycl::buffer`/`sycl::accessor` with framework pointers -- these take ownership
   and synchronize; use USM pointers directly.
4. Assuming a sub-group size of 32 -- query it or pin it explicitly.
5. Capturing a host pointer in the kernel lambda -- everything a device lambda captures
   must be device-accessible or trivially copyable by value.
6. Forgetting `TVM_FFI_DLL_EXPORT_TYPED_FUNC` -- the module will load but the entry point
   will not be found.
"""
