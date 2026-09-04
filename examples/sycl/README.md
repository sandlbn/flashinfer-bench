# SYCL kernels for Intel GPUs

A complete, runnable example: a SYCL RMSNorm kernel, the Definition it implements, and
workloads to benchmark it against the PyTorch reference.

## Why SYCL

SYCL is where the fastest Intel GPU kernels are written. FlashInfer-Bench treats it as a
first-class solution language alongside CUDA and Triton — same Definitions, same
correctness gates, same traces.

## Run it

```bash
# oneAPI DPC++ must be available. If it is installed but not on PATH:
source /opt/intel/oneapi/setvars.sh

pip install torch --index-url https://download.pytorch.org/whl/xpu
pip install -e .

flashinfer-bench run --local examples/sycl/Example-Intel-Trace
```

Measured on an integrated Intel GPU, RMSNorm at hidden=4096:

```
rmsnorm_sycl_example  batch=512   PASSED with 3.32x speedup vs. reference
rmsnorm_sycl_example  batch=4096  PASSED with 3.25x speedup vs. reference
```

The speedup comes from fusion: the PyTorch reference makes several passes over memory,
while the kernel reduces and rescales each row in one pass.

## The one rule that matters

Do not create your own `sycl::queue`. SYCL pointers are bound to a `sycl::context`, and
the tensors you receive belong to PyTorch's context — a queue you create will generally
have a different one, and using framework pointers with it is undefined behaviour.

Ask the environment for the queue instead. This is the exact counterpart of the CUDA path:

```cpp
// CUDA
cudaStream_t stream = static_cast<cudaStream_t>(
    TVMFFIEnvGetStream(dev.device_type, dev.device_id));

// SYCL
sycl::queue* q = static_cast<sycl::queue*>(
    TVMFFIEnvGetStream(dev.device_type, dev.device_id));
```

You get PyTorch's own queue, on the right device, in the right context.

Then submit and return — do not call `q->wait()`. The harness synchronizes around
measurements; waiting inside the kernel only serializes it.

## Anatomy of the kernel

See [`rmsnorm_sycl.cpp`](rmsnorm_sycl.cpp). The parts worth copying:

- **`nd_range`, one work-group per row.** Needed for group cooperation; plain `range` has
  no work-groups to reduce over.
- **`sycl::reduce_over_group`** for the sum of squares, rather than a hand-written
  shared-local-memory tree. Faster and much harder to get wrong.
- **Accumulation in `float`** regardless of input dtype.
- **Grid-stride loops** so one work-group handles any hidden size.

## Writing your own

Declare the language in the Solution's build spec:

```json
{
  "spec": {
    "language": "sycl",
    "target_hardware": ["xpu"],
    "entry_point": "my_kernel.cpp::my_kernel",
    "destination_passing_style": true,
    "dependencies": ["onemkl"]
  }
}
```

Arguments arrive in Definition order — inputs first, then outputs when
`destination_passing_style` is true. Export the entry point with
`TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_kernel, MyFunction)` or the module will load without it.

Sources use `.cpp`; SYCL is C++. The `sycl` language tag is what selects the compiler and
adds `-fsycl`.

For agents generating kernels, `flashinfer_bench.SYCL_PROMPT` carries the full guidance —
sub-group sizing, shared local memory, matrix units, and the common mistakes.

## Toolchain discovery

The builder looks for oneAPI DPC++ in this order:

1. `FIB_SYCL_COMPILER` (explicit path)
2. `CXX`, when it already names `icpx` or `dpcpp`
3. `PATH`
4. `ONEAPI_ROOT`, then `/opt/intel/oneapi` — so an installed-but-unsourced oneAPI is found

## Ahead-of-time vs JIT

With no known target architecture, kernels compile to SPIR-V and are JIT-compiled at load
time. That is the portable default, and it is what lets a kernel run on an Intel GPU
released after the kernel was written. When a device's AOT triple is known
(`Capabilities.sycl_target`), the builder passes `-fsycl-targets` and compiles ahead of
time instead.
