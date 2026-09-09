# _C::rms_norm

Pulled by `scripts/pull_kernel_source.py`. Everything here describes what **actually runs**
on `xpu:0` -- the dispatcher was asked, not a table.

| | |
| --- | --- |
| dispatch key | `XPU` |
| registered at | `/workspace/vllm_xpu_kernel/csrc/torch_bindings.cpp` |
| providing project | `vllm-xpu-kernels` |
| harness | `harness.py` -- calls this op at a shape the model called it at |

## Schema

```
_C::rms_norm(Tensor($0! -> ) result, Tensor input, Tensor? weight, float epsilon) -> ()
```

## Source

- `source/csrc/torch_bindings.cpp` (356 lines) -- registered here (csrc/torch_bindings.cpp)
- `source/csrc/layernorm.cpp` (884 lines) -- defines it
- `source/csrc/ops.h` (308 lines) -- defines it

The kernel is copied whole rather than sliced: the launcher, the functor and the dispatch
macros are all part of what you are changing, and a slice that keeps only the arithmetic
compiles into a different kernel.

## Before optimizing

Bound it. `flashinfer_bench.device.calibration.get()` gives this part's substitution cost,
timing floor and achievable bandwidth; a win smaller than what taking it costs is a loss.
See `.claude/skills/route-kernel-work/PLAN.md`.

Then: `python scripts/kernel_trials.py init _C_rms_norm <this-dir>/harness.py`
