# Intel GPU architectures

What a kernel author needs that the capability record cannot answer. Everything else is
queryable — query it.

## Query the device, do not look it up

```python
from flashinfer_bench.device import get_accelerator
caps = get_accelerator("xpu:0").capabilities("xpu:0")

caps.preferred_sub_group_size   # elementwise sub-group width; matrix kernels pin 16
caps.vector_width(2)            # elements per 16-byte access for a 2-byte dtype
caps.supports_large_grf         # gates -ze-opt-large-register-file
caps.supported_dtypes           # Battlemage has no FP8 — check before targeting a dtype
caps.l2_bytes, caps.sycl_target
```

A kernel that branches on `canonical_id == "INTEL_ARC_B580"` breaks on the next part; one
that reads `preferred_sub_group_size` does not.

`preferred_sub_group_size` is the **elementwise** width (32 on Battlemage). Every DPAS-backed
kernel pins 16 regardless of what the device reports, and Triton silently overrides it to 16
anyway — see [`xe-matrix.md`](xe-matrix.md) before writing anything with `tl.dot` or XMX.

## Parts

| | Battlemage | Xe3.0 integrated | Crescent Island |
| --- | --- | --- | --- |
| Example | Arc B580 / B570 / Pro B50-B60 | Panther Lake, Wildcat Lake | — |
| Device IP (`props.version`) | 20 | 30 | 35 |
| `sycl_target` | `bmg` | none (SPIR-V JIT) | `cri` |
| Baselines available | `vllm-xpu`, `sgl-kernel-xpu`, in-tree | `vllm-xpu`, in-tree | `sgl-kernel-xpu` (pre-silicon) |
| Status | validated hardware | development only | pre-silicon |

Get the IP with `torch.xpu.get_device_properties(0).version`.

A part with no `sycl_target` compiles to SPIR-V and JITs at load. That is the portable path
and how an unreleased device runs on the day it arrives — a feature, not a gap.

An integrated Xe3.0 part runs SYCL and Triton solutions fine and is a good development
machine. Do not *conclude* anything on one: it cannot run `sgl-kernel-xpu` at all, and its
performance characteristics differ enough from discrete Arc to invert a verdict.

### Adding a part

Data, not code. Add a row to `_PART_PROFILES` in `flashinfer_bench/device/xpu.py` keyed by
canonical id, carrying `sycl_target` and any capability overrides:

```bash
# 1. identify the part
python -c "import torch; p=torch.xpu.get_device_properties(0); print(p.name, p.version)"
# 2. add the _PART_PROFILES row, then confirm it took effect
python -c "from flashinfer_bench.device import get_accelerator; print(get_accelerator('xpu:0').capabilities('xpu:0'))"
```

A part with no row still works — conservative defaults plus SPIR-V JIT.

## Design rules the capability record cannot express

### Rows narrower than a sub-group need multi-row work-groups

When `hidden / vector_width < preferred_sub_group_size`, one work-group per row leaves most
of a sub-group idle and launches one group per row.

Pack several rows into a work-group instead. SYCL makes this cheaper than the CUDA-style
version: local ids linearize with the last dimension fastest, so a `(rows, sub_group)`
work-group puts **each row in exactly one sub-group**, and the row reduces with a plain
`reduce_over_group(it.get_sub_group(), ...)` — no hand-written shuffle tree.

An early `return` for out-of-range rows is safe here *only* because every lane of a
sub-group shares a row, so the exit is sub-group uniform and the collective is still
reached by all participating lanes.

### AOT target spelling

`-fsycl-targets=bmg` is not valid — `bmg` is an *ocloc device name*. The AOT spelling is
`-fsycl-targets=spir64_gen` with the device passed to the backend (`-Xs -device -Xs bmg`),
and the flags must be on the **link** step, because device code is generated at link.
`intel_gpu_bmg_g21` is a valid single-token alias; there is no published equivalent for
`cri`. `SyclBuilder` derives all of this from the capability record — see
`flashinfer_bench/compile/builders/sycl_builder.py`.

### Measuring small kernels

At decode batch sizes these kernels run single-digit microseconds, where launch overhead
dominates and repeated wall-clock readings scatter by more than the effect. Use the
accelerator's own timer (`flashinfer-bench run`, or `get_accelerator(dev).make_timer(dev)`),
one process per comparison, and report medians. Never report a single small-batch
wall-clock figure as a result.

Set a performance power profile before measuring anything — `docs/start/hardware-support.mdx`
carries the command and the variance it costs.

## Traps when deploying a kernel through `apply()`

A kernel that wins in the benchmark can still lose in a server. These are the reasons, in
the order worth checking.

### The default correctness gate rejects any bf16 kernel that is not bit-exact

`ApplyConfig` defaults to `max_atol=1e-2, max_rtol=1e-5` regardless of dtype
(`flashinfer_bench/apply/config.py`). bfloat16's worst-case relative spacing is `2**-7`, so a
kernel differing from the reference by a *single ULP* reports ~0.0078 relative error and
fails that gate by three orders of magnitude.

A kernel that reproduces the reference bit-for-bit — which a norm accumulating in float32
the same way often does — reports 0.0 and passes. Measured over this dataset: 47% of passing
traces are bit-exact and clear the strict gate; 52% are rejected. So the gate does not block
bf16 as such; it blocks every bf16 kernel that is merely *correct* rather than identical,
which includes most GEMM-shaped and fused work.

**Symptom:** every call reports `no-solution` and the server runs unchanged.
**Action:** pass a dtype-appropriate gate, e.g.
`enable_apply(path, ApplyConfig(max_atol=2e-2, max_rtol=2e-2))`.

### A definition's name carries its shape, not its dtype

`rmsnorm_h1024` extracted from an fp16 run and `rmsnorm_h1024` as vLLM serves it in bf16 are
the same name. The apply index keys on axes alone, and `on_miss_policy="use_def_best"` skips
the key entirely.

**Action:** extract definitions in the dtype the model is actually served in
(`scripts/extract_model_kernels_xpu.py` reads it from the model config; `--dtype` overrides).
Treat `use_def_best` as trading dtype and shape matching for hit rate.

### Verify substitution before believing an A/B

A patched method proves nothing: `apply` falls back silently and correctly when no
definition of that shape exists, when no solution clears the gate, or when the runtime key
matches no recorded workload.

**Action:** `flashinfer_bench.integration.vllm.adapters.stats` records the outcome of every
adapter call and prints counts to stderr at exit. If `applied` is zero there is no result —
fix that before reading the throughput number.

### Declare every output the caller consumes

If the upstream kernel produces a value in place — a fused residual-add norm's summed
residual — the definition must declare it as an output. Otherwise the adapter recomputes it
in eager PyTorch, a whole extra pass over the activation, costing more than the fusion saves.

**Rule:** count the bytes the *caller* moves, not the bytes inside your kernel.

### Match the upstream kernel's allocation behaviour

vLLM's norm kernels are in-place. Allocating a fresh output per call is a throughput
regression with no device-time cause.

**Action:** alias input and output when the kernel writes index `i` only after reading index
`i`, and pin that aliasing property with a test — it is a property of the kernel, not of the
interface.

### Dispatch cost sets a floor on what is worth substituting

`apply()` resolves the definition, builds a key from the arguments, looks it up and checks
dtypes on every call. That costs a few microseconds — negligible against a large GEMM,
comparable to the whole kernel for a decode-size elementwise op.

**Decision:** large kernels → `apply()` is fine. Elementwise kernels at decode sizes → bind
once at model load (resolve the solution and install the `Runnable` as the layer's forward),
or expect no throughput win however fast the kernel is.

## Adding to this file

Only what neither the capability record nor a trace can express: a trap, or a threshold that
changes what an agent does. Every number names the part it was measured on and the date. A
number that merely records a past measurement belongs in a trace, not here.
