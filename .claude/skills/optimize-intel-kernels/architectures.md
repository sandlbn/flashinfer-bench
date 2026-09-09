# Intel GPU architectures

What a kernel author needs that the capability record cannot answer. Everything else is
queryable — query it.

## Query the device, do not look it up

```python
from flashinfer_bench.device import get_accelerator
caps = get_accelerator("xpu:0").capabilities("xpu:0")

caps.preferred_sub_group_size   # elementwise sub-group width; matrix kernels pin 16
caps.vector_width(2)            # elements per widest access for a 2-byte dtype
caps.supports_large_grf         # gates -ze-opt-large-register-file
caps.supported_dtypes           # native ∪ emulated; pair with is_native_dtype()
caps.emulated_dtypes            # runs correctly, but not at the format's throughput
caps.l2_bytes, caps.sycl_target
```

Branch on capabilities, never on `canonical_id`.

`preferred_sub_group_size` is the **elementwise** width. Every DPAS-backed kernel pins 16
regardless of what the device reports, and Triton overrides it to 16 anyway — see
[`xe-matrix.md`](xe-matrix.md) before writing anything with `tl.dot` or XMX.

## Parts

| | Battlemage | Xe3.0 integrated | Crescent Island |
| --- | --- | --- | --- |
| Example | Arc B580 / B570 / Pro B50-B60 | Panther Lake, Wildcat Lake | — |
| Device IP (`props.version`) | 20 | 30 | 35 |
| `sycl_target` | `bmg` | none (SPIR-V JIT) | `cri` |
| Baselines available | `vllm-xpu`, `sgl-kernel-xpu`, in-tree | `vllm-xpu`, in-tree | `sgl-kernel-xpu` (pre-silicon) |
| Status | validated hardware | development only | pre-silicon |

Get the IP with `torch.xpu.get_device_properties(0).version`.

A part with no `sycl_target` compiles to SPIR-V and JITs at load. An integrated Xe3.0 part
runs SYCL and Triton solutions and is a usable development machine, but cannot run
`sgl-kernel-xpu`; do not draw performance conclusions on one.

### Adding a part

Add a row to `_PART_PROFILES` in `flashinfer_bench/device/xpu.py` keyed by canonical id,
carrying `sycl_target` and any capability overrides:

```bash
python -c "import torch; p=torch.xpu.get_device_properties(0); print(p.name, p.version)"
python -c "from flashinfer_bench.device import get_accelerator; print(get_accelerator('xpu:0').capabilities('xpu:0'))"
```

A part with no row still works — conservative defaults plus SPIR-V JIT.

## Design rules the capability record cannot express

### Rows narrower than a sub-group need multi-row work-groups

When `hidden / vector_width < preferred_sub_group_size`, pack several rows into a
work-group. Local ids linearize with the last dimension fastest, so a `(rows, sub_group)`
work-group puts each row in exactly one sub-group, and the row reduces with
`reduce_over_group(it.get_sub_group(), ...)`.

An early `return` for out-of-range rows is safe only because every lane of a sub-group
shares a row, so the exit is sub-group uniform.

### AOT target spelling

`-fsycl-targets=bmg` is not valid — `bmg` is an ocloc device name. The AOT spelling is
`-fsycl-targets=spir64_gen` with `-Xs -device -Xs bmg`, on the **link** step.
`intel_gpu_bmg_g21` is a valid single-token alias; there is no published equivalent for
`cri`. `SyclBuilder` derives this from the capability record
(`flashinfer_bench/compile/builders/sycl_builder.py`).

### Measuring small kernels

At decode batch sizes launch overhead dominates. Use the accelerator's own timer
(`flashinfer-bench run`, or `get_accelerator(dev).make_timer(dev)`), one process per
comparison, and report medians. Set a performance power profile first
(`docs/start/hardware-support.mdx`).

### A row pitch on the memory-channel period streams at reduced bandwidth

A 2-D weight whose row pitch in bytes is a multiple of the memory-channel period
(`flashinfer_bench.integration.weight_layout.channel_period_bytes()`) puts every row on the
same channel, and a decode GEMM that walks many rows at one column offset serialises on it;
a plain row reduction over the same tensor slows too, less. It is invisible while the weight
is cache-resident, so a one-weight loop cannot see it — stream a pool larger than
`caps.l2_bytes`. No driver interface reports the period, so it is calibrated per part:
`calibration.get().channel_period_bytes` sweeps the row pitch of a streaming read and takes
the spacing of the pitches at which it is slow (`scripts/calibrate_part.py` prints it), and
where the sweep resolves none the transform stands down rather than borrow a period. The
fix (`pad_rows_off_channel_period`) is keyed on the pitch arithmetic and verified per shape
at load, because the pad can lose at another height of the same pitch. The
harness A/B that cross-checks one pitch: `tools/kernel-harness/trials/linear_row_pad.py`.

## Traps when deploying a kernel through `apply()`

### The default correctness gate rejects any bf16 kernel that is not bit-exact

`ApplyConfig` defaults to `max_atol=1e-2, max_rtol=1e-5` regardless of dtype
(`flashinfer_bench/apply/config.py`). bfloat16's relative spacing is `2**-7`, so any kernel
that is correct but not bit-identical fails the default gate.

**Symptom:** every call reports `no-solution`.
**Action:** pass a dtype-appropriate gate, e.g.
`enable_apply(path, ApplyConfig(max_atol=2e-2, max_rtol=2e-2))`.

### A definition's name carries its shape, not its dtype

`<op>_h<width>` from an fp16 run and `<op>_h<width>` as served in bf16 are the same name.
The apply index keys on axes alone, and `on_miss_policy="use_def_best"` skips the key
entirely.

**Action:** extract definitions in the dtype the model is served in
(`scripts/extract_model_kernels_xpu.py` reads it from the model config; `--dtype` overrides).

### Verify substitution before believing an A/B

`apply` falls back silently when no definition of that shape exists, when no solution
clears the gate, or when the runtime key matches no recorded workload.

**Action:** `flashinfer_bench.integration.vllm.adapters.stats` prints per-adapter counts to
stderr at exit. If `applied` is zero there is no result.

### Declare every output the caller consumes

If the upstream kernel produces a value in place — a fused residual-add norm's summed
residual — the definition must declare it as an output, or the adapter recomputes it in
eager PyTorch. Count the bytes the caller moves, not the bytes inside your kernel.

### Match the upstream kernel's allocation behaviour

vLLM's norm kernels are in-place. Alias input and output when the kernel writes index `i`
only after reading index `i`, and pin that property with a test.

### Dispatch cost sets a floor on what is worth substituting

`apply()` resolves the definition, builds a key, looks it up and checks dtypes on every
call. The cost is the same whatever the kernel costs; read it from
`flashinfer_bench.device.calibration.get()` (`scripts/calibrate_part.py` prints it).

**Decision:** large kernels → `apply()` is fine. Elementwise kernels at decode sizes → bind
once at model load (resolve the solution and install the `Runnable` as the layer's
forward), or expect no throughput win.

## Adding to this file

Only a rule that neither the capability record nor a trace can express. No measured values;
a threshold reads from the calibration.
