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

The per-part rows are data, not documentation. They live in `_PART_PROFILES`
(`flashinfer_bench/device/xpu.py`), keyed by canonical device id, and the driver supplies
the rest. Print them instead of reading a copy that drifts:

```bash
python -c "
from flashinfer_bench.device.xpu import _PART_PROFILES
for cid, prof in _PART_PROFILES.items():
    e = prof.get('extra', {})
    print(f\"{cid:30} target={str(prof.get('sycl_target')):5} ip={e.get('device_ip_version')} \"
          f\"{e.get('architecture','')} {e.get('status','')}\")
"
```

For the device in front of you, the record and the driver together:

```bash
python -c "
import torch
from flashinfer_bench.device import get_accelerator
caps = get_accelerator('xpu:0').capabilities('xpu:0')
props = torch.xpu.get_device_properties(0)
print(caps.canonical_id, '| driver version', props.version,
      '| ip', caps.extra.get('device_ip_version'),
      '| target', caps.sycl_target,
      '| integrated', caps.extra.get('is_integrated_gpu'),
      '|', caps.extra.get('profile', 'has a _PART_PROFILES row'))
"
```

`props.version` is the driver's device-IP string; its leading component is the IP version
that `_PART_PROFILES` records as `device_ip_version`. A `sycl_target` of `None` means no
AOT triple is known for the part: solutions compile to SPIR-V and JIT at load, which is how
a device runs before its triple is published. `caps.extra["profile"] == "driver-reported"`
means the part has no row at all and is running on conservative defaults — see "Adding a
part".

Which baselines exist is a property of this box, not of the silicon:
`flashinfer-bench providers list` prints each provider, whether it is installed, its
version and where it came from, and `flashinfer-bench providers verify --local
tmp/flashinfer-trace` calls each registered kernel once so a provider that is present but
not built for this backend fails there rather than inside a benchmark.

An integrated part shares the package power budget with the CPU. What that costs a
measurement is recorded next to `_RECOMMENDED_WARMUP_RUNS` in
`flashinfer_bench/device/xpu.py`: no clock ramp to warm up, but sporadic excursions while
power shifts between CPU and GPU that no amount of warmup removes. Read
`caps.extra["is_integrated_gpu"]` rather than inferring it from the device name, and report
the spread `scripts/kernel_trials.py benchmark` gives alongside the median.

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

A small kernel's measured time can sit at the launch floor, where nothing inside the kernel
is being measured; `launch_floor_us` and `timing_floor_us` from
`flashinfer_bench.device.calibration.get()` say where that is for this part, and a sweep of
the varying axis that does not move the time confirms it. Use the accelerator's own timer
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

**Decision:** compare the kernel's own per-call time (unitrace `-d`, avg column) against
`dispatch_us` from the same calibration. Where the kernel is large against it, `apply()`
carries the substitution. Where it is not, the dispatch is a fraction of the call: bind once
at model load instead — resolve the solution and install the `Runnable` as the layer's
forward — and measure both arms rather than deciding from the shape of the op.

## Adding to this file

Only a rule that neither the capability record nor a trace can express. No measured values;
a threshold reads from the calibration.
