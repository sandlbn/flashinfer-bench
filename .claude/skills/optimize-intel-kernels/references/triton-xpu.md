# Writing Triton for Intel XPU: what differs from CUDA Triton

Read from the Intel XPU backend checkout that `/clone-repos` places at `tmp/intel-triton`:
`AGENTS.md`, `third_party/intel/backend/compiler.py` (the launch options and what the
compiler does with them), `python/tutorials/` (the fused-softmax, layer-norm,
tensor-descriptor matmul, fused-attention, grouped-GEMM and persistent-matmul tutorials)
and `docs/BLOCK_LOADS_LAYOUT.md`. Only the parts about writing a kernel are taken. The
checkout's compiler-development material -- MLIR reproducers, lit tests, lowering rules,
build instructions, C++ style -- is that project's and is not imported here. Re-read the
files named above when the checkout moves; this page cites them, it does not replace them.

## Query the device through the driver

Every number an XPU kernel is shaped by is read at run time, never carried from another
part or from a CUDA habit:

```python
import torch, triton
from triton.runtime import driver

target = driver.active.get_current_target()          # .backend names the backend in use
props = driver.active.utils.get_device_properties(torch.xpu.current_device())
props["sub_group_sizes"]        # the values warp_size may take
props["max_work_group_size"]    # bound on num_warps * warp_size
props["max_num_sub_groups"]     # bound on num_warps
props["max_compute_units"]      # what a persistent grid is sized by
props["last_level_cache_size"]  # what a streaming measurement must exceed
props["local_mem_size"]         # shared-memory ceiling the compiler checks against
props["has_subgroup_2d_block_io"]                   # descriptor loads become 2D block loads
props["has_subgroup_matrix_multiply_accumulate"]    # tl.dot lowers to DPAS
```

The tutorials read the compute-unit count for persistent grids through
`torch.xpu.get_device_properties(...).gpu_eu_count`; either source is the device's own
answer, and the point is that it is asked, not assumed.

## The block programming model

From `AGENTS.md`. It is where the model differs from the CUDA thread model, so a kernel
transcribed from CUDA has to be re-read against it:

- A Triton program operates on logical blocks, not on hardware threads. Scalar values,
  scalar control flow and the origin of a memory descriptor are uniform across the block.
  Anything that varies per element is a tensor with a layout.
- Do not manufacture a lane- or sub-group-varying scalar through inline assembly, a
  hardware id, or any other back door. A kernel that does so is outside the programming
  model, and a wrong result from it is not a compiler bug to report.
- Block-uniform is not loop-invariant: a value uniform within one iteration may differ
  across iterations and across warp-specialized regions, and the synchronization the
  language documents for asynchronous operations still applies.

## Launch options that exist only on this backend

Read `XPUOptions` in `third_party/intel/backend/compiler.py` for the current set; the ones
an author sets are:

| Option | What it is on XPU | What constrains it |
| --- | --- | --- |
| `warp_size` | the sub-group width; a `Config` key the kernel does not take as a parameter | must be one of `props["sub_group_sizes"]` -- the compiler refuses any other value; a kernel the compiler lowers to DPAS is pinned to the matrix width regardless (`xe-matrix.md`) |
| `num_warps` | sub-groups per work-group | `num_warps <= props["max_num_sub_groups"]` and `num_warps * warp_size <= props["max_work_group_size"]` |
| `num_stages` | depth of the backend's prefetch pipeline pass (`add_pipeline`), not CUDA's async-copy pipeline | the checkout's own GEMM and attention tutorials sweep it on XPU; put it in the autotune space and let the measurement choose, rather than pinning a value carried from either vendor |
| `grf_mode` | registers per hardware thread: `'default'`, `'128'`, `'256'`, `'512'` where the part has it, `'auto'` | the large modes halve the threads resident per engine and are refused with `num_warps` above the limit the compiler states |

`grf_mode='default'` is not a fixed mode. The compiler builds with the small register file,
reads the spill size out of the binary, and on any spill at all rebuilds with the large
mode -- silently. A kernel that spilled therefore runs, correctly, at halved occupancy with
nothing printed. After `kernel = fn.warmup(...)`, `kernel.metadata.build_flags` carries the
GRF flag the binary was finally built with; a flag naming a large mode is the spill
report. The layer-norm tutorial pins `grf_mode='256'` on XPU and the tensor-descriptor
matmul tutorial sweeps `grf_mode` inside its configs: both are configurations to include
in a sweep, neither is a rule.

## Loads, descriptors and the matrix engine

- `tl.make_tensor_descriptor` is supported on XPU (the tensor-descriptor matmul,
  persistent-matmul, grouped-GEMM and fused-attention tutorials all use it there), and a
  block pointer is rewritten to a descriptor by the compiler. On a device that reports
  `has_subgroup_2d_block_io`, a descriptor load feeding `tl.dot` becomes a 2D block load.
- `docs/BLOCK_LOADS_LAYOUT.md`: the size of that block load is derived from the DPAS layout
  the compiler attaches to the `tl.dot`, then enlarged to cover as many DPAS instructions
  of the sub-group as the hardware allows. The author does not size it; the author shapes
  the dot's operand tiles, and `xe-matrix.md` owns those shapes. A transposed B operand is
  transposed inside the load, at the cost the same document describes for narrow types.
- Dot operands are loaded from memory into registers directly; there is no shared-memory
  staging of A and B as on the other vendor. Reasoning about "SLM tiles" for a GEMM is
  reasoning about a machine this is not.
- `tl.dot` on this backend takes `default_dot_input_precision` from the options; the
  in-tree GEMM passes `allow_tf32=False` and accumulates in fp32. Arithmetic in bf16 that
  the part cannot do natively is emulated in fp32 by the pipeline; accumulate in fp32 and
  convert at the store.

## Kernel shapes the tutorials establish on XPU

- **Row-wise (softmax, norms).** `warp_size` taken from `props["sub_group_sizes"]`,
  `num_warps` derived from the block width and capped by
  `max_work_group_size // warp_size`; several rows per program when a row is narrower than
  a work-group (`_rows_per_program` in the in-tree preamble derives it from
  `max_work_group_size`). The softmax tutorial pre-compiles with `kernel.warmup(...)`,
  reads `kernel.metadata.shared`, and sizes a persistent grid from occupancy; the SLM
  allocation granularity it tabulates is marked in that file as a value it could not yet
  read from the device -- do not copy the table.
- **Persistent GEMM.** Grid of `min(tiles, compute_units)`, tile ids walked with a
  grouped ordering; descriptors created once per program. Tile sizes come from the
  autotune space, and the checkout's XPU config lists differ from its CUDA lists -- take
  the XPU list as a starting sweep, then measure.
- **Attention.** Descriptor loads for Q, K, V and O, `num_stages` swept; the
  warp-specialized path is gated on the other vendor's parts and is not a knob here.

## Autotuning

Autotune with `key=` on the size axes that move between workloads. Clear the Triton cache
(`${TRITON_CACHE_DIR:-$HOME/.triton/cache}`) when a result looks reused, and confirm the
chosen config differs between a small and a large workload; an autotuner that picked once
and never again produces the same latency at every size. `warp_size` belongs in the
elementwise sweep and not in a DPAS kernel's, where the compiler overrides it.

## Timing

The checkout's own benchmark harness (`benchmarks/triton_kernels_benchmark/benchmark_testing.py`)
takes its kernel times from the PyTorch profiler's device durations by default, the same
instrument `scripts/bound_candidates.py --measure` uses for `t_dev`. In this repository the
number that decides anything comes from `scripts/kernel_trials.py benchmark`, and from no
script of your own.

## What the checkout offers that this page does not use

`benchmarks/onednn_kernel/` (Triton against oneDNN, softmax), `benchmarks/sycl_tla_kernel/`
(Triton against SYCL-TLA, GEMM and attention) and `benchmarks/micro_benchmarks/` (dtype
conversions, descriptor loads, scaled dot) are that project's regression suite. They need
their C++ providers built, they publish per-part peaks in a stored table
(`benchmarks/gpu_info.json`), and they cover op classes the routing sends to the library
call. None of that is a calibrated input for this repository; the authored-stream probe in
`flashinfer_bench.device.calibration` is, because it is measured on the part in hand.
