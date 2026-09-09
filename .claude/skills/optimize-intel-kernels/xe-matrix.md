# Writing matrix kernels on Xe

Rules for DPAS-backed work — GEMM, attention, quantized matmul — established from Intel's
own sources. `architectures.md` covers everything else.

Sources: `tmp/sycl-tla` and `tmp/oneDNN` (from `/clone-repos`), and `tmp/intel-triton`
(clone github.com/intel/intel-xpu-backend-for-triton yourself when you need its lowering
passes).

Every Xe guard in sycl-tla branches on `SYCL_INTEL_TARGET`. It is not a part name: the
macro is redefined from the compiler's own target macro.

Source: `tmp/sycl-tla/include/cutlass/cutlass.h:39-46`, `tmp/sycl-tla/CMakeLists.txt:151-162`

| Build target | `SYCL_INTEL_TARGET` |
| --- | --- |
| `cri` / `intel_gpu_cri` — defines `__SYCL_TARGET_INTEL_GPU_CRI__` | 35 |
| every other entry of `INTEL_SYCL_TARGETS` | 20 |

So a block guarded by `== 35` is compiled only in a `cri` build; on any other target the
`#else` arm is what exists. Check that guard against the target this device builds as
before porting anything out of that repo:

```python
from flashinfer_bench.device import get_accelerator
get_accelerator(dev).capabilities(dev).sycl_target   # None -> SPIR-V JIT; see architectures.md
```

## Sub-group width is 16 for matrix work, and not negotiable

`caps.preferred_sub_group_size` is the **elementwise** answer. Matrix kernels pin 16:

- DPAS has a fixed execution size of 16 and N is fixed at 16 in hardware
  (`cute/arch/mma_xe.hpp`, `cute/atom/mma_traits_xe.hpp`).
- CUTLASS-SYCL hardcodes `constexpr int sg_size = 16` (`cute/util/sycl_vec.hpp:41`).
- Triton's Intel backend silently overrides the requested width to 16 for any kernel it
  can lower to DPAS (`third_party/intel/lib/TritonAnnotateModule/TritonAnnotateModule.cpp`,
  `setThreadsPerWarp`).

So autotuning a `tl.dot` kernel over `warp_size ∈ {16, 32}` measures the same kernel twice.
See `_GEMM_CONFIGS` in `flashinfer_bench/integration/intree_kernels.py`.

## DPAS shapes

One DPAS is `M×16×K` with systolic depth 8 and `1 <= M <= 8`. K is set by the widest
operand: `K = 256 / max(bits(A), bits(B))`.

| operands | K per DPAS |
| --- | --- |
| tf32 | 8 |
| bf16 / fp16 | 16 |
| int8 | 32 |
| int4 | 64 |

A is `M×K` row-major, B must be **VNNI-packed** (`K×16`, 32-bit groups along K), C is
`M×16` row-major. All three are work-item-interleaved: lane `i` owns elements
`i, i+16, i+32, …`. Reinterpreting between element widths at register level is a shuffle,
not a cast.

Do not put arbitrary computation between a VNNI load and the DPAS that consumes it — the
compiler inserts interleave/deinterleave around it and can spill
(`media/docs/cpp/xe_rearchitecture.md`).

## Which DPAS operand types exist

The operand-type declarations sit inside the guard above, so the set follows the build
target rather than a remembered part.

Source: `tmp/sycl-tla/include/cute/arch/mma_xe.hpp:230-300`

| Operand types declared | Where |
| --- | --- |
| tf32; bf16; fp16; int8 (u8/s8, either signedness in either position); int4 (u4/s4, likewise) | unguarded — every target |
| fp8 (`bf8`, `hf8`) DPAS, `e2m1` DPAS, and `XE_BDPAS_TT`, the block-scaled MX instruction (native mxfp8/mxfp4) | inside `#if SYCL_INTEL_TARGET == 35` |
| int8 × int4 mixed DPAS | inside that guard's `#else`, under the comment "Skip int8 x int4 for CRI as the dpas is removed" |

Whether a dtype reaches hardware *on this device* is a query, not a lookup:

```python
caps = get_accelerator(dev).capabilities(dev)
caps.is_native_dtype("float8_e4m3fn")   # False -> the fp8 path here is emulation
caps.emulated_dtypes                    # correct results, not the format's throughput
```

Where `is_native_dtype` returns False for fp8, sycl-tla's `examples/08_bmg_gemm_f8` and
PyTorch's `torch._scaled_mm` upconvert to fp16 and run the fp16 DPAS; the result is
bit-exact against an fp32-upcast reference. `validate-references` labels such a run
`[emulated: ... -- latency is not native]`, so the latency measured is the emulation
sequence's and is not reportable as an fp8 result. What the format still buys is the bytes
it moves, which `bytes_min` accounts for.

When the source dtype has no DPAS it has to be converted to one that does, and both fp16
and bf16 are available above. Which of the two the upconversion sequence costs least in is
a property of the sequence the compiler emits, not of the format: build the kernel both
ways and benchmark. sycl-tla's own sequences are in
`examples/cute/tutorial/xe_gemm.cpp`.

### Which oneDNN jit-GEMM strategies a target can reach

Every entry in `kernel.db` carries a one-character hardware tag, and the selector walks a
fallback chain when the target's own tag has no entry for the problem. A generation with
few entries of its own therefore runs an older generation's strategies.

Source: `tmp/oneDNN/src/gpu/intel/gemm/jit/include/gemmstone/kernel_catalog.hpp:75-82`, `tmp/oneDNN/src/gpu/intel/gemm/jit/selector/kernel_selector.cpp:289-296`

| Tag | Generation | Falls back to |
| --- | --- | --- |
| `'C'` | Gen12LP | — |
| `'E'` | XeHPG | — |
| `'F'` | XeHPC | — |
| `'G'` | Xe2 | `'F'` |
| `'H'` | Xe3 | `'G'` |
| `'I'` | Xe3p | `'H'` |

Which precisions a tag has entries of its own for is a count off the db, not something to
carry: the two strings after `"gemm"` are the A and B precisions. Set `TAG` to the row you
are targeting and count, then repeat for the tag it falls back to — the difference is what
that generation was actually tuned for.

```bash
DB=tmp/oneDNN/src/gpu/intel/gemm/jit/selector/db/kernel.db
TAG=G
grep -oE "^\{\{'$TAG', \"gemm\", \{\"[A-Z0-9]+\", \"[A-Z0-9]+\"" "$DB" | sort | uniq -c | sort -rn
```

## 2D block copy

The transfer mechanism for every operand. `XE_LOAD_2D<Bits,Height,Width,BlockWidth>`,
`XE_LOAD_2D_VNNI`, `XE_LOAD_2D_TRANSPOSE`, `XE_STORE_2D`, `XE_PREFETCH_2D` — each is one
vISA `lsc_load_block2d.ugm` / `lsc_store_block2d.ugm`.

Hard limits (static-asserted in `cute/arch/copy_xe_2d.hpp`):

- Height ≤ 32 on loads, **≤ 8 on stores**
- `Bits × Width ≤ 512` (one 64-byte row)
- block count ∈ {1,2,4}, `Bits × Count ≤ 64`
- element size ∈ {8,16,32,64}
- VNNI only for 8/16-bit; transpose only for 32/64-bit, and the transpose width limit is
  itself inside the target guard (`copy_xe_2d.hpp:140-147`): under `== 35`, d32 width ≤ 16
  and d64 width ≤ 8 at height 8; on the `#else` arm, d32 width ≤ 8 and d64 width ≤ 4 at
  height 8

Address constraints: base pointer **64-byte aligned**; width and pitch multiples of 4 bytes;
x-offset a multiple of 4 elements; width/height/pitch < 2²⁴.

Block-2D gives free out-of-bounds zero-fill, so remainder tiles need no load masking. A
shape that violates the constraints loses the whole mechanism.

When block-2D is unavailable (gathered rows for GQA/MoE, misalignment), the fallback is SLM
staging with `XE_1D_LDSM` or `XE_1D_LOAD_GLOBAL`, using d16u32/d8u32 messages for narrow
types. Restructuring the data so the constraints above hold restores block-2D; where both
forms are buildable, that is a candidate to measure against the fallback.

Create the address payload once and update only X/Y per copy (`prepare_payloads`,
`copy_with_multi_payloads`, `update_payloads`).

## The mainloop is prefetch-staged, not double-buffered

There is **no SLM double-buffering in any production Xe mainloop**. A CUDA kernel built
around `cp.async` into shared memory does not transfer structurally.

The Xe pattern (`cutlass/gemm/collective/xe_mma.hpp`):

```
prefetch K-tiles [0, Stages)                    # into L1, cooperatively across the WG
for each k_tile:
    barrier_arrive(workgroup)                   # split barrier
    block-2D load A, B  ->  registers
    prefetch k_tile + Stages
    reorder A, B        ->  MMA fragments
    cute::gemm(...)
    barrier_wait(workgroup)
```

`Stages` is a **prefetch distance in K-tiles**, not a buffer count. `SharedStorageSize` is
0 for the plain GEMM kernel; SLM appears only in CuTe tutorials and in FMHA's cross-sub-group
softmax reduction.

## Tile shapes that Intel ships

All use sub-group 16 and, unless noted, an 8×4 sub-group layout = 32 sub-groups = 512
work-items per work-group. Starting points to benchmark, not answers.

| Case | WG tile | SG layout | Stages |
| --- | --- | --- | --- |
| bf16/fp16 GEMM (default) | `256×256×32` | 8×4 | 2 |
| bf16 GEMM (throughput benchmark) | `512×256×32` | 8×4 | 2 |
| small-M GEMM | `8×128×32` / `16×64×32` | 1×4 / 2×4 | 2 |
| mixed bf16×s8 | `256×256×32` | 8×4 | 3 |
| mixed fp16×u4 (A transposed) | `16×64×64` | 1×2 | 3 |
| fp8 (emulated) | `256×256×32` | 8×4 | 2 |
| dual GEMM / SwiGLU | `128×128×64` | 8×4 | 2 |
| row-softmax epilogue | `32×512×32` | 2×16 | 3 |
| MoE / grouped, skewed M | `256×128×32` | 8×2 | 3 |
| FMHA prefill hd64/96 | QK `128×64×32`, PV `128×32×64` | 8 SGs | 2 |
| FMHA prefill hd128 | QK/PV `256×32×32` | 16 SGs | 2 |
| FMHA decode | QK `1×512×64`, PV `1×32×512` | 8 SGs (split on KV) | 1 |

`choose_tiled_mma` derives the K-tile and the sub-group layout from the operand layouts
and widths rather than from the case at hand:

Source: `tmp/sycl-tla/examples/cute/tutorial/xe_gemm.cpp:199-214`

| Choice | Condition in that function |
| --- | --- |
| K-tile `op.K` rather than `op.K*2` | `a_t` (A is K-major), or `byte` (widest operand at most 8 bits) and `b_n` (B is N-major) |
| `SGLayout4x8` rather than `SGLayout8x4` | `sizeof_bits_v<TB> < sizeof_bits_v<TA>`, or `b_n` and B narrower than 8 bits |

## Schedulers

Xe has two: `KernelXe` (one output tile per WG, non-persistent) and
`KernelXeCooperative` (persistent, grid = `min(sm_count, tiles)`, driving
`PersistentScheduler` or `StreamKScheduler`). `KernelXePtrArrayCooperative` handles
grouped/pointer-array.

Those three are the whole set — sycl-tla defines no Pingpong or warp-specialized Xe
schedule, so there is none to port to. Re-establish with
`grep -rho 'KernelXe[A-Za-z]*' tmp/sycl-tla/include | sort -u`
(`include/cutlass/gemm/dispatch_policy.hpp:147-149`).

`sm_count` on this backend is the Xe-core count, not an EU count. CUTLASS computes it from
`gpu_slices × gpu_subslices_per_slice` (`include/cutlass/kernel_hardware_info.h:69-71`);
PyTorch's own SM-count equivalent on XPU is `props.gpu_subslice_count`
(`torch/_inductor/runtime/hints.py`, `multi_processor_count`), which reaches the capability
record as `caps.extra["gpu_subslice_count"]`.

Stream-K self-disables: it assigns stream-K units only when there is wave quantization
*and* `k_tiles_per_output_tile > 8`. Reduction is deterministic (turnstile) by default.

## GRF mode

Kernels launch with `sub_group_size<16>`, and the examples set the register mode from the
same target guard: `grf_size<512>` under `== 35`, `grf_size<256>` otherwise
(`examples/12_xe20_moe_gemm_cute_interface/12_xe20_moe_gemm_cute_interface.cpp:324-329`,
`benchmarks/flash_attention/benchmark_runner.hpp:907-914`).
CUTLASS's generic `GemmUniversalAdapter` sets only the sub-group size, so for library GEMMs
the register mode comes from the environment. Intel's own CI exports:

```bash
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
export IGC_VectorAliasBBThreshold=100000000000
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
```

For our own AOT SYCL solutions the equivalent is `FIB_SYCL_LARGE_GRF=1`, which `SyclBuilder`
turns into `-Xs -options -Xs -ze-opt-large-register-file` on the link step. That covers AOT
only; a SPIR-V JIT build takes its register mode from the env vars above. State which mode a
measurement was taken in.

No CMake flag sets GRF mode, and no `reqd_work_group_size` / `work_group_size_hint`
attribute appears in sycl-tla.

## Adding to this file

A rule an agent would otherwise get wrong, with the file that establishes it. Values that
`Capabilities` can answer belong there; measurements belong in a trace.
