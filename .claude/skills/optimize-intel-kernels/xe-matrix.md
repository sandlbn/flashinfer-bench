# Writing matrix kernels on Xe

Rules for DPAS-backed work — GEMM, attention, quantized matmul — established from Intel's
own sources. `architectures.md` covers everything else.

Sources: `tmp/sycl-tla` and `tmp/oneDNN` (from `/clone-repos`), and `tmp/intel-triton`
(clone github.com/intel/intel-xpu-backend-for-triton yourself when you need its lowering
passes).

Battlemage is `SYCL_INTEL_TARGET == 20`. **Anything in sycl-tla guarded by `== 35` is
Crescent Island and does not exist on Battlemage.** Check that guard before porting a kernel
from that repo.

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

## What precisions Battlemage has

Available: tf32, bf16, fp16, int8, int4, and int8 × int4 mixed.

Absent on Battlemage, present only on Crescent Island: fp8 (e4m3/e5m2) DPAS, e2m1 DPAS,
and `XE_BDPAS_TT` — the block-scaled MX instruction, i.e. native mxfp8/mxfp4
(`mma_xe.hpp`). int8×int4 mixed DPAS is removed on CRI ("Skip int8 x int4 for CRI as the
dpas is removed").

**fp8 runs on Battlemage by emulation.** sycl-tla's `examples/08_bmg_gemm_f8` and PyTorch's
`torch._scaled_mm` upconvert to fp16 and use the fp16 DPAS, so fp8 is slower than the same
GEMM in bf16 here, and bit-exact against an fp32-upcast reference. An FP8 model on this part
buys memory capacity, not speed; never report it as an fp8 speedup. `Capabilities` lists fp8
under `emulated_dtypes`, and `validate-references` labels the result
`[emulated: ... -- latency is not native]`.

When the source dtype has no DPAS, convert to **fp16** rather than bf16
(`examples/cute/tutorial/xe_gemm.cpp`, "upconversion sequences are typically faster").

### Where Intel's Xe2 GEMM tuning is

oneDNN's `kernel.db` carries no Xe2-native bf16 × bf16 strategies; bf16 falls back to the
XeHPC/PVC table (`kernel_selector.cpp`). Xe2-specific entries are low-bit weights against
f16 compute. Count them yourself when choosing a target — the hardware tag is the leading
character (`gemmstone/kernel_catalog.hpp`: `'G'` = Xe2, `'F'` = XeHPC), and the two strings
after `"gemm"` are the A and B precisions:

```bash
DB=tmp/oneDNN/src/gpu/intel/gemm/jit/selector/db/kernel.db
grep -oE "^\{\{'G', \"gemm\", \{\"[A-Z0-9]+\", \"[A-Z0-9]+\"" "$DB" | sort | uniq -c | sort -rn
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
- VNNI only for 8/16-bit; transpose only for 32/64-bit — and on Xe2, transpose width ≤ 8
  for d32, ≤ 4 for d64 (Xe3P+ doubles both)

Address constraints: base pointer **64-byte aligned**; width and pitch multiples of 4 bytes
(16 in practice); x-offset a multiple of 4 elements; width/height/pitch < 2²⁴.

Block-2D gives free out-of-bounds zero-fill, so remainder tiles need no load masking. A
shape that violates the constraints loses the whole mechanism.

When block-2D is unavailable (gathered rows for GQA/MoE, misalignment), the fallback is SLM
staging with `XE_1D_LDSM` or `XE_1D_LOAD_GLOBAL`, using d16u32/d8u32 messages for narrow
types — slower, and worth restructuring the data to avoid.

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

Heuristics from `examples/cute/tutorial/xe_gemm.cpp`: K-tile is 2× the DPAS K unless A is
K-major or B is N-major byte data, in which case 1×; use a 4×8 sub-group layout instead of
8×4 when B is narrower than A.

## Schedulers

Xe has two: `KernelXe` (one output tile per WG, non-persistent) and
`KernelXeCooperative` (persistent, grid = `min(sm_count, tiles)`, driving
`PersistentScheduler` or `StreamKScheduler`). `KernelXePtrArrayCooperative` handles
grouped/pointer-array.

**There is no Pingpong or warp-specialized schedule for Xe2.** Do not port one.

On Intel, "sm_count" is XeCore count — `gpu_slices × gpu_subslices_per_slice`.

Stream-K self-disables: it assigns stream-K units only when there is wave quantization
*and* `k_tiles_per_output_tile > 8`. Reduction is deterministic (turnstile) by default.

## GRF mode

Kernels launch with `sub_group_size<16>`; BMG examples add `grf_size<256>` (CRI uses 512).
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
