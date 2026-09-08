# Writing matrix kernels on Xe

Rules for DPAS-backed work — GEMM, attention, quantized matmul — established by reading
Intel's own sources rather than inferred. `architectures.md` covers everything else.

Sources, cloned by `/clone-repos`, except `tmp/intel-triton` (the Intel Triton backend), which you clone yourself from github.com/intel/intel-xpu-backend-for-triton when you need to read its lowering passes
kernels), `tmp/intel-triton` (Triton's Intel backend), `tmp/oneDNN`.

Battlemage is `SYCL_INTEL_TARGET == 20`. **Anything in sycl-tla guarded by `== 35` is
Crescent Island and does not exist on B580.** That guard is the single most important thing
to check before porting a kernel from that repo.

## Sub-group width is 16 for matrix work, and not negotiable

`caps.preferred_sub_group_size` reports 32 on Battlemage. That is the **elementwise**
answer. Matrix kernels pin 16:

- DPAS has a fixed execution size of 16 and N is fixed at 16 in hardware
  (`cute/arch/mma_xe.hpp`, `cute/atom/mma_traits_xe.hpp`).
- CUTLASS-SYCL hardcodes `constexpr int sg_size = 16` (`cute/util/sycl_vec.hpp:41`). There
  is no 32-lane path anywhere in the library.
- Triton's Intel backend **silently overrides** the requested width to 16 for any kernel it
  can lower to DPAS: `setThreadsPerWarp` sets `ttg::AttrNumThreadsPerWarp` to `minSGSize`
  and returns, ignoring the option value
  (`third_party/intel/lib/TritonAnnotateModule/TritonAnnotateModule.cpp:119-151`).

The Triton override is why this matters practically: autotuning a `tl.dot` kernel over
`warp_size ∈ {16, 32}` is not coverage. Both configs compile to the same kernel and the
sweep measures it twice. See `_GEMM_CONFIGS` in
`flashinfer_bench/integration/intree_kernels.py`.

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
compiler inserts interleave/deinterleave around it and can spill to private memory
(`media/docs/cpp/xe_rearchitecture.md`).

## What precisions Battlemage actually has

Available: tf32, bf16, fp16, int8, int4, and **int8 × int4 mixed**.

Absent on Battlemage, present only on Crescent Island: fp8 (e4m3/e5m2) DPAS, e2m1 DPAS,
and `XE_BDPAS_TT` — the block-scaled MX instruction, i.e. native mxfp8/mxfp4
(`mma_xe.hpp:254-283`). `media/docs/cpp/xe_bdpas_unified_block_scaled_mma.md` documents an
instruction this part does not have.

Note the direction of travel: int8×int4 mixed DPAS is kept on Battlemage and **removed on
CRI** — the source comment is "Skip int8 x int4 for CRI as the dpas is removed."

**fp8 still runs on Battlemage, by emulation, and PyTorch already does it.**
sycl-tla's `examples/08_bmg_gemm_f8` and its FMHA runner upconvert fp8 to fp16
(`convert_FP8_to_FP16`) and use the fp16 DPAS. Measured on Arc B580, 4096x4096x4096:

| | latency | throughput |
| --- | --- | --- |
| `a_bf16 @ b_bf16.t()` | 1.262 ms | 108.9 TFLOP/s |
| `torch._scaled_mm` (e4m3) | 2.606 ms | 52.7 TFLOP/s — **0.48x** |

`_scaled_mm` is **bit-exact** against an fp32-upcast reference (max rel err 0.0), so this
is a throughput property, not a numerics one. Casts in both directions work.

So an FP8 model on this part buys **memory capacity, not speed** — half the weight bytes,
roughly half the GEMM rate. That is often still the right trade at 12 GiB, but never call
it an fp8 speedup.

This is why `Capabilities` separates `supported_dtypes` (native) from `emulated_dtypes`:
fp8 is in the latter on every Battlemage part, so it benchmarks rather than being skipped,
and `validate-references` labels the result `[emulated: ... -- latency is not native]`.

When the source dtype has no DPAS, prefer **fp16** over bf16 as the conversion target:
"upconversion sequences are typically faster" (`examples/cute/tutorial/xe_gemm.cpp:179-187`).

### Where Intel is actually tuning

oneDNN's `kernel.db`, counted by operand precision over rows native to Xe2:

| rows | operands |
| --- | --- |
| 24 | bf8 × f16 |
| 19 | **s4** × f16 |
| 15 | f16 × f16 |
| 9 | **nf4** × f16 |
| 6 | s8 × f16 |
| 5 | s4 × s8 |
| **0** | bf16 × bf16 |

There are no Xe2-native bf16 strategies at all; bf16 falls back to the XeHPC/PVC table
(`kernel_selector.cpp:289-295`). Intel's Xe2-specific GEMM tuning is overwhelmingly
**low-bit weights against f16 compute**, which is exactly what CUTLASS-SYCL's mixed-dtype
examples cover. That path is runnable on B580 today; native microscaling is not.

Consequence for picking a target: a bf16 GEMM here is running on tuning done for a
different part, and "oneDNN is at ~90% of peak so there is no headroom" was measured on
that inherited path.

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

Address constraints, which are the usual cause of a kernel that will not use block-2D:
base pointer **64-byte aligned**; width and pitch multiples of 4 bytes (16 in practice);
x-offset a multiple of 4 elements; width/height/pitch < 2²⁴.

Block-2D gives **free out-of-bounds zero-fill**, so remainder tiles need no load masking.
A shape that violates the constraints loses the whole mechanism — sycl-tla's own benchmark
file disables `seq_len_kv=77` for exactly this reason.

When block-2D is unavailable (gathered rows for GQA/MoE, misalignment), the fallback is SLM
staging with `XE_1D_LDSM` or `XE_1D_LOAD_GLOBAL`, using d16u32/d8u32 messages for narrow
types — materially slower, and worth restructuring the data to avoid.

Create the address payload once and update only X/Y per copy (`prepare_payloads`,
`copy_with_multi_payloads`, `update_payloads`); the FMHA mainloop does this for K and V.

## The mainloop is prefetch-staged, not double-buffered

There is **no SLM double-buffering in any production Xe mainloop**. If you are porting a
CUDA kernel built around `cp.async` into shared memory, that structure does not transfer.

The Xe pattern (`cutlass/gemm/collective/xe_mma.hpp:252-278`) is:

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

`Stages` is a **prefetch distance in K-tiles**, not a buffer count; 2 for GEMM and FMHA
prefill, 1 for decode, 3 for mixed-dtype. `SharedStorageSize` is 0 for the plain GEMM
kernel. SLM appears only in CuTe tutorials and in FMHA's cross-sub-group softmax reduction.

Reorders between copy and MMA fragments are the glue; when layouts already agree the
compiler removes them entirely.

## Tile shapes that Intel ships

All use sub-group 16 and, unless noted, an 8×4 sub-group layout = 32 sub-groups = 512
work-items per work-group.

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

Heuristics from `examples/cute/tutorial/xe_gemm.cpp:199-216`: K-tile is 2× the DPAS K
unless A is K-major or B is N-major byte data, in which case 1×; use a 4×8 sub-group layout
instead of 8×4 when B is narrower than A.

The MoE entry carries an explicit caveat from its authors: `256×128` beat `256×256` for
gpt-oss-20b prefill token distributions and "does not serve as an endorsement for all
possible input shapes." Treat every row here as a starting point to benchmark, not an
answer.

## Schedulers

Xe has exactly two: `KernelXe` (one output tile per WG, non-persistent) and
`KernelXeCooperative` (persistent, grid = `min(sm_count, tiles)`, driving `PersistentScheduler`
or `StreamKScheduler`). `KernelXePtrArrayCooperative` handles grouped/pointer-array.

**There is no Pingpong or warp-specialized schedule for Xe2.** Do not port one.

On Intel, "sm_count" is XeCore count — `gpu_slices × gpu_subslices_per_slice`.

Stream-K exists and self-disables: it assigns work to stream-K units only when there is
wave quantization *and* `k_tiles_per_output_tile > 8`, otherwise it runs data-parallel.
Reduction is deterministic (turnstile) by default.

## GRF mode

Kernels launch with `sub_group_size<16>`; BMG examples add `grf_size<256>` (CRI uses 512).
CUTLASS's generic `GemmUniversalAdapter` sets only the sub-group size, so for library GEMMs
the register mode comes from the environment. Intel's own CI — and their
`.github/copilot-instructions.md`, which says these "should be set locally for accurate
testing" — exports:

```bash
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
export IGC_VectorAliasBBThreshold=100000000000
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
```

For our own AOT SYCL solutions the equivalent is `FIB_SYCL_LARGE_GRF=1`, which `SyclBuilder`
turns into `-Xs -options -Xs -ze-opt-large-register-file` on the link step. That covers AOT
only; a SPIR-V JIT build takes its register mode from the runtime, i.e. from the env vars
above. **Measurements taken without either are small-GRF measurements** — say so, or set
them.

No CMake flag anywhere sets GRF mode, and no `reqd_work_group_size` /
`work_group_size_hint` attribute appears in sycl-tla at all.

## Adding to this file

A rule an agent would otherwise get wrong, with the file and line that establishes it.
Values that `Capabilities` can answer belong there instead; measurements belong in a trace.
