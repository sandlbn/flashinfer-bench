# Worked findings, one instance each

Each block below is one measurement taken once, on one part, against one library build, for
one shape family. None of them is a rule: the same probe on another part, another library
version or another shape family can come out the other way, which is why each carries the
command that re-establishes it here today.

They are kept because they show what the probes in `../SKILL.md` look like when they land,
and because the arithmetic in each is worth copying even when the outcome is not.

The raw tables these were condensed from are in this file's history:
`git show 13baf92:.claude/skills/optimize-onednn/SKILL.md`.

## The weight's layout tag against the varying axis

**Illustration (one instance): Arc B580 / the oneAPI-bundled oneDNN of 2026-09, link version not recorded in the note / fp16 `[M, 896] x [896, 4864]` projections — re-establish with: `FIB_HARNESS_M=<M> python scripts/kernel_trials.py ab <a harness whose weight layout is set from its environment, after `tools/kernel-harness/trials/linear_row_pad.py`> --env-a <weight as [N, K]> --env-b <weight as [K, N]>`, once per M**

Same shape, same implementation, same device; only the weight's layout tag differed
(`wei:...:ba` for the `[N, K]` weight `F.linear` already holds, `ab` for
`x @ w.t().contiguous()`). Timed one process per cell, 100 warmup and 300 measured calls:

| M | `F.linear(x, w)` | `x @ w.t().contiguous()` | ratio |
| --- | --- | --- | --- |
| 1 | 0.0137 ms | 0.0231 ms | 0.59x |
| 64 | 0.0292 ms | 0.0224 ms | 1.30x |
| 512 | 0.0566 ms | 0.0557 ms | 1.02x |
| 2048 | 0.1904 ms | 0.1925 ms | 0.99x |

The direction reversed inside the sweep: the transposed weight won in a band around M=64
and lost at M=1, where the layout `F.linear` already has was 1.7x ahead. A blanket
load-time transpose would have slowed the case it was written for.

The measurement discipline mattered as much as the result: swept inside one process, the
M=64 cell read anywhere from 1.17x to 1.71x, because what the library had already
dispatched and cached changed what the next shape cost. One process per point.

End illustration.

## Removing a host wait from the call path

**Illustration (one instance): Arc B580 / a C++ oneDNN solution on the framework's queue, 2026-09 / a fused SwiGLU at m=1, 701 and 2801, and seven plain GEMM shapes — re-establish with: `python scripts/kernel_trials.py ab <harness that calls the solution built with the wait> --harness-b <the same solution built without it>`**

The same kernel, timed with and without a `ctx.stream.wait()` after `execute`:

| | m=1 | m=701 | m=2801 |
| --- | --- | --- | --- |
| fused SwiGLU against its reference | 0.44x to 2.21x | 1.01x to 3.02x | 2.03x to 3.11x |

Across seven plain GEMM shapes the same removal moved an explicit oneDNN call from
0.36-0.87x to 1.01-1.66x against `torch.matmul`. Before the wait was removed, the same
solution looked structurally slower than the vendor path at every shape.

End illustration.

## Split-matmul SwiGLU through post-ops, against the stack's own MLP

**Illustration (one instance): Arc B580, device-event timing, 2026-09-06 / a C++ oneDNN solution against vLLM-XPU's unfused MLP / a 0.5B-class model's layer-0 MLP swept over token count — re-establish with: `python scripts/measure_serving_win.py --model <repo_id> --env FIB_VLLM_MLP_FUSION=1 --env FIB_VLLM_MLP_MIN_TOKENS=<M>` per M, after the host wait above is gone**

Against the serving stack's real path -- norm, then the merged gate/up GEMM, then the
activation kernel, then the down GEMM -- the two-matmul post-op form measured:

| M | 512 | 1024 | 2048 | 4096 | 8192 |
| --- | --- | --- | --- | --- | --- |
| fused against the stack | 1.09x | 0.98x | 1.16x | 1.18x | 1.17x |

so the gate was set at the token count where the win became consistent for that model on
that part, and below it the two arms were within each other's scatter. The crossover is an
output of this sweep, not a number to carry: re-run it per part, per model, per library
build, and read the token histogram the adapter prints to know which M the scheduler
actually presents.

End illustration.

## A decomposition that the library cannot rescue

**Illustration (one instance): Arc B580 / oneDNN 3.11.x / block-scaled fp8 `M=512, N=4096, K=2560` — re-establish with: `tools/onednn/repro_grouped_scales.cpp`, then time the composed form against a single fused kernel**

Block scales along the reduction dimension can be composed exactly out of primitives the
library gets right -- one matmul per K-block with an expanded per-column weight scale, a
`binary_mul` post-op for the per-row activation scale, and a `sum` post-op to accumulate.
Timed, the composition spent essentially all of its time in the K-block matmuls
(0.70 ms of 0.778 ms), and each of those reads and writes the whole float32 destination:
8 MiB each way in 0.035 ms, about 460 GB/s, which was this part's memory bandwidth. The
composition was bandwidth-bound on accumulator traffic it created itself, and no attribute
inside the library removes traffic the composition itself adds.

At the same shape, a naive dequantise-and-matmul in the framework measured 0.67 ms and a
Triton block-scaled matmul with no tuned config for the device measured 3.24 ms.

The reading that generalizes is the arithmetic, not the numbers: when a composition's
`bytes_min` is dominated by an intermediate it materialises per step, the regime is
`memory-bound-inefficient` by construction and the candidate is one kernel that keeps the
accumulator in registers, not a better-tuned sequence of library calls.

End illustration.

## Sources

- `../SKILL.md` -- the probes each of these is an instance of
- `quantized-matmul.md` -- the correctness findings behind the last block
- `../../optimize-model-kernels/references/read-the-numbers.md` -- the classification each of these lands in
