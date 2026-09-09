# What a trial should know before it starts

Read this before the first trial. It is short on purpose: the authoritative material lives in
the skills and the kernel prompt, and duplicating it here would let the two drift apart. What
this file carries is the part a *search* needs — measured numbers that decide whether a
direction is worth a trial at all.

Everything below was measured on Intel Arc B580 (Xe2, device IP 20), 2026-09-08. Re-measure
on another part before relying on it.

## Where the authoritative guidance lives

| Question | Read |
| --- | --- |
| How do I write a correct, fast SYCL kernel here? | `flashinfer_bench.SYCL_PROMPT` |
| What is special about this Intel part? | `.claude/skills/optimize-intel-kernels/architectures.md` |
| What does the Xe matrix engine support? | `.claude/skills/optimize-intel-kernels/xe-matrix.md` |
| Is my GEMM slow because of how it is *called*? | `/optimize-onednn` |
| Is this win worth deploying at all? | `/measure-serving-win` |

## Numbers that decide whether a trial is worth running

**Substituting a kernel costs ~5.9us.** Resolve, key build, dtype check and invocation, per
call, and it does not shrink with the kernel. A kernel whose whole runtime is a few
microseconds cannot pay for its own replacement, however large the ratio: three times faster
than a 5us kernel saves 3.3us and costs 5.9us to obtain. Below ~10us, win by *removing* the
call (fusion), not by making it faster.

**The instrument has a floor if you time one call per region.** Event record plus an
unpipelined launch is ~40us here, which reported a 4.9us kernel as 40.8us and put every
decode-sized kernel on the same reading. `kernel_trials.py benchmark` batches calls into one
region for this reason; do not hand-roll around it.

**An idle GPU ramps its clocks.** Timing alternatives in sequence charges the ramp to
whichever runs first. This inverted a real comparison from 0.53x to 1.09x. The benchmark
warms both arms then interleaves rounds; a sweep that does not is not evidence.

**Peak bandwidth is not the ceiling — the access pattern is.** Read the same bytes in the
same shape the kernel is obliged to touch, and compare against that. vLLM's unified attention
sustains 164 GB/s against a ~456 GB/s peak, which reads as a 2.6x opportunity; a bare read of
one KV-head slice of its `[pages, page_size, kv_heads, head_dim]` cache sustains 167 GB/s.
Note the trap in *that* measurement too: a PyTorch strided read is itself only one
implementation, so it is a **lower bound on what is achievable, not a ceiling**. Acting on it
as a ceiling was wrong here -- a hand-written kernel later reached 1.35x over the same
baseline, taking it from 164 GB/s to 221 GB/s. Never conclude "already optimal" from one
implementation agreeing with another.

## Directions with a measured outcome

| Direction | Outcome |
| --- | --- |
| Launch-parameter sweep (block size, warps, stages, tile) on vLLM's unified attention | **1.00x** — vLLM's defaults already best; `TILE=64` 0.23x, 16 warps 0.43x |
| Writing a paged decode kernel from scratch, one program per (seq, kv_head) | **1.27x** over vLLM's unified attention |
| Adding split-K over the sequence with a combine pass, 4 chunks | **1.35x** — the best found in 9 trials |
| Tile and warp geometry on *our own* decode kernel | exhausted: TILE 16 → 1.10x, 64 → 0.71x, 8 warps → 0.48x, against TILE 32 / 4 warps |
| More split-K chunks | 2 → 1.29x, 4 → **1.35x**, 8 → 1.33x, 16 → 1.28x; the combine pass outgrows the parallelism |
| Fusing SwiGLU into a GEMM epilogue via oneDNN post-ops | **1.09x–1.31x** over vLLM's merged GEMM + fused activation |
| Splitting a merged gate/up projection into two GEMMs | **free** (0.96x of one merged GEMM at M=64) — not the compromise it looks like |
| Hand-writing a plain GEMM against oneDNN | do not; oneDNN is already the path `F.linear` takes |
| A `wait()` after every oneDNN submit | **2.21x becomes 0.44x** — the structure gets blamed for a per-call host block |

`num_warps=4` is **64 work-items** on Intel, not 128: Triton runs 16 threads per warp here, so
a warp count carried from an NVIDIA tuning guide describes a different workgroup.

## Correctness constraints that have actually bitten

- **Output buffers may alias inputs.** Destination-passing callers pass
  `output is hidden_states` deliberately. Safe only if each work-item reads every index it
  needs before writing any index another work-item might still read.
- **Match the definition's dtype exactly.** An fp16 kernel returning fp16 into a bf16 model
  kills the next matmul. bf16 has 7 mantissa bits; a dequantised 4-bit weight cannot spare
  three of them, so a bf16 path that "passes" at 88% of elements is not passing.
- **Accumulate in float32** even when inputs and outputs are half precision.
