# What a trial should know before it starts

Rules and procedure only. No measured values live here: a latency, bandwidth, ratio or
cost is a property of one part, one shape, one software stack and one run, and belongs in a
trace, a log or the calibration cache, which carry the hardware id, timer, shape and date.

## Where the numbers live

| What you need | Where it comes from |
| --- | --- |
| What a substitution costs, the timer's floor, achievable bandwidth | `flashinfer_bench.device.calibration.get()` — measured once per (part, timer) and cached; `scripts/calibrate_part.py` prints it |
| What a kernel written here streams at — the bound the routing prices an authored kernel to | `flashinfer_bench.device.calibration.authored_stream_probe()` — a sidecar of the same record; `scripts/calibrate_part.py` prints it |
| Whether a kernel beats the one a deployment runs | `scripts/rank_vs_provider.py` — computed from traces on this part |
| What a family costs in a real model | `/profile-intel` |
| What a change is worth end to end | `/measure-serving-win` |
| What a trial tried and what came of it | `tmp/kernel-trials/<name>.json` — the trial tree, with strategies |

Every threshold in the tooling reads from the first row. On new silicon, run the calibration
and the gates follow.

## Rules

- **Compete against the kernel a deployment would otherwise run**, not the definition's
  `reference`. The reference decides correctness; the provider baseline decides deployment.
- **A faster kernel is not automatically a win.** Substitution costs the same whatever the
  kernel costs; below a few multiples of that cost the win has to come from removing the
  call (fusion), not from speeding it. Read the cost from the calibration.
- **Peak bandwidth is not a kernel's ceiling; its access pattern is.** Read the same bytes in
  the shape the kernel is obliged to touch and compare against that. Treat that figure as a
  lower bound on what is achievable, not as proof the kernel is optimal.
- **Check which backend is live before choosing a target.** Share of device time says where
  the time is, not whose kernel spends it; a platform may reach a Triton kernel only as a
  fallback.
- **When tile and warp sweeps plateau, change the algorithm**, not the launch shape.
- **Query threads-per-warp; do not carry a warp count from another vendor.**

## Correctness constraints

- **Output buffers may alias inputs.** Destination-passing callers pass `output` aliased to
  an input deliberately. Safe only if each work-item reads every index it needs before
  writing any index another work-item might still read.
- **Match the definition's dtype exactly.** A narrower mantissa cannot carry a dequantised
  low-bit weight; a path that passes on most elements is not passing.
- **Accumulate in float32** even when inputs and outputs are half precision.

## Where the authoritative guidance lives

| Question | Read |
| --- | --- |
| How do I write a correct, fast SYCL kernel here? | `flashinfer_bench.SYCL_PROMPT` |
| What is special about this Intel part? | `.claude/skills/optimize-intel-kernels/architectures.md` |
| Is my GEMM slow because of how it is *called*? | `/optimize-onednn` |
| How do I harness and tune a kernel in place? | `/wrap-kernel-for-tuning` |
| Which regime is this kernel in, and what does that admit? | `.claude/skills/optimize-model-kernels/references/read-the-numbers.md`, then `mechanisms.md` beside it |
| What has a number got to clear before it counts? | `.claude/skills/optimize-model-kernels/references/gates.md`, which owns the rules on this page in full |
