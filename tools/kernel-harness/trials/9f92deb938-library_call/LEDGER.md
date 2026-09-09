# 9f92deb938 / library_call / aten.linear.default 5040x1024/4096x1024

Row: candidate 9f92deb938, `aten.linear.default`, 5040x1024/4096x1024, bfloat16, class onednn,
regime compute-bound-inefficient (`regime_test: t_dev_us-t_cmp_us=79.3701 > spread_us=1.37703`),
mechanism library_call, ceiling_us 79.37008327108748, worth 0.023646831517624232,
share_pct 59.01 (resolution:launched). native true.
t_cmp_us 402.584 (flops 4.2279e10 / matmul_peak_tflops bf16 105.018), t_dev_us 481.954,
t_host_us 459.01, launch_floor_us 3.0076.

Baseline: streaming derivation (tools/kernel-harness/trials/stream_harness.py) of
tools/kernel-harness/auto-pf2/aten_linear_default_5040x1024.py, because
t_dev_source = profiler:harness-streaming. FIB_WEIGHT_POOL_MB=64 (l2_bytes = 18874368) and
FIB_HARNESS_BASE = this series' baseline.py for every command.

Note on which operand streams: at this shape the activation (5040x1024, 10.3 MB) is larger
than the weight (4096x1024, 8.4 MB), so stream_harness rotates the *activation* while
linear_call rotates the *weight*. t0 measures that difference and it is inside the noise
(0.993, floor 1.0%) -- as expected for a problem whose operands are 18.7 MB against 42.3
GFLOP.

Interpreter: .venv (torch 2.14.0+xpu), which is `measured_with` in bound.json; `init` did
not report a toolchain difference.
K = 4 (caller set none; default used). Budget: caller set none.

What oneDNN selects at this shape (ONEDNN_VERBOSE=all, dispatch lines only):
  matmul -> jit:gemm:any, src:bf16::blocked:ab::f0 wei:bf16::blocked:ba::f0 dst:bf16::blocked:ab::f0,
  problem 5040x1024:1024x4096, attr-scratchpad:user
  jit:xe_hp:gemm:any skipped at src/gpu/intel/gemm/jit_xe_hp_systolic.cpp:85
  ("skipping or dispatching to another implementation" -- a different gate from the
   decode-shape bundle, which reported line 79 / unsupported format tag)

| id | parent | lever / change | VERDICT | SPEEDUP | SPREAD_PCT | SPILLS | plateau |
|----|--------|----------------|---------|---------|------------|--------|---------|
| t0 | baseline | control: production aten.linear.default through the linear_call wrapper, nothing altered | NOISE | 0.993 | 0.5 | n/a (no code of mine) | - |
| t1 | baseline | operand descriptor: weight pre-transposed once to [K,N] contiguous, so oneDNN sees `wei ab` instead of `ba` | WIN | 1.016 | 0.3 | n/a | reset |
| t2 | baseline | problem orientation: transposed problem 4096x1024:1024x5040, dst ba, no copy | LOSS | 0.992 | 0.3 | n/a | 1 |
| t3 | baseline | operand descriptor: oneDNN direct, weight offered as format_tag::any, reordered once | WIN | 1.010 | 0.3 | none | reset |
| t4 | t1 | operand descriptor, leading dimension: t1's [K,N] weight with ld 4128 instead of 4096 (what oneDNN picks under `any`) | WIN | 1.016 | 0.2 | n/a | reset |
| t5 | baseline | control for t3 -- primitive lifetime and entry point only: oneDNN direct, weight descriptor unchanged (ba) | LOSS | 0.995 | 0.2 | none | - |
| t6 | t1 | destination lifetime: 41 MB result preallocated once and passed as out= | WIN | 1.016 | 0.1 | n/a | reset |
| t7 | t1 | problem decomposition (N): two 5040x1024:1024x2048 products, halves concatenated | LOSS | 0.525 | 0.2 | n/a | 1 |
| t8 | t1 | entry point: aten.mm.default instead of torch.matmul on t1's weight | WIN | 1.017 | 0.2 | n/a | reset -> best |
| t9 | t8 | problem decomposition (K): two 5040x512:512x4096 products summed | LOSS | 0.540 | 0.2 | n/a | 1 |
| t10 | t8 | entry point: torch.nn.functional.linear (what the serving stack's Python calls) | NOISE (at 22 rounds x 60 calls) | 0.995 | 0.3 | n/a | 2 |
| t11 | t8 | problem shape offered to the selector: M zero-padded 5040 -> 5120, a whole multiple of the kernel's 256-row M tile | LOSS | 0.820 | 0.2 | n/a | 3 |
| t12 | t8 | operand descriptor, last corner of the grid: torch.matmul on a transposed *view* (no copy), so the tag stays ba | NOISE (at 22 rounds x 60 calls) | 0.995 | 0.3 | n/a | 4 |

t3, t10, t12 were NOISE on the first block and re-run once at `--rounds 22 --calls 60` as the
branch table requires; t3 resolved to WIN, t10 and t12 stayed NOISE and are branched from as LOSS.
SPILLS n/a on every trial except t3/t5: those trials compile no code of mine (plain PyTorch call
shapes), so nothing can spill. t3 and t5 build SYCL+oneDNN and the block reported `none`.

## Close

PLATEAU. Four consecutive non-wins from `best` (t9 LOSS, t10 NOISE, t11 LOSS, t12 NOISE) across
the decomposition, entry-point and problem-shape levers, after the operand-descriptor lever had
been swept to both ends (ba / ab / any / ab+ld-pad / transposed view).

best = t8 at 1.017 (SPREAD_PCT 0.2, NOISE_FLOOR_PCT 0.3): the weight pre-transposed once to
[K, N] contiguous, issued through aten.mm.default.
Control comparison (series 9f92deb938-library_call.control, baseline = the production call
through the same wrapper): **WIN, 1.024**, control 481.09 us vs best 469.58 us. The win is the
call, not the wrapper.
Saving per call 8.15 us against the series baseline (477.85 -> 469.70), 11.51 us against the
control (481.09 -> 469.58); ceiling_us 79.37. One spread is 0.2% of 469.58 = 0.94 us, so the
ceiling is not reached and this does not close as CEILING.

### What the win is, and why the rest of the ceiling is not there

The observable that moves with t1/t8, read from ONEDNN_VERBOSE=all:

  production (wei ba):  kernel: gemm BB[SB] T@16N@16N 64 40 ... wg 4x8 sys xaf st k64 grf256 ...
  t1/t8    (wei ab):    kernel: gemm BB[SB] N@16N@16N 64 40 ... wg 4x8 sys xaf    k32 grf256 ...

and the same 4096-cube `torch.matmul` that `calibration.measure_matmul_peak_tflops` times to
produce `matmul_peak_tflops[bfloat16] = 105.018` selects **that same** `N@16N@16N ... k32`
kernel, because its B operand is already `ab`. So t1 is exactly the change that makes this call
run the kernel the peak was measured on.

Having made that change, the achieved rate is 42.2786 GFLOP / 469.58 us = **90.0 TFLOP/s**
(92.3 TFLOP/s at the 457.87 us of t1's block) against the 105.018 TFLOP/s of the probe -- 86-88%.
The probe is M=N=K=4096; this problem is K=1024, a quarter of the K the peak amortizes its tile
loads over. `t_cmp_us = flops / matmul_peak_tflops = 402.58 us` therefore prices this problem at
a K=4096 problem's throughput, and the remaining ~67 us of the 79.37 us ceiling is that
difference, not anything the caller holds.

unitrace on the selected kernel (both arms): `gemm_kernel[SIMD16 {20; 1; 1} {64; 8; 1}]`, AOT,
SIMD 16, SLM 0, private 0, **spill 0**, register file 256 -- a persistent kernel of one
work-group per Xe-core in large-GRF mode, fully occupied. The occupancy row named in
`regime_rows_skipped` ("occupancy-limited: no geom") is closed by that reading: nothing to gain.

### A hypothesis that overlaps a rejected mechanism

t4 (leading dimension padded 4096 -> 4128, +64 B) is a row-pitch pad, which is what
`layout_transform` does; that mechanism is REJECT for this candidate at
`layout_transform,class_admits,REJECT,layout_nominations=0 > required=0`. It measured 1.016, i.e.
exactly t1 and nothing more, so nothing is lost by not pursuing it. For the routing stage: the
gate reads `layout_nominations=0` and would have to reach 1 for that mechanism to be exercised
here -- and the evidence from t4 is that it would then find nothing, because oneDNN already
applies the same pad itself when the weight is offered as `format_tag::any` (od_any's descriptor
is `wei bf16:a:blocked:ab:4128x1`).
