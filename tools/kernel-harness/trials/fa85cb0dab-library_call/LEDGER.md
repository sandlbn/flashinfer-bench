# fa85cb0dab-library_call

Row: candidate fa85cb0dab, aten.linear.default, 4x1024/6144x1024, bf16, class onednn,
regime memory-bound-inefficient (regime_test `t_dev_us-t_mem_pattern_us=1.34629 > spread_us=0.27195`),
mechanism library_call, ceiling_us 1.3462884083207207, worth 0.009414666955735682.
bytes_min 12591104, bw_pattern_gbs 428.4227236082379, launch_floor_us 3.007608000189066.

Baseline: streaming derivation (tools/kernel-harness/trials/stream_harness.py) of
tools/kernel-harness/auto-run/aten_linear_default_4x1024_6144x1024.py, because
t_dev_source = profiler:harness-streaming. FIB_WEIGHT_POOL_MB=64 (l2_bytes = 18874368).
FIB_HARNESS_BASE = this series' baseline.py for every variant trial.

Interpreter: /home/sand/Projects/vllm-xpu-venv/bin/python (torch 2.13.0+xpu). Under
.venv (torch 2.14.0+xpu) `init` printed ROUTING_TOOLCHAIN: DIFFERS -- the routing was
measured under the vllm interpreter, so the whole series was run there and the ceiling
stays comparable.

K = 4 (caller set none; default used). Budget: caller set none.

| id | parent | lever / change | VERDICT | SPEEDUP | SPREAD_PCT | SPILLS | plateau |
|----|--------|----------------|---------|---------|------------|--------|---------|
| t0 | baseline | control: production aten.linear.default through the linear_call wrapper, nothing altered | NOISE | 1.004 | 0.4 | unknown | - |
| t1 | baseline | primitive lifetime + entry point: oneDNN matmul called direct on PyTorch's queue, engine/stream/primitive cached, weight descriptor unchanged (ba) | WIN | 1.007 | 0.2 | none | reset |
| t2 | t1 | operand descriptor: weight offered as format_tag::any, reorder once and cached | LOSS | 0.966 | 0.2 | none | 1 |
| t3 | baseline | operand descriptor: weight pre-transposed to [K,N] contiguous (wei tag ab) | LOSS | 0.971 | 0.7 | unknown | 2 |
| t4 | baseline | problem orientation: transposed problem 6144x1024:1024x4, dst ba, no copy | NOISE | 1.000 | 0.4 | unknown | 3 |
| t5 | baseline | entry point: aten.mm.default on a pre-transposed weight view | NOISE | 1.001 | 0.4 | unknown | 4 |
| t6 | baseline | problem decomposition: split-N=2 into two 4x1024:1024x3072 products | LOSS | 0.746 | 0.4 | unknown | (after 2nd regime row) |

t2, t4, t5: first block was NOISE; re-run once at --rounds 22 --calls 60 as the branch
table requires. t2 resolved to LOSS; t4 and t5 stayed NOISE and are branched from as LOSS.

SPILLS unknown on t0, t3, t4, t5, t6: those trials compile no code of mine (plain PyTorch
call shapes), so nothing can spill. t1/t2 build SYCL+oneDNN and reported `none`.
unitrace is not installed on this box (`which unitrace` empty, no /opt/unitrace), so the
Kernel Properties reading is unavailable here -- recorded as a gap, not a stop.

## moves= checks

- t1 (the only WIN) confirmed with ONEDNN_VERBOSE=all: the baseline issues, per call,
  `primitive,create:cache_hit,gpu:0,matmul,jit:gemm:any,...,attr-scratchpad:user` plus
  `primitive,exec:check,primitive,unused primitive execution argument (80)`; t1 issues
  neither and only `primitive,exec,gpu:0,matmul,jit:gemm:any,...` with no scratchpad attr.
  The observable moved.
- t2 confirmed: wei descriptor `bf16::blocked:ba::f0` -> `bf16:a:blocked:ab:6176x1:f0`
  with one cached `reorder jit:ir` per weight. The observable moved and the time got worse:
  the library's packed layout pads N 6144 -> 6176 (12352 B/row) and reads along N instead
  of along K, which costs 3.4% at this shape.
- t4 confirmed: problem string `4x1024:1024x6144` -> `6144x1024:1024x4`, dst tag ab -> ba.
  Observable moved, timing did not.

## Second regime row

The regime rows the routing skipped are `host-bound: no profiler CPU time in this run` and
`occupancy-limited: no geom`. torch.profiler on the baseline (200 calls): a single
`gemm_kernel` per call, 100% of Self XPU time, no reorder and no second launch; aten::mm
CPU total 40.2us/call against 50.9us XPU/call under the profiler, so the loop is
device-bound and the host-bound row does not apply. Geometry for the occupancy-limited row
needs unitrace, which is absent; the achieved rate stands in for it -- 12591104 B / 30.74us
= 409.6 GB/s = 95.6% of the calibrated bw_pattern_gbs 428.42, i.e. the whole ceiling is
that 4.4%, which is not an occupancy signature.

## Control

Series fa85cb0dab-library_call.control: baseline control.py (= t0, the production
aten.linear.default through the same variant wrapper), candidate t1.
  --rounds 22 --calls 60:   BASELINE_US 30.83  CANDIDATE_US 30.74  SPEEDUP 1.004  SPREAD_PCT 0.3  NOISE_FLOOR_PCT 0.5  VERDICT NOISE
  --rounds 44 --calls 120:  BASELINE_US 30.57  CANDIDATE_US 30.51  SPEEDUP 1.002  SPREAD_PCT 0.2  NOISE_FLOOR_PCT 0.4  VERDICT NOISE
t1's 1.007 against the streaming baseline does not survive against the control: the
kernel's own contribution is not distinguishable from zero.

## Not exercised (rejected mechanism)

`layout_transform` was REJECTED for this candidate at
`class_admits,REJECT,layout_nominations=0 > required=0`. The linear_call `pad` and `pitch`
variants are exactly the quantity that gate reads (row pitch moved off the memory-channel
period), so they were not run. For the row to be worth trying,
`layout_nominations` would have to move from 0 to >= 1 -- i.e.
`weight_layout.pad_rows_off_channel_period` would have to nominate this [6144,1024] bf16
weight (2048 B row pitch). Neighbouring evidence agrees it would not pay: lcs_qkv_proj t6
and t13 (4x1024/4096x1024) measured "row pitch +64B unconditionally" at 0.924 and 0.923.

`apply_substitution` was REJECTED at `net_positive,REJECT,ceiling_us=-4.57326 > spread_us=0.27195`
(dispatch_us 5.91955 exceeds headroom 1.34629), so no hand-written GEMM was drafted.

## Stopping condition

EXHAUSTED. Every lever the `library_call` mechanism holds was measured: operand
descriptors (t2 any, t3 ab), problem orientation (t4), attributes (the only one in play,
`attr-scratchpad:user`, is dropped by t1), primitive lifetime (t1), entry point (t5, t1),
problem decomposition (t6), synchronization (the loop has no host sync; the direct arm
issues no wait). Only t1 won and its win is not attributable against the control.
No finalize: `best` (t1) is a measured WIN in the tree, but the control comparison for it
was NOISE, so promoting it is not allowed.
