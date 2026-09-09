# e95e9118e7-library_call

Row: candidate e95e9118e7, aten.linear.default, 4x2048/4096x2048, bf16, class onednn,
regime memory-bound-inefficient (regime_test `t_dev_us-t_mem_pattern_us=25.5127 > spread_us=0.35685`),
mechanism library_call, ceiling_us 25.512732248294668, worth 0.07409681262159201.

Baseline: streaming derivation of tools/kernel-harness/auto-m2/aten_linear_default_4x2048.py
(t_dev_source = profiler:harness-streaming), FIB_WEIGHT_POOL_MB=64 (l2_bytes = 18874368).
FIB_HARNESS_BASE = the routed harness. K = 4 (caller set none; default used).

| id | parent | lever / change | VERDICT | SPEEDUP | SPREAD_PCT | SPILLS | plateau |
|----|--------|----------------|---------|---------|------------|--------|---------|
| t0 | baseline | control: production aten.linear through the variant wrapper, nothing altered | NOISE | 0.998 | 0.5 | unknown | - |
| t1 | baseline | entry point: torch.matmul(x, w.t()) | NOISE | 0.999 | 0.4 | unknown | 1 |
| t2 | baseline | descriptor: weight as [K,N], wei tag ab | NOISE | 1.002 | 0.1 | unknown | 2 |
| t3 | baseline | orientation: transposed problem NxK:KxM into a ba-strided out view | NOISE | 0.996 | 0.4 | unknown | 3 |
| t4 | baseline | primitive lifetime: oneDNN called direct on PyTorch's queue, engine/stream/primitive cached, weight ba | NOISE | 1.001 | 0.2 | none | 4 |
| t5 | baseline | descriptor: weight offered as format_tag::any, reorder once and cached | WIN | 1.020 | 0.2 | none | reset |
| t6 | t5 | decomposition: split-N=2 | LOSS | 0.845 | 0.5 | unknown | 1 |
| t7 | t5 | decomposition: split-K=2 via bmm + sum | NOISE | 0.998 | 0.1 | unknown | 2 |
| t8 | t5 | selector input: M zero-padded 4 -> 16 | LOSS | 0.797 | 0.4 | unknown | 3 |

Control series e95e9118e7-library_call.control: baseline control.py (= t4, direct oneDNN with
the production ba weight, identical wrapper and build path), candidate t5.
  first run (default rounds/calls): BASELINE_US 72.13 / CANDIDATE_US 58.74, arm spreads 27.5 / 28.4 -- transient GPU contention, discarded
  --rounds 30 --calls 400:  BASELINE_US 41.24  CANDIDATE_US 40.59  SPEEDUP 1.016  SPREAD_PCT 0.1  VERDICT WIN

`moves=` for t5 confirmed: ONEDNN_VERBOSE wei descriptor goes from
`bf16::blocked:ba::f0` to `bf16:a:blocked:ab:4128x1:f0` (oneDNN's own padded pitch,
4128 elements = 8256 B/row), with one cached `reorder jit:ir` per weight.

Stopping condition: EXHAUSTED.
