# b79383ecf2 / provider_patch / _C.fused_add_rms_norm.default 5040x1024

row: candidate b79383ecf2, op `_C.fused_add_rms_norm.default`, shape 5040x1024/5040x1024/1024,
mechanism provider_patch, regime memory-bound-inefficient, ceiling_us 8.220689480924776,
regime_test `t_dev_us-t_mem_pattern_us=8.22069 > spread_us=0.20284`, native true.

baseline: tools/kernel-harness/trials/b79383ecf2-provider_patch/baseline.py
  (stream_harness derivation of tools/kernel-harness/auto-pf2/_C_fused_add_rms_norm_default_5040x1024.py)
pool var: FIB_WEIGHT_POOL_MB=64 (l2_bytes=18874368 = 18 MiB) for every command of the series.
K (plateau) = 4 -- caller set none.

Measurement mode: `kernel_trials.py ab` -- two builds of one `torch.ops` symbol cannot share a
process. Arms differ only by PYTHONPATH selecting the rebuilt provider overlay.
  build   : tmp/provider-builds/b79383ecf2/{configure.sh,build.sh}
  overlay : tmp/provider-builds/b79383ecf2/select
  results : tmp/provider-builds/b79383ecf2/ab-<trial>.json

| trial | parent | strategy | VERDICT | SPEEDUP | SPREAD_PCT | SPILLS | plateau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t0 | (baseline) | control: rebuilt-unpatched provider through the PYTHONPATH overlay | LOSS (A/A at the floor) | 0.999 | 0.1 | 0 (unitrace: Spill Memory Per Thread 0) | - |
| t1 | t0 | regime=memory-bound-inefficient; lever=remove the phase-2 re-read of the residual; change=fused_add_rms_norm_reg_kernel holds the summed slice in registers across the barrier; moves=profiler kernel name | LOSS | 0.985 | 0.2 | - | 1 |
| t2 | t0 | same kernel as t3, measured provider-vs-provider with `ab` | UNCOMPARABLE (tool limit: the op accumulates in place, so a non-bitwise kernel diverges the arms' post-loop input digest) | - | - | 0 (unitrace) | - |
| t3 | t0 | regime=memory-bound-inefficient; lever=reduction geometry; change=one row per 32-lane sub-group, 4 rows per 128-item work-group, no SLM, no barrier, row in registers; moves=unitrace kernel name + SLM Per Work Group 0 (was 4096) | LOSS | 0.919 | 0.2 | 0 (unitrace) | 2 |
| t4 | t0 | control: the vendor kernel through the variant wrapper, nothing altered | NOISE (0.997, floor 0.6%) | 0.997 | 0.3 | n/a | - |
| t5 | t0 | regime=occupancy/geometry; lever=per-lane access width and the thread count it implies; change=vendor kernel at vector_width 4 (8 B/lane, 256 work-items per row, 2x the threads) | NOISE | 0.997 | 0.5 | 0 | 3 |
| t6 | t0 | regime=memory; lever=which bytes reach DRAM; change=phase-1 input load annotated read_hint<streaming,L1,L2> | NOISE | 0.999 | 0.4 | 0 | 4 |
| t7 | t0 | regime=occupancy/geometry; lever=access width, other end; change=vendor kernel at vector_width 16 (32 B/lane, 64 work-items per row) | NOISE | 0.993 | 1.3 | 0 | 5 |
| t8 | t0 | regime=memory; lever=which bytes occupy L1; change=phase-2 input store annotated write_hint<streaming,L1>, L2 default | NOISE | 0.999 | 0.5 | 0 | 6 |
| t9 | t0 | regime=launch/occupancy; lever=work-group count held apart from thread count; change=two rows to a 256-item work-group, per-row SLM reduction slot (5040 -> 2520 work-groups, threads unchanged) | NOISE | 0.995 | 0.3 | 0 | 7 |
| t10 | t0 | regime=memory; lever=when a load is issued; change=phase-2 weight slice loaded before phase 1 and held across the barrier | LOSS | 0.909 | 0.9 | 0 | 8 |

## Close

PLATEAU. Eight consecutive non-wins from `best` (= the production kernel), four in each of two
regime rows:

* memory-bound-inefficient (bytes moved, where they live, when they are fetched): t1, t6, t8, t10
* occupancy / geometry / launch (thread count, access width, reduction shape, work-group count): t3, t5, t7, t9

`best` is the production kernel; no trial beat it. The control (t4) is 0.997 with a 0.6% floor,
so the wrapper and the rebuilt .so are neutral and every number above is the kernel.

### Why nothing could move it

The routed ceiling is `t_dev_us - t_mem_pattern_us = 104.597 - 96.376 = 8.22069 us`, where
`t_dev_us` came from `profiler:harness-streaming`. Measured by `kernel_trials benchmark`
(wall clock over 30 calls, which cannot under-count the device), the whole call at this shape
is **78.1 us**, and `bytes_min = 41289728` over 78.1 us is **528 GB/s** -- above this part's
calibrated `bandwidth_gbs = 428.42` and above its DRAM peak. The routing's own memory-pattern
bound for the kernel alone (96.376 us) exceeds the entire measured call by 18 us. Under the
routing's arithmetic with a wall-clock `t_dev`, this candidate is `below-the-bound`, and that
row admits nothing to author.

Corroboration: two profilers both over-report this kernel's device duration --
torch profiler 104.597 us and unitrace 123.857 us (streaming, `FIB_WEIGHT_POOL_MB=64`),
both larger than the 78.1 us wall clock for the same call.
