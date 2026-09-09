# Read the numbers: which regime a kernel is in

One classification, written once. Every skill that optimizes something links here rather
than restating it, so two agents reading the same profile reach the same row.

**What a regime decides:** which comparisons mean anything next, and which quantities are
still unknown. **What it does not decide:** the change to make. Candidates come from the
mechanism principles in `mechanisms.md`, derived per case; no row carries a remedy, and no
remedy order exists.

## The machinery does this first

`scripts/bound_candidates.py` derives the quantities below and applies the rows below
(`derive`, `classify`). For any candidate that has been through it, `bound.json` already
carries `regime`, `regime_test` (the comparison that selected the row, both sides
evaluated), and `regime_rows_skipped` (rows this run had no input for). Read those before
computing anything by hand.

Classify by hand only for a kernel the routing has not measured -- a solution against a
definition, a trial in a series opened without `--bound`, a repro built from a workload.
Use these field names when you record the result, so a hand classification and a routed one
can be compared.

## The inputs, and the command that produces each

| Symbol | What it is | Where it comes from |
| --- | --- | --- |
| `t_dev` | device time of the kernel, per call | unitrace `-d` Device Timing, average column: device-side, no host path (`tools.md`, "unitrace") |
| `t_host` | wall time per call through the call path the stack uses | `scripts/kernel_trials.py benchmark`, the baseline arm |
| `bytes_min` | bytes the operation is obliged to move: its inputs read once plus its outputs written once, at the dtypes the harness passes | the harness shapes (`harness.get_inputs()`); `bound_candidates.py --bytes-min <candidate_id>=<n>` overrides one the sizing gets wrong |
| `flops` | arithmetic the operation requires | the op's definition or its schema |
| `bw` | contiguous bandwidth this part reaches | `calibration.get().bandwidth_gbs` |
| `bw_pattern` | bandwidth of the access pattern the kernel is obliged to use | time a contiguous read of `bytes_min`, then the same bytes in the kernel's stride pattern, through the same timer; the routing does it for a candidate given `--pattern` |
| `authored_gbs` | what a kernel written on this box streams at, as distinct from the framework's own reduction | `calibration.authored_stream_probe()` |
| `peak` | matrix throughput this part reached at the harness dtype | `calibration.get().matmul_peak_tflops[<dtype>]` |
| `floor` | fixed cost one event-timed call carries; zero for a device-side duration, which has no region overhead | `calibration.get().timing_floor_us` |
| `launch` | time no kernel goes below, whatever it contains | `calibration.get().launch_floor_us` |
| `spread` | the candidate's own paired round-to-round scatter | `SPREAD_PCT` from `kernel_trials.py benchmark` |
| `spill` | register spill per thread | the compiler's `spilled around` line, or unitrace Kernel Properties `Spill Memory Per Thread` |
| `geom` | the global and local sizes actually launched | unitrace `-v` |
| `native` | whether the part runs the dtype natively rather than emulating it | `get_accelerator(dev).capabilities(dev).is_native_dtype()` |
| `occupancy`, stall mix, instruction mix | XVE active / stalled / idle, where stalls come from, whether vector loads and matrix instructions were emitted | VTune `gpu-hotspots`; `python -m flashinfer_bench.agents.vtune --check` says first whether it collects here |

Any of these can come back `None`. A `None` input removes the rows that read it; it never
becomes a default. A quantity nobody measured is unavailable, not zero.

## Derived

```
t_mem          = bytes_min / bw
t_mem_pattern  = bytes_min / bw_pattern
t_mem_authored = bytes_min / authored_gbs
t_cmp          = flops / peak
bound          = max(t_mem, t_cmp, floor, launch)          # over the terms that are not None
bound_pattern  = max(t_mem_pattern, t_cmp, floor, launch)  # what a kernel keeping this access pattern cannot go below
```

## The rows

Applied top to bottom. The order is a **validity** order, not a ranking of remedies: each
row is a precondition for the arithmetic below it. A number inside its own scatter
classifies nothing; a spilling kernel moves traffic `bytes_min` does not count, so `t_mem`
understates it; a host-bound call's `t_host` is not the kernel's time.

| Regime | Selected when | What the next step may read from it |
| --- | --- | --- |
| `unmeasured` | `t_dev` is `None`, or `spread` is `None` | nothing to classify: measure before reasoning |
| `unmeasurable` | `t_dev < floor` | the instrument, not the kernel, is being measured. Raise `--rounds` or `--calls`, or take the measurement at a larger problem size |
| `emulated` | `native` is False | the latency is the emulation's, not the format's. Do not report a ratio as a result about the dtype |
| `launch-bound` | `t_host - t_dev > t_dev`, or `t_dev <= launch` | changing what the kernel computes moves nothing; only removing launches can -- see the launch-bound row of `mechanisms.md` |
| `host/sync-bound` | profiler CPU time exceeds device time for the op, gaps between kernels in the timeline, primitive `create:` lines in steady state, or a host wait inside the call | the call path is what is being timed. Any kernel ratio taken here is an artefact of it |
| `spill-limited` | `spill > 0` | the memory and compute rows cannot be evaluated: spill traffic is outside `bytes_min`. Remove the spill, re-measure, classify again |
| `below-the-bound` | `bound - t_dev > spread` | the op did not move `bytes_min` in that time, so one of `t_dev`, `bytes_min` and the timer is measuring something else. Settle which before using any row below |
| `at-the-bound` | `abs(t_dev - bound) <= spread` | tuning geometry cannot move it. Only a change to the mathematics -- fewer bytes, fewer flops, one launch instead of two -- can, and which term forms `bound` says which |
| `memory-bound-layout-limited` | `abs(t_dev - t_mem_pattern) <= spread` and `t_mem_pattern - t_mem > spread` | the gap is the layout the data arrives in, not the kernel. The change is a load-time or serving-stack transform |
| `memory-bound-inefficient` | `t_mem > t_cmp` and `t_dev - t_mem_pattern > spread` | the memory row of `mechanisms.md` applies: access pattern, bytes in flight, cache reuse |
| `compute-bound-inefficient` | `t_cmp >= t_mem` and `t_dev - t_cmp > spread` | the compute row of `mechanisms.md` applies: matrix unit, tile against register budget, redundant arithmetic |
| `occupancy-limited` | `geom` leaves fewer threads resident than the part holds, or a sweep of the work-group size moves `t_dev` outside `spread` | the occupancy row of `mechanisms.md` applies |
| `unclassified` | no row matched | the inputs are inconsistent; say which are `None` and what would measure them |

Two rows need inputs a plain trial does not collect: `host/sync-bound` needs profiler CPU
time alongside device time, and `occupancy-limited` needs `geom` or a work-group sweep.
When a run has neither, the row is recorded as skipped -- `bound_candidates.py` writes them
into `regime_rows_skipped` -- and is never assumed to be false.

## Recording a classification

Whatever consumes it -- a trial's `--strategy` string, a note on a worklist row, a report --
carries the row name and the comparison that selected it with both sides evaluated, in the
form `regime=<row>; test=<lhs>=<value> <cmp> <rhs>=<value>`. A row name with no arithmetic
behind it is an opinion, and the next agent cannot check it.

## Sources

- `scripts/bound_candidates.py` -- `derive`, `classify`, and the per-candidate fields in `bound.json`
- `flashinfer_bench/device/calibration.py` -- every denominator above, measured per part and cached
- `scripts/kernel_trials.py` -- `SPREAD_PCT`, and the only sanctioned way to obtain `t_host`
- `mechanisms.md` -- what each row admits as a candidate change
- `tools.md` -- which instrument answers which of these questions
- `gates.md` -- what a candidate must clear before its number counts
