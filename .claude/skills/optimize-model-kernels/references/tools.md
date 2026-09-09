# Tools: the question each answers, and when its numbers can be trusted

An instrument earns its place by answering a question the others cannot. This file is that
index. Invocations are here in full only where no skill owns them; where one does, it is
named.

## The timer that produces a verdict

| | |
| --- | --- |
| **Question only it answers** | is this candidate faster than the thing the deployment would otherwise run, by more than the two arms' own scatter? |
| **Invocation** | `python scripts/kernel_trials.py benchmark <series> <candidate.py> --trial <id>`; `... ab <harness.py> --env-a K=V --env-b K=V` when two builds of one symbol cannot share a process |
| **Trust conditions** | it gates correctness before it times anything and prints no timing when correctness fails; it warms every arm before timing any; it interleaves the arms round by round and judges the paired ratio's own scatter. A number obtained any other way carries no statement of how it was measured |

`benchmark` prints one `KEY: value` per line and ends with `DONE`. The keys, and how a
driving loop reads them, are in `gates.md`. A series opened with `--bound <dir> --mechanism
<m>` re-checks its routing row on every `benchmark` and refuses a pair the routing rejected
before timing anything.

For a solution measured against a definition rather than a harness, the sanctioned command
is `flashinfer-bench run --local <dataset> --definitions <name> --save-results`: it
validates against the definition's reference, warms before timing, and records the timer's
name in the trace.

## unitrace -- what the device did

| | |
| --- | --- |
| **Question only it answers** | which kernels actually launched, their device-side duration free of the host path, the geometry they launched with, and their register spill |
| **Invocation** | `unitrace -d -v -o prof python <script>.py`. From Python, `flashinfer_bench.agents.unitrace` profiles a Solution on a Workload through the shared solution runner (`flashinfer_bench_run_unitrace`), and `find_unitrace()` resolves the binary through `FIB_UNITRACE`, then `PATH`, then a pti-gpu checkout under `tmp/` |
| **Trust conditions** | the script must end in a sync unitrace hooks -- `torch.xpu.current_stream().synchronize()`, an event sync, or a host read-back. Without one the report holds only a totals line. Hardware metric modes need a sysctl the owner sets |

Two uses that nothing else covers: reading `Spill Memory Per Thread` from the Kernel
Properties table for a kernel whose build log is not in view, and proving identity -- that
the kernel name the harness launched is the kernel name the model launched. `/profile-intel`
owns the full invocation table.

## VTune -- where a kernel's time went

| | |
| --- | --- |
| **Question only it answers** | the execution-unit active / stalled / idle split, achieved occupancy, the stall reason per instruction, and the instruction mix -- whether the wide loads and the matrix instructions were emitted at all |
| **Invocation** | `python -m flashinfer_bench.agents.vtune --check --device xpu:0` first; then `--mode characterization` or `--mode stall` against a `--harness` file |
| **Trust conditions** | optional, and its absence is a `--check` result taken on this box, never an assumption. Its counter modes are gated by a driver sysctl and its tracing mode by a Pin prerequisite; `--check` reports which modes the present state allows. A sampling mode needs the kernel to run long enough to be sampled, which is what `--harness ... --seconds` is for |

It enters the loop when the memory and compute rows of `read-the-numbers.md` both remain
open on the same kernel: the counter signature separates them where a timer cannot.
`/profile-intel` carries the prerequisite and failure tables.

## torch.profiler -- which op spent the time

| | |
| --- | --- |
| **Question only it answers** | the mapping from device time back to the framework op that launched it, and the operand shapes of that op (`record_shapes`) -- which is what makes a share per family, an edge between a producer and a consumer, and a bytes-moved estimate possible |
| **Invocation** | `python scripts/profile_intel.py --model <hf_repo_id> --device xpu:0`; `scripts/harness_from_model.py` uses the same activity to write `discovered.json` |
| **Trust conditions** | it needs no sync and no build, and it does not report spill, sub-group width or register mode. Its CPU-time column against its device-time column is what selects the `host/sync-bound` row |

## The library's own verbose output

| | |
| --- | --- |
| **Question only it answers** | which primitive implementation the library chose for this problem, the memory descriptors and attributes it was given, which candidate implementations it rejected and at which source line, and whether a primitive was created or found in cache on this call |
| **Invocation** | `ONEDNN_VERBOSE=1`, `=dispatch`, `=profile,exec`, `=debuginfo=5` on the repro; values compose |
| **Trust conditions** | the exec line's own time is the library's report, not the sanctioned timer's, and the two are not interchangeable in a verdict. Use the exec line to see *what ran and with what*, and `kernel_trials.py` to decide whether a change won |

`/optimize-onednn` owns the field glossary and the reading.

## Calibration -- the denominators

| | |
| --- | --- |
| **Question only it answers** | what this part costs: contiguous bandwidth, matrix throughput per dtype, the timing floor, the launch floor, the memory-channel period, what one `apply()` substitution costs, and what a kernel written here streams at |
| **Invocation** | `python scripts/calibrate_part.py`; from code, `flashinfer_bench.device.calibration.get()` and `authored_stream_probe()` |
| **Trust conditions** | measured once per part, timer and stack, cached on disk, and re-measured on a cache version bump. Any field can be `None`, which means unmeasurable on this box -- a gate reading `None` treats the mechanism as unavailable, never as free. Measure on an idle box under the performance profile, or the values are the contention's |

## Routing artefacts -- what was already decided

`scripts/bound_candidates.py` writes `bound.json`, `bound.log` and `worklist.json`: per
candidate, the regime and the arithmetic that selected it; per (candidate, mechanism), every
gate with both sides evaluated and a PASS or REJECT; and the accepted rows ordered by worth.
Reading a row costs nothing and answers questions the instruments above would have to be
re-run to answer. The skills `README.md` explains the log format field by field, and the
script's module docstring is the contract.

## Choosing between them

| The question | The instrument |
| --- | --- |
| where does this model's device time go, by op | torch.profiler, through `scripts/profile_intel.py` |
| is this harness calling the kernel the model called | unitrace, by kernel name |
| how long did the kernel itself take, without the call path | unitrace device timing |
| how much does the call path add | that against `kernel_trials.py` baseline arm |
| did my change win | `kernel_trials.py benchmark`, and nothing else |
| does the kernel spill | the build log's spill line, else unitrace Kernel Properties |
| why is a kernel short of its bound when the bytes and flops say otherwise | VTune counters |
| which primitive and descriptors the library used | the library's verbose output |
| what does this part cost | calibration |
| what has already been priced, and why a route was closed | `bound.log` |

## Sources

- `scripts/kernel_trials.py` -- the trial loop and its printed contract
- `flashinfer_bench/agents/unitrace.py`, `flashinfer_bench/agents/vtune.py` -- binary discovery, modes, and the prerequisites each mode needs
- `scripts/profile_intel.py`, `scripts/harness_from_model.py` -- the torch.profiler paths
- `flashinfer_bench/device/calibration.py` -- every measured denominator
- `/profile-intel` -- the profiler manuals in full
- `/optimize-onednn` -- the library verbose glossary
