# Agent skills

A skill is procedure text an agent executes. Each `<skill>/SKILL.md` here carries a
`name` and a `description` in its frontmatter and is invoked as `/<name>`. The skills that
matter most fit together as one pipeline for optimizing the kernels a model runs; the rest
onboard models into the trace dataset or bring an Intel box up. This file is the map: what
to run, in what order, what each stage decides, and where the rules come from. It does not
restate any skill.

Two rules bind everything in this directory, and the scripts enforce them:

- **Nothing here is a stored measurement.** Every number a procedure needs is named by the
  command that measures it on this box, for this model, under this stack.
- **Targets are discovered, not named.** No skill takes a definition name, a model name or
  a kernel family as the thing to optimize; the model run says what it executes.

## The pipeline

Every stage is one script. Each writes an artifact the next stage reads, and each reads only
artifacts and calibration -- never a table kept in a skill.

| # | Stage | Command | Writes | Read by |
| --- | --- | --- | --- | --- |
| 0 | Calibrate the part | `python scripts/calibrate_part.py` | the calibration cache behind `flashinfer_bench.device.calibration.get()` | every gate in stages 4-6 |
| 1 | Discover what the model runs | `python scripts/harness_from_model.py --model <repo_id> --out-dir <dir>` | `<dir>/discovered.json` (ops, shapes, calls, device-time share, producer-consumer edges) and one verified harness per (op, shape) | 2, 3, 4 |
| 2 | Resolve each op to the kernel that implements it | `python scripts/pull_kernel_source.py --from-report <dir>/discovered.json --from-harnesses <dir> --bundle <pulled> --json <dir>/resolution.json` | `resolution.json` (class per op, source, schema, launched kernels) and one bundle per op under `<pulled>/` | 3, 5 |
| 3 | Fusions the model's own edges support | `python scripts/fusion_candidates.py --report <dir>/discovered.json` | a report, for the operator, of producer-consumer pairs that occurred, against the fusion tool's own preset list | nobody as a file: stage 4 loads the same module and recomputes the edge test from `discovered.json` for its `edge_present` and `epilogue_expressible` gates |
| 4 | Bound every candidate against every delivery mechanism | `python scripts/bound_candidates.py --report <dir>/discovered.json --resolution <dir>/resolution.json --out-dir <dir>/bound --measure --harness-dir <dir>` | `bound.json` (authoritative), `bound.log` (one line per gate evaluation), `worklist.json` (the ACCEPT rows, ordered by worth) | 5, 6 |
| 5 | Optimize, in the trial loop | `python scripts/kernel_trials.py init <series> <pulled>/<op>/harness.py` then `save` / `benchmark` / `finalize` | `tmp/kernel-trials/<series>.json`, the trial tree | 6 |
| 6 | Prove end to end | `python scripts/measure_serving_win.py --model <repo_id> --bound <dir>/bound --mechanism <m> --candidate <id>` | tokens/sec per arm, token digests, dispatch counters, a verdict | the decision to promote |
| 7 | Feed back | `python scripts/fill_serving_gaps.py --from-json <result.json>` on a stage-6 result written with `--json` | definitions for the shapes the stack asked for and got nothing for (`no-solution` counters); nothing is written without `--write` | 1, on the next run |

Stages 1, 2 and 6 import the serving stack and run under its interpreter; the rest run
under the dev venv (the last section says how to tell them apart). `<dir>` is one
discovery's output directory; the scripts default it to `tools/kernel-harness/auto`, which
is gitignored, as is `tmp/`.

What each stage decides:

- **Discover** decides the candidate set. A candidate is `(op, shape, dtype)` with the
  calls and the share of device time the model spent in it. The harness for each calls the
  same op the model called, imported from where it lives, and is discarded with a reason if
  it does not reproduce the call.
- **Resolve** decides the *class* of each candidate -- a provider kernel, a Triton kernel,
  a library primitive, an ATen kernel inside PyTorch, a Python-registered op, a
  decomposition -- by asking the dispatcher and, for GEMM-shaped work, by running the op
  under the library's own verbose output. The class is what admits or excludes each
  delivery mechanism in stage 4.
- **Bound** decides the worklist and its order. Nothing else does; see the next section.
- **Optimize** decides one candidate at a time, against the production kernel as baseline,
  never against a definition's PyTorch reference.
- **Prove** decides whether the change is worth anything to a served model, and halts on
  any gate that would make the number unreadable.

`/discover-model-kernels` walks stages 1-4 as a skill; `/wrap-kernel-for-tuning` covers
stage 5; `/measure-serving-win` covers stage 6. The module docstring of each script is
its reference: the flags, the artifact schema, and the contract it prints.

## Routing: why mechanisms are priced, not kernels chosen

The same kernel improvement is worth different amounts depending on how it reaches the
running stack. A change delivered through `apply()` intercepts every call, and pays a
per-call cost the calibration measures (`dispatch_us`); a patch to the provider's source, a
tuned Triton kernel at its JIT site, a change to how the library is called, or a fusion
at the call site pays none of it. So the question is never "which kernel" but "which
(candidate, mechanism) pair has a ceiling above what delivering it costs". Collapsing the
two turns "substituting this op does not pay" into "this op cannot be improved", which does
not follow.

`scripts/bound_candidates.py` does for delivery mechanisms what oneDNN does for its own
implementations under `ONEDNN_VERBOSE=dispatch`: every candidate is evaluated against
**every** mechanism, each gate in the chain writes one line, the first REJECT ends the
chain for that pair, and the worklist is whatever survived. Nothing is preferred. The
mechanisms, the fact about how the system is built that admits each, the cost each pays,
and the gate order are the tables in that script's module docstring; do not copy them --
the file is the source and the log names its line numbers.

A route that was not taken can therefore be seen to have been *priced*, not forgotten.
Four lines from one candidate's chain, as `bound.log` writes them:

```
fib_bound,v1,828abce9b752,46ed2981a4,aten.linear.default,4x2048/1024x2048,onednn,memory-bound-inefficient,provider_patch,class_admits,REJECT,class=onednn == admits=provider_kernel,bound_candidates.py:894
fib_bound,v1,828abce9b752,46ed2981a4,aten.linear.default,4x2048/1024x2048,onednn,memory-bound-inefficient,library_call,net_positive,PASS,ceiling_us=1.85666 > spread_us=0.21204,bound_candidates.py:1179
fib_bound,v1,828abce9b752,46ed2981a4,aten.linear.default,4x2048/1024x2048,onednn,memory-bound-inefficient,library_call,ACCEPT,ceiling_us=1.85666,worth=0.0129723,mechanism_us=0
fib_bound,v1,828abce9b752,46ed2981a4,aten.linear.default,4x2048/1024x2048,onednn,memory-bound-inefficient,apply_substitution,definition_exists,REJECT,definitions_matching=0 > required=0,bound_candidates.py:1058
```

How to read a line, field by field (thirteen, comma-separated, so `cut -d, -f<n>` works):

| Field | Meaning |
| --- | --- |
| 1-2 | `fib_bound,v1` -- the record type and its version |
| 3 | `run_id`: a digest of the `discovered.json` the routing was computed from. Every consumer recomputes it; re-running discovery voids the routing |
| 4-8 | `candidate_id`, op, shape, class (from stage 2), regime (from the measurements) |
| 9 | the mechanism being priced; `-` on an UNROUTABLE line |
| 10 | the gate; `ACCEPT` or `UNROUTABLE` on the summary lines |
| 11 | `PASS` or `REJECT` on a gate line |
| 12 | the arithmetic that decided, with **both sides evaluated**: `<lhs>=<value> <cmp> <rhs>=<value>`. On a REJECT this is what would have to move; `bound.json` carries the same as `needs` |
| 13 | `bound_candidates.py:<line>`, the gate's implementation -- as oneDNN names the source line of the gate that fired |

So the first line above says the provider-patch route is closed to this candidate because
its class is a library primitive, not a provider kernel; the second says the library-call
route clears the noise of its own harness; the third accepts that route with its ceiling
and worth; the fourth says substitution through `apply()` is closed because no definition
matches the op. Each rejection is one fact, one comparison, and one line of code.

What a consumer of these files may rely on (the full contract is in the module docstring):

- Every candidate that carries device-time share has at least one line.
- Every (candidate, mechanism) pair ends in exactly one ACCEPT or one REJECT; a candidate
  with no ACCEPT has one UNROUTABLE line after its last mechanism.
- The worklist is exactly the ACCEPT rows, ordered by `worth` descending, ties broken by
  `candidate_id` then mechanism. Nothing else orders it.
- An unmeasurable input is rendered as `None`, never as a number. A mechanism whose delivery
  cost is unmeasured is rejected at `cost_calibrated` with `dispatch_us=None`: unavailable,
  never free.
- `worklist.json` can be rebuilt from `bound.log` alone (`worklist_from_log`).

Downstream stages check the routing rather than trusting the caller: `kernel_trials.py
init <series> <harness> --bound <dir> --mechanism <m>` ties a series to its ACCEPT row and
every `benchmark` re-checks it; `measure_serving_win.py --bound <dir> --mechanism <m>
--candidate <id>` does the same for the serving A/B. Both refuse, before timing anything, a
pair the routing rejected, a routing whose discovery has since changed, or a mechanism
measured in the wrong mode. Without those flags the printed contract says
`ROUTING: UNCHECKED`.

## The skills

Grouped by what they are for. "Hands to" is the skill or artifact the procedure ends in.

### Kernel optimization (the pipeline above)

| Skill | Reach for it when | Hands to |
| --- | --- | --- |
| `/discover-model-kernels` | choosing anything to optimize; the entry point | the routed worklist; then `/wrap-kernel-for-tuning`, `/optimize-onednn`, `/find-kernel-gaps` per row's class |
| `/wrap-kernel-for-tuning` | the kernel lives inside a serving stack and must be tuned where it runs, or a harness came out of discovery | the trial loop; `/measure-serving-win` |
| `/optimize-onednn` | the resolver's class for a candidate is the library, or a profile says GEMM is slow: the axes a *caller* controls | `/measure-serving-win` |
| `/optimize-intel-kernels` | writing a SYCL or Triton kernel: for an existing definition, for a CUDA Triton solution being ported, or for an op the routing accepted as `authored_callsite` / `authored_apply` | `/measure-serving-win`; `/submit-onboarding-prs` for a solution that belongs in the dataset |
| `/find-kernel-gaps` | device time sits in ops no definition covers, or the model is not a plain transformer | `/extract-kernel-definitions` for a real kernel; `/optimize-ssm-scan` for a scan; `/measure-serving-win` |
| `/optimize-ssm-scan` | the gap is a state-space or SSD scan | `/extract-kernel-definitions`, `/optimize-intel-kernels` |
| `/measure-serving-win` | before claiming any win; converts a per-kernel ratio into a serving number, or proves there is none | the `no-solution` counters, which `measure-serving-win/references/fill-serving-gaps.md` turns back into stage 1 input |
| `/compare-implementations` | a gate needs to know whether an alternative implementation is competitive for an op class on this part, before search effort is spent on it | the calibrated input the routing reads; `/optimize-intel-kernels` to write the alternative |

### Shared references, linked from every optimization skill

These four files under `optimize-model-kernels/references/` are written once and cited by
name from the skills, so two agents reading the same profile reach the same row. They are
live procedure, unlike the plan documents in the next-but-one table.

| File | What it owns |
| --- | --- |
| `optimize-model-kernels/references/read-the-numbers.md` | the regime classification: its inputs and the command for each, the derived quantities, the rows and the comparison that selects each, and the field names to record. `scripts/bound_candidates.py` implements it, and `bound.json` carries the result per candidate |
| `optimize-model-kernels/references/mechanisms.md` | what each regime admits as a candidate change, as generators: the principle, the measurement that says it applies, and the measurement that says it worked. Nothing in it is ranked |
| `optimize-model-kernels/references/tools.md` | one entry per instrument: the question only it answers, how it is invoked, and the conditions under which its numbers can be trusted |
| `optimize-model-kernels/references/gates.md` | the key contract a trial prints, how a loop branches on it, every gate and what enforces it, the conditions that close a row, and who decides what |

Alongside those, each skill keeps references it alone owns. They are cited by name from
their skill and are not shared:

| File | What it owns |
| --- | --- |
| `optimize-model-kernels/references/deploy-provider-patch.md` | building a provider's kernels from source, selecting that build for one process, and the checks that prove the rebuilt kernel is the one running rather than the stock one |
| `optimize-onednn/references/strategy-selection.md` | reading which implementation the library chose and why, and what a caller can change to reach a different one |
| `optimize-onednn/references/illustrations.md` | worked findings for the library call, each naming the part, version and shape family it came from and the command that re-establishes it |
| `optimize-onednn/references/quantized-matmul.md` | the quantized-matmul path, including a correctness trap established by probing |
| `optimize-intel-kernels/references/xe-forge-knowledge.md` | an index into the vendor's kernel-optimization corpus: what to take from it, what to leave, and where its claims and this project's measurements disagree |

### Intel box: setup and profiling

| Skill | Reach for it when | Hands to |
| --- | --- | --- |
| `/setup-intel-env` | a new box, or any component missing; run before any Intel work | `/clone-repos`, then whichever skill brought you |
| `/profile-intel` | ranking kernel families by share of device time on a model, and routing each family | `/optimize-onednn`, `/optimize-intel-kernels`, `/wrap-kernel-for-tuning`, `/find-kernel-gaps` |
| `/onboard-model-intel` | "support this model on Intel", or an Intel deployment is slow or wrong | the per-phase skills it orchestrates, ending in `/submit-onboarding-prs` |

### Dataset onboarding (CUDA path; the trace dataset lives on HuggingFace)

| Skill | Reach for it when | Hands to |
| --- | --- | --- |
| `/clone-repos` | before any skill that reads upstream sources or the dataset clone under `tmp/` | the skill that needed the clone |
| `/onboard-model` | onboarding a new model end to end | the four phases below, through a run manifest |
| `/discover-models` | classifying a model new to the project and writing its kernel inventory | `/extract-kernel-definitions`, `/collect-workloads` |
| `/extract-kernel-definitions` | a definition JSON is needed; owns naming and axis rules | `/add-reference-tests` |
| `/add-reference-tests` | a definition needs its reference validated before a PR, or a reference is suspect | `/submit-onboarding-prs` |
| `/collect-workloads` | a definition has no workloads, or a benchmark reports `NO_WORKLOAD` | `/submit-onboarding-prs` |
| `/submit-onboarding-prs` | publishing: the dataset PR and the coverage-doc PR, behind the validation gate | `/track-models` |
| `/track-models` | refreshing `docs/model_coverage.mdx` | -- |

### Plan documents, deliberately without a `SKILL.md`

These are not routable and an agent must not treat them as procedure to execute:

| Document | What it is |
| --- | --- |
| `SKILLS-REWRITE-PLAN.md` | the design the skills are being rewritten to; its "Shared conventions" section is the source of the rules below, and its section on deriving the mechanism the way oneDNN derives an implementation is the specification `bound_candidates.py` implements |
| `optimize-model-kernels/PLAN.md` | the end-to-end pipeline skill, awaiting acceptance; becomes a `SKILL.md` when its open gaps are closed. Its `references/` directory is **not** a plan document -- those four files are live and are linked from the skills, as the table above says |
| `route-kernel-work/PLAN.md`, `route-kernel-work/RUN.md` | the router's design and its one-command run plan; stage 4 above is the part of it that now exists |
| `lint-baseline.json` | the recorded violation set the linter checks new edits against |

Agent definitions, when present, live in `.claude/agents/`, one markdown file per agent
with the model and tools it runs with in its frontmatter. Dispatching one starts a subagent
with that definition as its instructions and its own context, so a skill can hand a bounded
task -- authoring one kernel against one worklist row -- to a fresh agent and read back
only its report.
<!-- lint-skills: allow BROKENREF the agents directory is being created by work in flight; see the report -->

## The conventions, and why

Each rule exists so that a skill stores a measurement and a way to reason from it, rather
than a conclusion somebody reached once on one part. The linter rule that catches each is
in the last column.

| Convention | What it prevents | Rule |
| --- | --- | --- |
| No stored measurements: no microseconds, ratios, bandwidths, tile winners in procedure text | a number is a property of one part, one shape, one stack and one day; it drifts on the part it came from and is wrong on every other. Thresholds read from `calibration.get()` or from the run's own spread | `MEASURE` |
| No performance expectations: no sentence tells the agent what it will find | an expectation becomes the result; what may be stated without measuring is a fact about how the system is built | `EXPECT` |
| No closed lists of remedies; no remedy order | a numbered list of fixes encodes yesterday's profile as tomorrow's policy; the order is the output of a classification the agent performs in this run | `ENUM`, `ORDER` |
| No verdict columns | a routing table carries the measurement that selects the row, not the conclusion | `VERDICT` |
| No model, part, definition or library-version names in procedure text | they tie the procedure to one instance; commands discover them. Inside a marked illustration they are required | `NAME`, `ILLUS` |
| No machine paths | a home directory or a venv location belongs to one machine; a vendor default is allowed only beside what discovers it | `MACHINE` |
| Never the package-manager command that replaces this box's torch | it does so silently | `UVRUN` |
| Cross-references by name and heading, and only to things that exist | step numbers move; a link to a missing file is a procedure that cannot be followed | `STEPREF`, `BROKENREF` |

`scripts/lint_skills.py` enforces these mechanically. It carries a recorded baseline so the
rewrite can proceed one skill at a time; `--check-baseline` fails only on violations that
are not in it, and a finished skill re-records the baseline so its count goes to zero and
stays there.

```bash
source .venv/bin/activate
python scripts/lint_skills.py --check-baseline .claude/skills/lint-baseline.json   # the whole directory; exit 1 on a new violation
python scripts/lint_skills.py .claude/skills/README.md                             # one file, no baseline; this file must stay clean
python scripts/lint_skills.py --list-rules                                         # the rule table with the plan row each enforces
```

Exemptions are greppable markers, never intent: fenced code, an illustration block with the
required header, a fact table with a `Source:` line above it, or a line pragma with a
mandatory reason. The linter's module docstring lists them.

## Getting a number you can trust

**Sanctioned measurements.** `scripts/kernel_trials.py benchmark` (and `ab`, for two
builds of one symbol that cannot share a process) for kernels; `scripts/measure_serving_win.py`
for serving; `flashinfer-bench run` for solutions against a definition. The first two gate
on correctness before they time anything, warm every arm before timing any, alternate the
arms in interleaved rounds, and judge the difference against the arms' own paired scatter;
each prints a key contract -- one `KEY: value` per line, `VERDICT`, then `DONE` -- so a
driving agent reads keys, not sentences. `flashinfer-bench run` validates against the
definition's reference, warms before timing, and records the timer's name in the trace it
writes.

**Not sanctioned.** An ad-hoc timing script; a single run with no spread; a ratio against a
definition's PyTorch reference (`scripts/rank_vs_provider.py` computes the one that decides
deployment, against the provider kernel); a per-kernel ratio quoted as a serving result; any
number taken below the calibrated timing floor.

**Why the timer is the only path.** Timing methodologies are not interchangeable -- a
device-side trace and an event timer that includes launch overhead measure different
things -- so every timer in `flashinfer_bench/device/timer.py` reports its name, that name
is recorded with every latency, and the calibration is measured through the same path as
the gates that consume it. A number that reached you another way carries no statement of
how it was measured, and cannot be compared with one that did.

**What a failed gate does.** The stage halts and withholds the result it invalidates.
`measure_serving_win.py` prints no throughput table when an arm failed, the arms generated
different tokens, nothing was substituted, or the routing rejected the pair; the verdict
names the gate and the exit is non-zero. `kernel_trials.py finalize` refuses to copy a best
trial that is not a measured win. `calibration.get()` returns a record with `None` for any
quantity that did not settle, and does not cache it. A delta that survives a failed gate is
the line that gets quoted, so none is printed.

**Two interpreters, not interchangeable.** The dev venv (`source .venv/bin/activate`) runs
`flashinfer_bench`, the linter, calibration, bounding and the trial loop. Anything that
imports the serving stack -- discovery, resolution, the serving A/B -- runs under the venv
whose `python -c "import vllm"` succeeds. That venv is written down nowhere in the code: the
scripts run under whatever interpreter is active (`sys.executable`) and their children
inherit it. Find it as `CLAUDE.md` "Python Environments" says -- a venv already active in
the shell, else the venv directories beside this checkout -- and activate it rather than
naming its `python` by path, which leaves its `bin/` off `PATH`. A script started in the
wrong one says so and exits; a harness records the interpreter it was made under and refuses
to run where its op is not registered. Never `uv run` or `uv pip` in either.

Before any timed stage: the GPU must be idle apart from the desktop (`fuser -v
/dev/dri/renderD*`) and the power profile must read `performance` (`powerprofilesctl get`).
One GPU means one benchmark at a time.
