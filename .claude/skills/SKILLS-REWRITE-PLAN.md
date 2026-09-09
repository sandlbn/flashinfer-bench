# Plan: rewrite the agent skills as measurement-driven procedures

Status: accepted by the owner. The rewrites proceed in the order of §8; `scripts/lint_skills.py`
enforces §3.

Every claim below cites `file:line` in the current tree. No measured value appears in this
file; where a number is needed, the command that measures it is named.

## 1. The defect, stated once

The owner's example is `optimize-onednn/SKILL.md`. Its table at lines 11-18 lists
"Fix 1 … Fix 5" and its body (lines 103-155) is one section per fix. An agent reading it
tries the five, in order, and stops. The five are the answers one person found on one part
for a handful of shapes; the space of call-level changes to a library GEMM is open.

The same shape appears wherever a skill stores a *conclusion* where it should store a
*measurement and a way to reason from it*. Four variants, each found in the audit (§4):

| Variant | Recognise it by | Example |
| --- | --- | --- |
| Closed enumeration presented as the solution space | "Fix N", "the N levers", "in order of value" | `optimize-onednn/SKILL.md:11-18`, `optimize-intel-kernels/SKILL.md:105-113` |
| Fixed remedy order that is really one ranking measured once | "Try first / Then / Last resort" | `onboard-model-intel/SKILL.md:149-160` |
| Decision table keyed on a remembered verdict, not on a number the agent takes | a "Because" column full of conclusions | `profile-intel/SKILL.md:56-66`, `scripts/profile_intel.py:20-77` |
| Procedure that says what to conclude instead of what to measure | "X is usually where the time goes", "is the correct implementation on <part>" | `optimize-ssm-scan/SKILL.md:8`, `optimize-onednn/SKILL.md:63-66` |

Not every table is a defect. Tables of **facts about a tool, an API or the silicon**
(oneDNN verbose field order, the unitrace sync that is hooked, DPAS operand layout, a
provider's op schema) and **failure tables** (a tool's error → its fix) are closed because
their subject is closed. §3 says how to tell the two apart; §9 says where deleting them
would cost something.

## 2. Design target: what a rewritten optimization skill contains

Every skill that optimizes something follows one section order. The order is the loop the
agent runs; the sections are the inputs it needs at each step.

| # | Section | Contains | Must not contain |
| --- | --- | --- | --- |
| 1 | **Obtain** | The exact command for each data source; a glossary of every field of its output; the sync or env var without which the output is empty | prose about why the tool is good |
| 2 | **Read** | How to turn the fields into a regime classification (§2.1): the arithmetic, the comparisons, the thresholds — each threshold a value from `calibration.get()` or from the run's own spread | a verdict per op family |
| 3 | **Generate** | Mechanism-level principles per regime (§2.2), each with (a) the measurement that says it applies, (b) the measurement that says it worked. Framed as generators: "any change that reduces X" | a numbered list of remedies |
| 4 | **Tools** | For each tool: the question it answers that the others cannot; the invocation; which of its numbers are trustworthy and under what conditions | a tool listed without the question it answers |
| 5 | **Gates** | What a candidate must pass, in order, and how each is enforced by machinery rather than by the agent's judgement; how to tell a win from noise and from a call-path artefact | tolerances or margins as literals |
| 6 | **Illustrations** | Worked findings, each headed `Illustration (one instance):` with the part, library version and shape family it came from, and the command that re-establishes it today | an illustration that reads as the general rule |
| 7 | **Failure table** | tool/build/harness symptom → fix | performance verdicts |
| 8 | **Sources** | files and commands, by name | — |

### 2.1 Read: classifying the regime from numbers the agent just measured

This is the core the current skills lack. It is written **once**, in the pipeline skill's
`references/read-the-numbers.md` (§6), and every other skill links to it.

Inputs, all from the same run, all named by the command that produces them:

| Symbol | Meaning | From |
| --- | --- | --- |
| `t_dev` | device time of the kernel per call | unitrace `-d` Device Timing, avg column (device-side, no host path) |
| `t_host` | wall time per call through the call path the stack uses | `kernel_trials.py benchmark` baseline arm |
| `bytes_min` | bytes the operation is obliged to move: inputs read once + outputs written once, at the dtypes in the harness | computed from the harness shapes; `harness.get_inputs()` |
| `flops` | arithmetic the operation requires | computed from the op's definition |
| `bw` | achievable contiguous bandwidth on this part | `calibration.get().bandwidth_gbs` |
| `bw_pattern` | bandwidth of the access pattern the kernel is obliged to use | time a contiguous read of `bytes_min`, then the same bytes in the kernel's stride pattern, same timer (`optimize-intel-kernels/SKILL.md:229-235` already says this; it becomes a numbered probe) |
| `peak` | achieved matrix throughput on this part at the harness dtype | new calibration field (§7): a large square matmul under the sanctioned timer |
| `floor` | fixed cost one event-timed call carries | `calibration.get().timing_floor_us` |
| `spread` | run-to-run scatter of the candidate | `kernel_trials.py benchmark` output |
| `spill` | register spill per thread | unitrace Kernel Properties `Spill Memory Per Thread` |
| `geom` | global/local sizes actually launched | unitrace `-v` |
| `native` | whether the dtype runs natively | `caps.is_native_dtype()` |
| `occupancy`, `stall mix`, `inst mix` | EU active/stalled/idle; where stalls come from; whether vector loads and DPAS were emitted | VTune `gpu-hotspots` (resolved by `find_vtune()`: `FIB_VTUNE`, `PATH`, the oneAPI default; preflight must confirm it collects, §6 Stage 0) |

Derived: `t_mem = bytes_min / bw`, `t_mem_pattern = bytes_min / bw_pattern`,
`t_cmp = flops / peak`, `bound = max(t_mem, t_cmp, floor)`.

Classification, each row a comparison the agent performs. The rows are applied top to
bottom because each is a precondition for the arithmetic below it (a number inside spread
classifies nothing; a spilling kernel's `bytes_min` is wrong; a host-bound call's `t_host`
is not the kernel). The order says nothing about which remedy to try:

| Regime | Test | What it means for the next step |
| --- | --- | --- |
| Unmeasurable | `|t_host_candidate - t_host_baseline| < spread`, or `t_dev < floor` | raise rounds/calls or problem size before believing anything |
| Launch-bound | `t_dev` does not move across a sweep of the varying axis, or `t_host - t_dev` dominates `t_host` | no kernel change wins; only removing launches (fusion, batching, primitive caching, no host sync) can |
| Host/sync-bound | profiler CPU time > device time, gaps in the chrome timeline, `create:` lines in steady state, a wait inside the kernel | fix the call path first; a kernel ratio measured here is an artefact |
| At the bound | `t_dev ≈ bound` within spread | only an algorithmic change (fewer bytes, fewer flops, fusion) can win; tuning geometry cannot |
| Memory-bound, layout-limited | `t_dev ≈ t_mem_pattern` and `t_mem_pattern ≫ t_mem` | the gap is the data layout; a serving-stack or load-time transform, not a kernel |
| Memory-bound, inefficient | `t_mem > t_cmp` and `t_dev ≫ t_mem_pattern` | access pattern, bytes in flight, spill, occupancy — §2.2 memory row |
| Compute-bound, inefficient | `t_cmp ≥ t_mem` and `t_dev ≫ t_cmp` | matrix unit not used, tile vs registers, spill, emulated dtype — §2.2 compute row |
| Spill-limited | `spill > 0` | the memory and compute rows cannot be evaluated: spill traffic is not in `bytes_min`, so `t_mem` understates what the kernel moves. Remove the spill, re-measure, then classify |
| Occupancy-limited | `geom` gives fewer resident threads than the part can hold, or a work-group-size sweep moves `t_dev` | §2.2 occupancy row |
| Emulated | `native == False` | latency is not the format's; do not report a ratio as a dtype result |

### 2.2 Generate: mechanism principles that produce candidates

Each row is a generator, not a list. The agent derives a specific change from the principle
and the measurement; a change nobody wrote down is expected.

| Regime | Principles (each: what it changes → measurement that confirms it applied → measurement that confirms it won) | Illustrations that exist today, to be relabelled |
| --- | --- | --- |
| Memory-bound | fewer passes over memory (fuse producer/consumer, in-place, drop an intermediate, recompute instead of reload) → `bytes_min` falls → `t_dev` falls toward new `t_mem`. Better locality (contiguous per sub-group, block-2D eligibility, avoid strides that camp one channel, SLM staging for gathers) → `bw_pattern` rises. More bytes in flight (vector width from `caps.vector_width`, rows per work-item, prefetch distance) → `t_dev` approaches `t_mem_pattern`. Cache reuse (tile traversal order, working set vs `caps.l2_bytes`) | vectorized loads; multi-row work-groups for narrow rows (`architectures.md:56-64`); the strided-read probe |
| Compute-bound | use the matrix unit (dtype native, operand layout the unit wants, sub-group width it requires) → instruction mix shows it. Tile shape vs register budget → `spill == 0` and occupancy adequate. Reduce redundant flops. Precision only where `native`. Create parallelism when the output is small (split-K) → `geom` fills the part | `xe-matrix.md` DPAS/2D-block facts; tile table "starting points to benchmark" (`xe-matrix.md:131-149`) |
| Launch-bound | remove a launch (epilogue post-op, EVT fusion — the full route is §6.7, merged op), batch small ops, cache primitives, remove host syncs, graph capture if `caps.supports_graphs` → call count per step falls, `t_host - t_dev` falls | oneDNN post-ops (`optimize-onednn/SKILL.md:121-146`); Xe-Fuse presets (read live with `--list-presets`); primitive caching |
| Occupancy-limited | work-group size vs `max_work_group_size`; GRF mode trade (large GRF vs resident threads); rows per work-group; split-K | `FIB_SYCL_LARGE_GRF` note (`optimize-intel-kernels/SKILL.md:164-165`) |
| Library call (oneDNN or any provider op) | the axes a *caller* controls: operand descriptors (layout tag, strides, dtype); attributes (post-ops, scales, fpmath mode); primitive lifetime (cache hit vs `create:`); problem decomposition (split/merge/batch/k-split); synchronization (no host wait); which implementation/strategy the selector chose (`ONEDNN_VERBOSE=dispatch`, `debuginfo=5` consider/score lines) and whether a different one is reachable by changing a descriptor; library version behind the call → each confirmed by the `primitive,exec` line changing, then by `t_dev` | Fix 1-5 of the current skill become one illustration each |
| Wrong-level target | the op is a decomposition or a Python-registered op: optimize what it decomposes to, or tune in place | `discover-model-kernels/SKILL.md:69-76` |

## 3. Shared conventions — stated once, binding on every rewrite

| Rule | Detail |
| --- | --- |
| Section order | §2 table. A skill that is not an optimization skill (dataset, PR, setup) keeps its own order but obeys every rule below |
| No solution-space enumerations | No "Fix N", "lever N", "try A then B then C", "in order of value". A list is allowed only when its subject is closed (a tool's flags, a schema, a file layout) and the heading says what closes it |
| Ordering must be derived | A remedy order appears only as the output of a §2.1 classification performed in this run. No skill carries a default remedy order per regime, ranked list of remedies, or "start with X". The one fixed order is the row order of §2.1 itself, and it is a validity order, not a remedy ranking: a later row's arithmetic is undefined until the earlier row is cleared (no `spread`-clean number → nothing to classify; `spill > 0` → `bytes_min` undercounts the kernel's traffic, so `t_mem` is wrong; host/launch-bound → `t_host` is not the kernel) |
| No verdict columns | A routing or decision table carries the *measurement* that selects the row, not the conclusion. "Because" columns are deleted or become "Selected when" |
| No stored measurements | No microseconds, percentages, ratios, GB/s, tile winners. A threshold reads from `calibration.get()` or from the run's spread. `tools/kernel-harness/knowledge/README.md:3-5` already states this; it becomes global |
| Names outside illustrations | No model names, definition names, part names, library versions in procedure text. Placeholders `<…>` and commands that discover them. Inside an `Illustration (one instance):` block they are required |
| Illustration header | `Illustration (one instance): <part> / <library version> / <shape family> — re-establish with: <command>` |
| Every number has a command | Any quantity the procedure needs is named by the command that measures it on this box today |
| No performance expectations | No sentence tells the agent what it will find. A claim about performance appears only as a §2.1 comparison the agent performs, or inside an `Illustration` block. What may be stated without measurement is a **fact about how the system is built** (which code the stack already runs, what an API exposes, what a tool reports) — never what that code will cost |
| Environment | `source .venv/bin/activate` (dev); for serving, the venv whose `python -c "import vllm"` succeeds, found as `CLAUDE.md` "Python Environments" says — never a recorded path. Never `uv run`, `uv pip`, or `pip` with dependency resolution. No installs without the owner. Before any timed step: GPU idle (`fuser -v /dev/dri/renderD*`), `powerprofilesctl get` reads `performance` |
| One benchmark | `scripts/kernel_trials.py benchmark` for kernels; `flashinfer-bench run` for solutions against a definition; `scripts/measure_serving_win.py` for serving. No ad-hoc timing scripts |
| Cross-references | by skill name and section heading, never by step number (`f88dad6` already moved this way) |
| Tools are queried | device facts from `get_accelerator(dev).capabilities(dev)`; provider coverage from `find_baselines`/`REGISTRY`; Xe-Fuse presets from `--list-presets`; oneDNN versions from `onednn_link_version()/onednn_runtime_version()` |
| Length | `SKILL.md` carries the loop and the reading rules; manuals, hardware facts, and illustrations live in `references/` |
| Lint | `scripts/lint_skills.py` (§7) rejects: `Fix \d`, `Step \d` cross-references, digits followed by `us|ms|%|GB/s|x` outside fenced code and outside `Illustration` blocks, a blocklist of part/model names outside `Illustration` blocks, and any absolute path that belongs to one machine — a home directory, a clone or venv location, or a vendor default with nothing beside it naming how it is found (`MACHINE`; fenced code and illustrations do not exempt it) |

## 4. Every skill, one by one

Priority: **P0** rewrite first (owner's example and the loop); **P1** rewrite; **P2** sharpen
in place; **P3** light edits or leave.

| Skill | Current shape | Hardcoding / closed-enumeration evidence | Becomes | Priority |
| --- | --- | --- | --- | --- |
| `optimize-onednn` | 5 numbered fixes + verbose reading + decide table | `SKILL.md:11-18` fix table; `:8-9` "do not try to beat" as a rule; `:63-66` names a part and declares the correct implementation; `:103-113` M-dependent layout result as a rule; `:121-136` one fusion as "Fix 3"; `:180-188` decision table keyed on remembered outcomes; `:202-205` checklist ends with two questions; `references/strategy-catalog.md:1` "Fix 5" | §5.1: obtain/read/generate for a library GEMM call; the five fixes become five illustrations; strategy-catalog's observability (`debuginfo=5`) moves into Read | **P0** |
| `optimize-intel-kernels` | 8 steps; language decision table; ordered levers; failure table | `:65-72` language table keyed on remembered verdicts; `:74-76` "both SYCL and Triton" as policy; `:105-113` "Levers, in order of value"; `:120-135` Triton knobs as a closed list with "keep num_stages small" asserted; `:164-165` spill remedy as an either/or pair; `:184` "it will not find a fusion" about a sweep whose space is hardcoded (`scripts/optimize_model_kernels_xpu.py:44-49`); good: `:229-235` access-pattern probe, `:12-26` preflight, Step 6 | §5.2: the SYCL/Triton *authoring* skill: contract, build, spill check, then hand to the loop; language choice becomes a measured decision; levers become §2.2 links | **P0** |
| `profile-intel` | run one script; recoverable formula; family→skill routing table; two profilers | `:56-66` routing table with a "Because" column of conclusions; `scripts/profile_intel.py:20-77` regex→family→skill→advice with verdicts in advice strings (`:53`, `:60`); `:68` model-class claim; good: `:27-47` recoverable arithmetic, `:73-105` profiler facts | §5.3: profile → *classify each family by regime* (§2.1) → route by mechanism class; advice strings removed from the script | **P1** |
| `find-kernel-gaps` | 6 steps around one script | `:11` "most large gaps are contractions" as a rule; `:26-31` four rewrite signatures as the space (`scripts/find_kernel_gaps.py:21-45`); `:54` "small gap: stop" with no measured threshold; `:63` names a definition | §5.4: the general test (bytes the op moves vs bytes the maths needs; equivalence; ceiling vs spread) with the four signatures as illustrations; absorbs `optimize-ssm-scan` | **P1** |
| `optimize-ssm-scan` | one worked finding as a skill | `:3` names model families; `:8` verdict as opening line; `:34-54` "where a kernel can come from, in order" 1-3; `:36-37,45-50` names kernels and definitions; `:57-66` path-specific narration | Deleted as a skill. Its recognition rule (operand rank on the `aten` row) and its config-key mapping become `find-kernel-gaps/references/materialised-contractions.md` as an illustration | **P1** (merge) |
| `onboard-model-intel` | 7 phases; Phase 5 decision table; Phase 6 triage; escalation ladder | `:149-160` "Try first / Then / Last resort" per op family — the largest single instance; `:162-168` provider cost/risk (facts, keep); `:190-211` triage is a genuine 3-way measurement but its verdict lines are fixed; `:216-222` ladder (about where a change *ships*, keep); `:8-9` good framing | §5.5: Phase 5 becomes "probe coverage, measure every candidate baseline, hand to the loop"; Phase 6 keeps the measurement, its verdicts rewritten as §2.1 rows | **P1** |
| `discover-model-kernels` | 5 steps around three scripts | `:69-76` class→route table (mechanism table, acceptable; "Route" column names skills); `:116-121` "same dead end every time on this hardware"; `:166` wrong CLI (`init --harness`); `:170-176` "usually a constant" as first move; good: dispatcher-asking, edges, bound step | §5.6: stays; becomes Stages 1-3 of the pipeline skill by reference; the two verdicts deleted and re-expressed as facts about the mechanism and the bundle; CLI fixed | **P2** |
| `wrap-kernel-for-tuning` | contract + three checks + loop rules | `:104-110` loop policy is sound but "plateau: change the algorithm" has no measured trigger; `:113-118` good (calibration); no reading rules | §5.7: keeps the harness contract and the three checks; loop text moves to the pipeline skill; plateau gets a measured definition | **P2** |
| `measure-serving-win` | 6 steps; sitecustomize; counters; failure table | `:117-118` family verdicts ("GEMM, attention … not decode-sized norms") as a rule; `:131-133` "headroom is small by construction"; `:56-57` tolerance literals in the snippet; otherwise measurement-driven and the gate logic (`:86-103`, `:136-154`) is right | §5.8: sharpen; verdicts become the arithmetic they summarise plus an illustration; tolerance derived from dtype spacing; adopt the plain-arm and digest gaps (§7) | **P2** |
| `setup-intel-env` | stepwise checks with expected output | `:27`, `:45-48` `uv pip install` — violates the environment rule; `:19-20` IP→part literals (fact table; move to `architectures.md`); no VTune step though `find_vtune()` may resolve one; unitrace path not discoverable | §5.9: environment rule fixed; VTune and unitrace preflight rows with expected output; installation rows become "ask the owner" | **P2** |
| `clone-repos` | loop + optional Intel repos | `:110-112` `uv pip install`; `:96-97` "tuning oneDNN's own GEMM is not the goal" contradicts `strategy-catalog.md` | env rule fixed; contradiction removed; otherwise keep | **P3** |
| `optimize-model-kernels/PLAN.md` (draft, not accepted) | 7 stages, provenance, patch, promote | `:19-20,25,490` box-specific paths; `:26,312` asserts VTune absent — presence is a `find_vtune()` / `--check` result, not a plan fact; `:269-283` the loop is one 5-row table with "first hypotheses come from constants" — no reading rules, no mechanism generators; `:222-231` bound uses bytes/bandwidth only, no compute arm; good: `:294-309` control port, `:509-525` mechanical gates, `:527-541` who-decides, `:543-583` gaps 4-7 | Replaced by §6. Stages 5 and 7 (patch, prove identity, promote) become `references/deploy-provider-patch.md` | **P0** (superseded) |
| `route-kernel-work/PLAN.md`, `RUN.md` (drafts) | bound arithmetic; forbidden-list; 7-stage run | `RUN.md:26-33` fixed attempt per class; `PLAN.md:29-40` forbidden table is exactly right and is adopted into §3 | Merged into §6 Stage 3; directory deleted | **P0** (merged) |
| `onboard-model` (CUDA orchestrator) | thin orchestrator over per-phase skills | procedural; `:95` `pip install -e` acceptable on a CUDA box; no optimization content | unchanged except cross-reference spelling and the lint | **P3** |
| `discover-models` | config read; naming; who-supplies tables | `:87-100` op_type→FlashInfer path lookup (drifts; replace with the grep already at `:106`); `:112-119` asks the registry (good) | light edit | **P3** |
| `extract-kernel-definitions` | three paths; B2 naming table | B2 is normative by design and says so (`:165-168`); no optimization content | unchanged | **P3** |
| `collect-workloads` | SGLang collection procedure | `:38-45` flag table (facts about SGLang, keep); `:108-110` "fewer than 5 distinct values" — an unexplained threshold; state the reason (a sweep with one value cannot have exercised the varying axis) and let the count be a parameter | light edit | **P3** |
| `add-reference-tests` | generate/copy/tolerance/run | `:100-111` ground-truth lookup (keep; it already says "read FlashInfer's tests"); `:122-127` tolerance literals per dtype — replace with the derivation (a multiple of the dtype's relative spacing) so a new dtype is derivable | light edit | **P3** |
| `submit-onboarding-prs` | procedure + checklist | procedural; nothing to change under this critique | unchanged | **P3** |
| `track-models` | document format owner | procedural | unchanged | **P3** |

Reference files:

| File | Verdict | Change |
| --- | --- | --- |
| `optimize-intel-kernels/architectures.md` | Right shape (`:8-20` query, `:131-133` "no measured values") | keep; absorb the IP→part rows from `setup-intel-env`; "Traps" section links from the pipeline gates |
| `optimize-intel-kernels/xe-matrix.md` | Hardware facts with the source file for each; tile table framed as starting points (`:134`) | keep; add the sub-group/DPAS facts as the "compute-bound" measurement hooks in §2.2 |
| `optimize-intel-kernels/xe-fuse.md` | Tool manual + silent-garbage traps | keep as the build/layout manual; preset table (`:73-80`) replaced by "run `--list-presets`" (already what `fusion_candidates.py:47-66` does); add the trap rows from §6.7 (`tile_selector.py`, `run_kernel.sh` hardcoded device/GRF/sizes); the *route* — applicability, generation, wiring, measurement, gates — lives in `optimize-model-kernels/references/epilogue-fusion.md` (§6.7) and this file links to it |
| `optimize-intel-kernels/references/quantized-gemm.md` | Probed findings stated as rules | header becomes `Illustration (one instance)` with the re-probe command (`scripts/tune_vllm_fp8_config.py`) |
| `optimize-onednn/references/quantized-matmul.md` | Probed correctness finding (`:11`, `:15-23`) — valuable | keep, relabel with library version and the re-probe (`tools/onednn/repro_grouped_scales.cpp`) |
| `optimize-onednn/references/strategy-catalog.md` | "Fix 5" label; contents are an observability path | rename `strategy-selection.md`; its `debuginfo=5` reading moves into the skill's Read section; the rebuild procedure stays here |
| `onboard-model-intel/providers.md` | Inventories with "ask the registry" (`:9`) | keep; `:119-127` "what is built" table becomes "read `providers verify` output" |
| `onboard-model-intel/references/wiring-baselines.md` | procedure | keep |
| `measure-serving-win/references/fill-serving-gaps.md` | procedure | keep |
| `onboard-model/*.md` templates | PR/issue bodies | keep |
| `tools/kernel-harness/knowledge/README.md` | Already the right rules | becomes the pipeline skill's `references/gates.md` seed; the tools README links to the skill |

## 5. Rewrite details, per skill

### 5.1 `optimize-onednn` → "Tune a library GEMM call"

| Section | Stays | Replaced | Moves to `references/` | Deleted |
| --- | --- | --- | --- | --- |
| Front matter | trigger "a profile says GEMM/`F.linear` is slow on XPU" | description no longer lists four fixes | — | — |
| Obtain | repro (`:22-41`), `ONEDNN_VERBOSE` value table (`:47-52`), exec-line field glossary (`:54-61`), engine table note (`:68-69`), dispatch reason table (`:73-77`), source-line lookup (`:82-87`), knob table (`:93-98`), version check (`:157-178`) | add: `debuginfo=5` consider/score reading (from `strategy-catalog.md:10-24`); unitrace on the repro for `t_dev` and `geom`; the M-sweep (one process per shape, `:107-109`) as a numbered probe | — | `:63-66` (names a part, declares the correct implementation) |
| Read | — | new: compute `t_mem`/`t_cmp` for the shape from the exec-line problem field; classify per §2.1 — is `t_dev` within spread of `max(bytes_min/bw_pattern, flops/peak, floor)` (at the bound: only an algorithmic change can win), or above it (which term it exceeds says whether the headroom is in traffic or in the matrix unit), `create:` in steady state (host-bound), exec time ≈ `floor` (launch-bound). A fact stated without measurement: the library matmul is what `F.linear` already executes on this backend, so its `t_dev` is the baseline by construction and costs no trial | — | `:8-9` ("do not try to beat oneDNN's matmul") — a verdict; whether headroom exists is the `At the bound` row, per shape, per part |
| Generate | — | the library-call row of §2.2, spelled for oneDNN: descriptors, attributes, lifetime, decomposition, sync, implementation selection, version. Each with the exec-line change that confirms it applied | — | the "Fix 1-5" headings |
| Gates | version match before any comparison (`:157-171`) | add: one process per shape; `kernel_trials.py` as the timer; the dispatch-cost gate for anything delivered through `apply()` | — | — |
| Illustrations | — | — | `references/illustrations.md`: layout vs M (`:103-113`), primitive caching (`:115-119`), split-matmul SwiGLU post-ops (`:121-146`), host wait (`:148-155`), strategy-catalog entry | — |
| Decide | — | `:180-188` rewritten: rows selected by a measurement (`Selected when`), not by a remembered finding; the upstream-report recipe (`:190-192`) stays | — | — |
| Checklist | `:196-200` commands | followed by the §2.1 classification, not by two questions | — | `:202-205` |

### 5.2 `optimize-intel-kernels` → "Write a SYCL or Triton solution"

Scope narrows to authoring and building a solution; searching is the pipeline skill.

| Section | Stays | Replaced | Moves | Deleted |
| --- | --- | --- | --- | --- |
| Preflight | `:12-26` | — | — | — |
| Pick a target | `:49-63` validate-references table | `:28-47` becomes "come here from the pipeline worklist; without one, run `/profile-intel`" | quantized note → `references/quantized-gemm.md` link only | — |
| Choose the language | — | a measured decision: does a provider/library already own the op (`find_baselines`, `ONEDNN_VERBOSE`)? is the fusion expressible as post-ops? does a Triton solution exist to port? Each a command; the SYCL-vs-Triton choice stated as a trade (portability vs control over sub-group/vector width) with the same harness measuring both | — | `:65-72` verdict table; `:74-76` "both" policy unless `tests/integration/test_intree_kernels.py` still enforces it (state it as the test's rule, not a performance rule) |
| Write SYCL | `:80-103` contract, examples table | `:105-113` "levers in order of value" → link to §2.2 with the two device queries that apply (`caps.vector_width`, `preferred_sub_group_size`) | — | — |
| Write Triton | `:117-137` preamble facts | `:120-135` numbered knobs → "knobs the Intel backend exposes" (a closed fact list about the backend, labelled so) with the measurement for each (does the chosen config differ between small and large workloads?) | — | "keep num_stages small" as an instruction; becomes a mechanism note (no async-copy pipeline on this backend) with the measurement (sweep it) |
| Build and check | `:157-168` spill check | spill remedies: "reduce register pressure" as the principle (smaller tile, fewer live values, large GRF, lower unroll) each measured alone | — | the either/or pair as the space |
| Search | — | one line: `/optimize-model-kernels` Stage 4 | `:177-184` unattended sweep → note that its space is hardcoded (§7 item) | — |
| Benchmark, deploy, record | `:186-216` | — | — | — |
| Access-pattern probe | `:229-235` | becomes the `bw_pattern` probe referenced from §2.1 | — | — |
| Failure table, sources | keep | — | — | — |

### 5.3 `profile-intel` → "Profile, classify, route"

- Obtain: `scripts/profile_intel.py` output glossary; unitrace and torch.profiler facts
  (`:71-105`) stay; add VTune as the third tool with the question it answers (occupancy,
  stall attribution, instruction mix) and the preflight that proves it collects.
- Read: the recoverable formula (`:27-47`) stays. New: for each family in the top rows, run
  §2.1 on its dominant kernel (a harness from `harness_from_model.py`) and record the regime.
- Route: the table (`:56-66`) becomes keyed on **regime and class**, not family name:
  library-owned → `/optimize-onednn`; provider kernel or Triton → the pipeline loop;
  launch-bound edge → fusion candidates; materialised contraction → `/find-kernel-gaps`;
  data movement → layout. The "Because" column is deleted.
- `scripts/profile_intel.py`: `ROUTES` keeps the regex→family mapping (a fact about kernel
  names); the `advice` strings (`:24-28`, `:34`, `:40`, `:46`, `:53`, `:60`, `:67`, `:74`)
  are removed or reduced to the mechanism class. §7.

### 5.4 `find-kernel-gaps` (absorbs `optimize-ssm-scan`)

- Obtain: `scripts/find_kernel_gaps.py` output glossary (unchanged).
- Read: the general test replaces the signature table: for each hot `aten` row compare
  bytes moved (`record_shapes` operand sizes × calls) against the bytes the mathematics
  needs (inputs + outputs of the composite op); a large ratio is a materialisation whatever
  the op name. Then the equivalence check and the ceiling check (`:46-54`) with "small"
  defined as inside `spread` or below `floor`.
- Generate: express the composite as one call (`bmm`/`einsum`/library op), or fuse in
  SYCL/Triton keeping the intermediate in registers, or change the producer's layout.
- Illustrations: the four signatures (`:26-31`), the state-space scan
  (`optimize-ssm-scan/SKILL.md:12-32`, `:57-66`) in `references/materialised-contractions.md`
  with the config keys as placeholders and the definition/kernel names inside the block.
- Delete `:11`, `:63` from procedure text; `:76-86` stays.

### 5.5 `onboard-model-intel`

- Phases 0-4, 7 stay.
- Phase 5 "Decision table" (`:149-160`) is replaced by:
  1. Probe coverage: `find_baselines(definition)`, `REGISTRY`, `providers verify`,
     `ONEDNN_VERBOSE` on the reference — one table of *what exists*, generated per run.
  2. Measure every candidate baseline on the definition's workloads (`add-baselines`, `run`).
  3. Hand the winner as baseline to `/optimize-model-kernels`.
  The source cost/risk table (`:162-168`) stays as provider facts.
- Phase 6 keeps the three-suspects order (`:192-211`) and the 3-way timing; the verdict
  lines become §2.1 rows (at the bound / provider inefficient / needs fusion) with the
  comparison that selects each. The escalation ladder (`:216-222`) stays: it is about where
  a change ships.
- Source map (`:237-251`) updated to the new skill set.

### 5.6 `discover-model-kernels`

- Fix `:166` to `kernel_trials.py init <series> <harness.py>`.
- `:116-121` ("the same dead end every time on this hardware") is deleted. What it carried
  is re-expressed as a fact about the mechanism: a fused GEMM replaces the call the stack
  already makes, so it pays no `apply()` substitution cost; its value is the consumer's
  measured share (`fusion_candidates.py` output), computed per run.
- `:170-176` ("the fastest first move is usually a constant") is deleted. What it carried is
  re-expressed as a fact about the bundle: the source's constants (sub-group size, block
  size, vector width) are the knobs that exist; the device query gives the legal values;
  which value wins is a `kernel_trials.py` result. No order among knobs is stated.
- Add: the unitrace identity check per bundle (draft PLAN `:171-212`) as the gate that
  the harness calls what the model called.
- Steps 3-5 become links into the pipeline skill's Stages 3, 4, 6.

### 5.7 `wrap-kernel-for-tuning`

- Keeps: contract (`:14-40`), SYCL/oneDNN via `sycl_harness.py` (`:42-65`), where shapes
  come from (`:67-71`), the three checks (`:73-82`), failure table.
- Replaces `:84-118` with a link to the pipeline loop; "plateau" gets a definition: `K`
  consecutive children of `best` inside `spread`, `K` set at `init`.

### 5.8 `measure-serving-win`

- `:117-118`, `:131-133` become the arithmetic (`kernel_us` vs `cal.dispatch_us`; share ×
  (1 − 1/ratio)) with the family list moved into an illustration block.
- `:56-57` tolerance literals → derived from the definition's output dtype spacing; the
  sitecustomize snippet reads them from a helper (§7).
- Add the plain arm (no `apply()`, arms differ only in `--env`) and per-arm token digest
  once §7 lands; until then the skill says how to run the child arm by hand, as the draft
  PLAN `:464-469` does.

### 5.9 `setup-intel-env`

- `:27`, `:45-48`: replace with "torch-xpu is installed by the owner; verify with the
  import; do not install". Keep the never-`uv run` section (`:33-51`) as the rule.
- Add rows: unitrace binary location (`FIB_UNITRACE` or the pti-gpu build path, verified by
  `unitrace --version`); VTune (`vtune --version`; then a one-kernel `gpu-hotspots` collect
  whose summary must list a computing task — if it lists none, the driver/sysctl step is
  an owner action, recorded as unavailable, and every skill treats VTune as optional).
- `:19-20` IP→part literals move to `architectures.md`.

## 6. The new pipeline skill: `optimize-model-kernels`

One skill, the Xe-Forge shape: **reason from measurement → propose one change → build →
measure → feed back the number or the compiler error → branch on regression**, with every
gate enforced by machinery. Replaces the draft `PLAN.md` and both `route-kernel-work` files.

Files:

```
optimize-model-kernels/
├── SKILL.md                      the stages, the loop, who decides, stopping
└── references/
    ├── read-the-numbers.md       §2.1, the single owner; every skill links here
    ├── mechanisms.md             §2.2
    ├── tools.md                  unitrace / VTune / torch.profiler / ONEDNN_VERBOSE / trial loop:
    │                             question answered, invocation, trust conditions
    ├── gates.md                  from tools/kernel-harness/knowledge/README.md + draft PLAN §10,
    │                             plus the §6.6 key contract
    ├── epilogue-fusion.md        §6.7: the fusion route end to end, its traps, and the
    │                             hardcoded pieces of Xe-Forge that must not be imported
    └── deploy-provider-patch.md  draft PLAN §7-§9 (installation shape, overlay build,
                                  identity proofs, three-arm rebuild control, promote/revert)
```

### 6.1 Stages

| # | Stage | Command | Output | Gate to next |
| --- | --- | --- | --- | --- |
| 0 | Preflight | env activation, GPU idle, power profile, `scripts/calibrate_part.py`, `unitrace --version`, `vtune --version` + collect probe, `SyclBuilder.is_available()`, provenance record (draft `:85-104`) | calibration cache; tool availability table; provenance | every row prints its expected value; dispatch cost `None` means the `apply()` mechanism is *unavailable*, not free |
| 1 | Discover | `scripts/harness_from_model.py --model <id> --out-dir <dir>` at a decode-sized and a prefill-sized setting | `discovered.json` (`op_share`, `device_time_by_kernel`, `edges`, `triton`), verified harnesses | at least one harness verified; `device_time_total_us > 0` |
| 2 | Resolve | `scripts/pull_kernel_source.py --from-report … --from-harnesses … --bundle …`; unitrace on each bundle harness | class per op; source bundle; `t_dev`, `geom`, `spill` per harness | every op with share has a class; unitrace kernel name agrees with the class |
| 3 | Derive mechanism, bound, rank | new `scripts/bound_candidates.py` (§6.8, §7) over `discovered.json` + `fusion_candidates.py` edges + calibration + Stage 2 unitrace: for every candidate × every mechanism, evaluates the gates and writes one dispatch-style log line per gate; computes `ceiling_us = t_dev − max(t_mem_pattern, t_cmp, floor) − mechanism_us`, `worth = ceiling_us × calls / total` for the mechanisms that pass | `bound.log` (every candidate, every mechanism, every gate, PASS/REJECT with the arithmetic); worklist = ACCEPT rows ranked by `worth` | every candidate with share appears in the log; a candidate absent from the worklist has a REJECT line for every mechanism |
| 4 | Loop | `scripts/kernel_trials.py` per row, driven by the key contract (§6.6); each iteration per §6.2 | trial tree; `best`; control result | `VERDICT: WIN` against production and against the control port |
| 5 | Attribute and deploy | `references/deploy-provider-patch.md` where the class is a provider kernel or the stack's own source; `flashinfer-bench run --save-results` where a definition exists | identity proofs; trace or patch diff | proofs agree; rebuilt-unpatched control measured |
| 6 | Serving A/B | `scripts/measure_serving_win.py` (plain arm for source patches; overhead arm for `apply()`) | tokens/sec per arm, counters, digests | delta outside spread on a window that survives doubling; digests equal; identity holds |
| 7 | Feed back | `no-solution` counters → `scripts/fill_serving_gaps.py` → Stage 1 candidates; regressions at 100% substitution → Stage 1 at serving shapes | next worklist | — |

Stages 4-6 iterate; a failure at 5 or 6 returns to 4 with the failure as input, never
forward.

### 6.2 One iteration of Stage 4, written against the key contract

Every iteration consumes the keys the previous `benchmark` printed (§6.6) and produces one
`save` + one `benchmark`. The agent reads keys, never prose.

| Step | Input keys | Agent does | Machinery does | Exit key → next step |
| --- | --- | --- | --- | --- |
| Reason | the last block's `VERDICT`, `SPILLS`, `SPEEDUP`, `SPREAD_PCT` (+ `REASON` or diagnostics if failed); bundle `source/`, `PROVENANCE.md` schema; Stage 2 unitrace (`t_dev`, `geom`); `read-the-numbers.md` | classifies the regime (§2.1) from the numbers; writes one hypothesis as the `--strategy` string: `regime=<row>; principle=<§2.2 row>; change=<what>; moves=<which key or unitrace field>` | — | the strategy |
| Change | the `best` node's file (from `status`) | one change that tests the hypothesis | — | `<trial.py>` |
| Save | `<trial.py>`, `--parent <id>`, `--strategy` | — | appends a node to the tree | trial id |
| Build + measure | `benchmark <series> <trial.py> --trial <id>` | — | builds (or loads); reads spill from the build log; gates correctness; times both arms interleaved; prints the block | see the branch table below |
| Verify the mechanism | unitrace on the trial (`benchmark --unitrace`, §7) when `VERDICT: WIN` | checks the field the hypothesis named in `moves=` actually moved | records `t_dev`, `spill`, `geom` on the trial | moved → the win is attributed; did not move → the node is kept in the tree but the hypothesis is recorded as `unattributed`, and Reason runs again on this node before any child is made |
| Branch | `status` | — | prints `best`; a regression branches from `best` | per the branch table |

Branch table — the exit key decides, not a reading of the text:

| Exit key | Next `--parent` | Next Reason input | Notes |
| --- | --- | --- | --- |
| `VERDICT: BUILD_FAILED` | the same parent as the failed trial | the `--- build/load diagnostics ---` block, verbatim | a build failure is an input, not an exit; count it against the budget, not against `K` |
| `VERDICT: INCORRECT` | the failed trial itself | `REASON:` (which argument or output, max abs error, tolerance) | fix in place; no timing exists to reason from; never change `--atol/--rtol` |
| `VERDICT: NOISE` | unchanged | raise `--rounds`/`--calls` once and re-run `benchmark` on the same trial; if still `NOISE`, treat as `LOSS` | counts toward `K` |
| `VERDICT: LOSS` | `best` | `SPEEDUP`, `SPREAD_PCT`, `SPILLS` | counts toward `K` |
| `VERDICT: WIN`, `SPILLS: none` | this trial | `SPEEDUP`, `SPREAD_PCT` | resets `K` |
| `VERDICT: WIN`, `SPILLS: <n>` | this trial | `SPILLS` first: the next hypothesis must be `regime=spill-limited` (§2.1 makes the other rows unevaluable while spill is nonzero) | a spilling win is not final; the bound is unknown until spill is zero |

### 6.3 Who decides what

| Decision | Agent | Owner | Machinery |
| --- | --- | --- | --- |
| which row next | no — reads worklist order | may reorder with a stated reason | Stage 3 ranking |
| regime classification | yes, from the §2.1 table, recorded in the strategy | — | inputs come from tools |
| hypothesis and code change | **yes** | — | — |
| whether a number counts | no | no | `benchmark` (spread), unitrace (identity, mechanism) |
| tolerances | no | by changing the definition's dtype | derived from dtype spacing |
| skip the control port | no | no | Stage 4 gate |
| `K`, trial budget | reads them | sets at `init` | — |
| close a row | yes, when §6.4 fires, reason recorded | may close earlier | — |
| rebuild a provider (no install) | yes | — | overlay build |
| install, `sudo`, promote, switch the stack's branch | never | approves each | — |
| declare a deployment win | no | reads Stage 6 | `measure_serving_win.py` |

### 6.4 Stopping conditions for a row, in terms of the keys

| Condition (keys and tree state) | Verdict |
| --- | --- |
| `best` has `VERDICT: WIN` against production **and** the control-port trial (§6.3 of the draft, kept) has been benchmarked and `best.CANDIDATE_US < control.CANDIDATE_US` by more than `SPREAD_PCT` | proceed to Stage 5 |
| `best.t_dev` (unitrace) within `SPREAD_PCT` of `max(t_mem_pattern, t_cmp, floor)` | at the bound; proceed if `best` is also a `WIN`; otherwise the bound itself is the finding — record which term binds |
| `K` consecutive children of `best` with `VERDICT` in `{LOSS, NOISE}` | plateau: re-run Reason once with VTune data and a different §2.1 row; if the next `K` also plateau, close |
| trial budget spent (count of `save`d trials, `BUILD_FAILED` included) | close with `status` output |
| `N` consecutive `VERDICT: INCORRECT` with the same `REASON` (`N` set at `init`) | schema misread; re-read `PROVENANCE.md`; never loosen tolerance |
| `best` has `SPILLS: <n>` and every child that removed spill is `LOSS` | record that the spill-free form is slower on this shape with both numbers; the row stays open only if another §2.1 row is untried |
| Stage 3 `ceiling` for this mechanism falls below `SPREAD_PCT` after re-measurement | close as unroutable for this mechanism; take the next ACCEPT mechanism for the same candidate from `bound.log`, if any |

### 6.5 What the agent must never do inside the loop

Write a timing script; compare against the definition's reference instead of the harness;
loosen tolerance; carry a constant from another vendor without measuring; keep a win whose
predicted mechanism number did not move; run two GPU jobs at once; install anything; pass
`--tile auto` to the Xe-Fuse generator (§6.7).

### 6.6 The loop contract — what the machinery prints, verified against the code

The loop is driven by fixed keys, not by reading sentences. This is the interface
Xe-Forge's `tmp/Xe-Fuse/autotune/run_kernel.sh` established and `scripts/kernel_trials.py`
(`a64fa41`) reproduces. Both are documented here from the scripts, not from memory.

Xe-Forge's `run_kernel.sh` (`:10-13`, `:58-87`):

| Key | Source in the script | Meaning |
| --- | --- | --- |
| `BUILD: OK\|FAILED` | `:63-70`; on `FAILED` the whole `build.log` is printed and the script exits | compile result plus the compiler's own diagnostics |
| `SPILLS: <n>\|none` | `:71-76`, `grep "spilled around N"` over the build log | register spill as the compiler reported it, read for free on every build |
| `<kernel_name>: [<tflops>]TFlop/s (<time>)ms` | the binary's own print, three fixed problem sizes (`:80`) | throughput; no correctness key exists in this interface (`--verify=0`, `:83`) |
| `=== DONE ===` | `:87` | end of block |

`scripts/kernel_trials.py benchmark` (`_report`, `cmd_benchmark`, `:230-337`), in gate order:

| Key | Values | Printed when | Read it as |
| --- | --- | --- | --- |
| `BUILD:` | `OK` / `FAILED` | always | `FAILED` is followed by `--- build/load diagnostics ---` and the exception text verbatim (`:276-284`); the loop's next input |
| `SPILLS:` | `none` / `<n>` / `unknown` | always | matched by `_SPILL_RE` (`:217`): the compiler's `spilled around N` or unitrace's `Spill Memory Per Thread` |
| `CORRECT:` | `OK` / `FAILED` | after a successful build | on `FAILED`, `REASON:` follows and **no timing key is printed** (`:296-307`) |
| `REASON:` | text | only with `CORRECT: FAILED` | which argument or output differed, max abs error, tolerance |
| `MAX_ABS_ERROR:` | number | with `CORRECT: OK` | — |
| `BASELINE_US:` / `CANDIDATE_US:` | µs per call, medians of interleaved rounds | with `CORRECT: OK` | the harness (production kernel) vs the trial |
| `SPEEDUP:` | ratio | with `CORRECT: OK` | `BASELINE_US / CANDIDATE_US` |
| `SPREAD_PCT:` | percent | with `CORRECT: OK` | the candidate's own run-to-run scatter |
| `VERDICT:` | `WIN` / `LOSS` / `NOISE` / `INCORRECT` / `BUILD_FAILED` | always | `NOISE` when `|SPEEDUP − 1| < spread` (`:311`); the only field the branch table (§6.2) reads |
| `DONE` | — | always | end of block |

Properties the loop is written against:

| Property | Where enforced | Consequence for §6.2 |
| --- | --- | --- |
| a build or load failure is a block, not an exit | `:270-284` | `BUILD_FAILED` is the next Reason input; the tree records it |
| spill is read on every trial from the build log | `:217-227`, `:295` | no profiler run is needed to learn that a trial spills — **but see §7: `benchmark()` does not yet place the builder's log in `result["build_log"]`, so on the success path `SPILLS` is `none` regardless; until fixed, `--unitrace` supplies it** |
| a failed correctness gate prints no timing | `:296-307` | there is no number to be tempted by; `INCORRECT` branches on `REASON` alone |
| `NOISE` is a verdict, not a note | `:311` | an inside-spread result is neither kept nor discarded on its ratio |
| `finalize` copies `best` regardless of verdict | `:378-386` | until `--require-win` lands (§7) the operator refuses to run it unless `best` is a `WIN` |

Differences from Xe-Forge's script that the plan keeps deliberately: ours gates on
correctness and prints none of the timing without it; ours times the harness shape rather
than a fixed size list; ours does not export a register-file mode or an AOT device for
every trial (`run_kernel.sh:32-35`, `:48` hardcode both — see the trap table in §6.7).

### 6.7 Epilogue fusion as a full delivery route

A fusion folds an elementwise consumer into the producing GEMM's epilogue, so the fused
GEMM **replaces the call the stack already makes**. Delivered at the call site or by a
load-time bind it pays no `apply()` dispatch cost; delivered through the current adapter
(`flashinfer_bench/integration/vllm/adapters/mlp.py:223` calls `apply(...)`) it pays
`cal.dispatch_us` like any substitution. Stage 3 prices both (§6.8). Its ceiling is the
consumer's measured share plus the launch it removes, computed per run; nothing here says
what that comes to.

End to end, one row per stage:

| Stage | What | Command / file | Gate |
| --- | --- | --- | --- |
| Applicable? | the model ran the producer→consumer pair back to back (an edge, not two counts); the producer resolves to a GEMM class in Stage 2; no real op sits between them (a view is carried through, a `split`/`cat` is not, `harness_from_model.py:211-228`); the epilogue can express the consumer — as an oneDNN post-op (elementwise/binary on one GEMM's output) or as an Xe-Fuse EVT (adds pairwise lane ops) | `python scripts/fusion_candidates.py --report <dir>/discovered.json` — reads `edges`, matches consumers to presets read live from `generate_kernel.py --list-presets` (`fusion_candidates.py:47-66`) | an edge row with at least one preset, or a post-op expression; otherwise REJECT in `bound.log` with gate `edge_present` or `epilogue_expressible` |
| Which path | post-op (stays inside the library the stack already calls) vs EVT (a generated CUTLASS-SYCL kernel) — decided by expressibility, not preference: a lane-pairwise op is not a post-op | `ONEDNN_VERBOSE=1` on the producer to see the primitive the stack runs; the preset list for EVT | both paths that are expressible become separate candidates in Stage 3, each bounded |
| Generate | one translation unit per kernel, at the harness shape | `python tmp/Xe-Fuse/autotune/generate_kernel.py --preset <name> --m <M> --n <N> --k <K> --tile <MxNxK> -o <trial>.cpp` — `--tile` is **always explicit** and is a swept trial parameter | file written |
| Wrap | replace the generated `main` with a TVM-FFI entry on the framework's queue; take the model's weights through a one-time load transform (`flashinfer_bench/integration/weight_layout.py`: `interleave_gate_up`, `rms_row_scale`, `deinterleave_output`) | template `examples/sycl/xefuse_gemm_swiglu.cpp`; `xe-fuse.md` "Adapting a model's weights" | the wrapper takes the stack's tensors as stored (the adapter's docstring `mlp.py:52-56` records why a weight copy is refused) |
| Build | declare the `xe-fuse` dependency; `SyclBuilder` supplies `-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET`, include paths, and the `-spirv-ext` list (`xe-fuse.md:30-47`); `FIB_XE_FUSE_DIR`, `FIB_SYCL_TLA_DIR` set | `sycl_harness.build(<definition>, SOURCE, entry_point=..., dependencies=["xe-fuse"])` inside a trial file | `BUILD: OK`; `SPILLS` read |
| Establish the layout | the operand layout and the pairwise convention are not readable from the stride code (`xe-fuse.md:104-129`); test candidate references against the kernel's output — the right one agrees to the dtype's spacing, every wrong one disagrees by a large constant; there is no intermediate case | a one-off script per preset, kept beside the trial | `CORRECT: OK` against the established reference; a large error is a layout error, never a tolerance problem |
| Measure | baseline = the stack's own unfused sequence in one harness (the merged projection kernel it runs, then its activation kernel); trials = the fused kernel at each tile; a second sweep over `M` at the shapes the scheduler presents (`mlp.py:9-13`: with fusion enabled and no threshold, the adapter prints the histogram of token counts) | `kernel_trials.py init <series> <unfused_harness.py>`; `save`/`benchmark` per tile and per `M` | `VERDICT: WIN` per `M`; the crossover `M` is an **output** recorded per part, never carried |
| Floor check | two plain half-GEMMs with no epilogue, same shapes, same harness | one more trial | shows what the epilogue itself costs; if the fused kernel is not within spread of this floor the EVT, not the GEMM, is the target |
| Wire | (a) call-site patch in the stack (`deploy-provider-patch.md`: installation shape decides edit vs overlay), (b) load-time bind of the layer's forward, (c) the `apply()` adapter with `FIB_VLLM_MLP_FUSION=1 FIB_VLLM_MLP_MIN_TOKENS=<measured crossover>` | per path | dispatch counters show `applied` for (c); identity proof (unitrace kernel name inside the worker) for all three |
| Prove | serving A/B with the plain arm for (a)/(b), the overhead arm for (c) | `scripts/measure_serving_win.py` | §6.1 Stage 6 gate |
| Record | a Solution with `dependencies: ["xe-fuse"]` and a trace where a definition exists; otherwise the patch diff, the tile/`M` sweep table and the serving JSON | `flashinfer-bench run --save-results` | — |

Failure modes and traps, each with what to do:

| Failure / trap | Evidence | Action |
| --- | --- | --- |
| IntelLabs marks Xe-Fuse **not stable** | `xe-fuse.md:5-7` | a regression, a build break or a wrong result in Xe-Fuse loses one contender in Stage 3; it never blocks the pipeline. Other mechanisms for the same candidate proceed |
| `tile_selector.py` is a hardcoded tile table | `tmp/Xe-Fuse/autotune/tile_selector.py:5` — "Rules derived from sweeps on" a named part; `TILES` at `:14-46`; `select_tile` returns a fixed default at `:255` | **do not import it, and never pass `--tile auto`** (`generate_kernel.py:813-818` imports it on `auto`). It is the exact defect this plan removes — a ranking measured once on one part — inside the tool being borrowed. Tile shape is a swept trial parameter; the §6.6 contract makes the sweep cheap (one `save` + `benchmark` per tile, spill read for free) |
| `run_kernel.sh` hardcodes the AOT device and forces a register-file mode | `:48` (`-device` literal), `:32-35` (IGC and `SYCL_PROGRAM_COMPILE_OPTIONS` exports) | do not use `run_kernel.sh` as the build; build through `SyclBuilder`, which takes the target from the capability record; treat GRF mode as a trial parameter (`FIB_SYCL_LARGE_GRF`), measured alone |
| `run_kernel.sh` benchmarks three fixed problem sizes with `--verify=0` | `:80-83` | its numbers are not the harness shape and carry no correctness; use `kernel_trials.py benchmark` |
| compile wants `cuda_runtime_api.h` | build diagnostics | the two `-D` flags are missing; declare the `xe-fuse` dependency |
| output is garbage with no error | `CORRECT: FAILED` with a large `REASON` error | operand layout or interleaved-vs-split-half; re-derive with candidate references |
| right values, wrong positions | `CORRECT: FAILED`, values present | duplicated lanes: `deinterleave_output` |
| chained fused kernels no faster than unfused | `VERDICT: LOSS` on the chain, `WIN` on each alone | the compacting copy between them costs what the fusion saved (`xe-fuse.md:125-130`); resolve before wiring a second kernel |
| `SPILLS: <n>` at the chosen tile | contract | spill-limited row of §2.1: smaller tile or large GRF, each a separate trial |
| fused kernel wins in the harness, serving regresses | Stage 6 delta negative, `applied` nonzero | the scheduler's `M` distribution is not the harness `M`; take the histogram, re-run the `M` sweep, set the gate from it |

### 6.8 Stage 3 derives the mechanism the way oneDNN derives an implementation

oneDNN carries no table of which kernel to use. It evaluates candidates against the problem
descriptor and, with `ONEDNN_VERBOSE=dispatch`, prints why each candidate was rejected,
naming the gate and its source line. `scripts/bound_candidates.py` does the same for
delivery mechanisms: every candidate is evaluated against **every** mechanism, every gate
writes one line, and the worklist is what survives. Nothing is preferred; everything is
priced, and the pricing is inspectable.

Mechanisms and the facts that admit them (facts about how the system is built, from Stages
1-2; no performance claim):

| Mechanism | Admitted when (fact) | Delivery cost `mechanism_us` | Where the change lives |
| --- | --- | --- | --- |
| `provider_patch` | class = provider kernel; source bundled | 0 | provider source, rebuilt (`deploy-provider-patch.md`) |
| `triton_in_place` | class = Triton with `file:line` inside the stack | 0 | that file |
| `library_call` | class = oneDNN (a `primitive,exec` line) | 0 | the stack's call site or a load-time transform |
| `layout_transform` | regime = memory-bound, layout-limited | 0 | load-time weight/activation layout |
| `fusion_callsite` | edge present; producer class = GEMM; epilogue expressible (post-op or preset) | 0 | stack call site or module bind |
| `fusion_apply` | as above, plus a definition exists | `cal.dispatch_us` | dataset + adapter |
| `apply_substitution` | a definition exists or is authorable; a solution can be built | `cal.dispatch_us` | dataset |
| `source_rewrite` | bytes moved by the composite ≫ bytes the maths requires (`find-kernel-gaps`) | 0 | model source, or definition + python solution |
| `upstream_report` | class = ATen inside PyTorch, no local source | — | an issue; not a worklist row |

Gates, evaluated in this order for each (candidate, mechanism); the first REJECT ends the
chain for that pair, as in a dispatch list:

| Gate | Test | Inputs (command) |
| --- | --- | --- |
| `class_admits` | mechanism's admission fact holds | Stage 2 `pull_kernel_source.py` class |
| `edge_present` | (fusion only) the producer→consumer edge count ≥ the discovery threshold | `discovered.json.edges` |
| `epilogue_expressible` | (fusion only) a preset matches or the consumer is a post-op | `fusion_candidates.py`; `--list-presets` |
| `source_present` | (patch/Triton) the source is on this box | bundle `PROVENANCE.md` |
| `definition_exists` | (apply) a definition matches op, inputs, outputs | `TraceSet` lookup |
| `cost_calibrated` | (apply) `cal.dispatch_us` is not `None` | `calibration.get()` |
| `measurable` | `t_dev > floor` and Stage 2 unitrace matched the class | unitrace; `calibration.get().timing_floor_us` |
| `headroom` | `t_dev − max(t_mem_pattern, t_cmp, floor) > 0` | §2.1 inputs |
| `net_positive` | `ceiling_us = headroom − mechanism_us > spread_us` of the harness | `kernel_trials.py benchmark` on the harness against itself gives `spread_us` |
| `worth_cutoff` | `worth = ceiling_us × calls / total_us ≥ cutoff` (cutoff set by the operator at run time, default the noise floor of Stage 1's share estimate) | `discovered.json` |

Log format, one line per gate evaluation, comma-separated after oneDNN's own convention so
`grep` and `cut` work on it; written to `<out-dir>/bound.log` and mirrored as
`bound.json`:

```
fib_bound,v1,<run_id>,<candidate_id>,<op>,<shape>,<class>,<regime>,<mechanism>,<gate>,PASS|REJECT,<lhs>=<value> <cmp> <rhs>=<value>,<file>:<line>
fib_bound,v1,<run_id>,<candidate_id>,<op>,<shape>,<class>,<regime>,<mechanism>,ACCEPT,ceiling_us=<v>,worth=<v>,mechanism_us=<v>
fib_bound,v1,<run_id>,<candidate_id>,<op>,<shape>,<class>,<regime>,-,UNROUTABLE,<n> mechanisms evaluated,<n> rejected,-
```

Fields:

| Field | Content |
| --- | --- |
| `run_id` | the provenance record id (Stage 0), so a log is tied to a stack revision |
| `candidate_id` | stable hash of `(op, shape, dtype)` |
| `class`, `regime` | Stage 2 class; §2.1 row name |
| `mechanism`, `gate` | names from the two tables above; `-` on the UNROUTABLE line |
| `PASS`/`REJECT`/`ACCEPT`/`UNROUTABLE` | REJECT ends the pair; ACCEPT is written after the last PASS; UNROUTABLE after the last mechanism when none ACCEPTed |
| `<lhs>=<value> <cmp> <rhs>=<value>` | the arithmetic that decided, with both sides evaluated, e.g. `headroom_us=<v> > 0`, `ceiling_us=<v> > spread_us=<v>`, `dispatch_us=None` |
| `<file>:<line>` | the line in `bound_candidates.py` implementing the gate — as oneDNN names its gate's source line |

Rules the log makes checkable:

- Every candidate with share in `discovered.json` has at least one line; a candidate
  missing from the worklist has a REJECT line for each mechanism and one UNROUTABLE line.
- The worklist is exactly the ACCEPT lines, sorted by `worth`; no other input orders it.
- A mechanism rejected on `cost_calibrated` reads `dispatch_us=None`, never `0`: the
  mechanism is unavailable, not free.
- Fusion appears as two mechanisms with two costs; the operator can see which one the
  arithmetic admitted.
- `bound.json` carries the same rows so Stage 4 can open a series from an ACCEPT row and
  Stage 6 can quote the ceiling that motivated it.

## 7. Machinery changes the rewrite depends on (proposals, owner-approved, not done)

| Item | Where | Why the skill text cannot do without it |
| --- | --- | --- |
| `finalize --require-win` (default on) | `scripts/kernel_trials.py:378-386` (verified still absent after `a64fa41`) | today it copies `best` even when `best` is a `LOSS` |
| capture the builder's log into `result["build_log"]` | `scripts/kernel_trials.py` `benchmark()` (`:130-214`) and `:295` | `_spill_from(result.get("build_log", ""))` reads a key nothing writes; `SPILLS` is `none` on every successful build until the SYCL builder's stderr (and Triton's compile log) are captured — the free spill read the contract promises does not yet exist |
| `benchmark --unitrace` | `scripts/kernel_trials.py` | attaches `t_dev`, `spill`, `geom` to each trial so "verify the mechanism" is mechanical, and supplies `SPILLS` until the row above lands |
| `ab` subcommand | `scripts/kernel_trials.py` | two builds of one `torch.ops` symbol cannot coexist in one process (draft `:561-564`) |
| `scripts/bound_candidates.py` with `bound.log`/`bound.json` | new; format in §6.8 | Stage 3: every candidate × every mechanism × every gate, dispatch-style; today the arithmetic lives only in prose and a rejected route is indistinguishable from a forgotten one |
| `scripts/xefuse_trial.py` | new | emits a trial file for `(preset, M, N, K, tile)` — generator call with explicit `--tile`, TVM-FFI wrap from the example, `sycl_harness.build(..., dependencies=["xe-fuse"])` — so a tile sweep is one `save` + `benchmark` per tile; refuses `--tile auto` |
| call-site / load-time delivery for the fused MLP | `flashinfer_bench/integration/vllm/adapters/mlp.py` (today: `apply(...)` at `:223`) | the `fusion_callsite` mechanism (§6.8) has no delivery path in the integration; add a bind-at-load mode so a fusion can be A/B'd without `apply()` in the path |
| calibration: `matmul_peak_tflops`, `launch_floor_us`, strided-read helper | `flashinfer_bench/device/calibration.py`, `scripts/calibrate_part.py` | `t_cmp` and `bw_pattern` need measured denominators |
| `--plain-arm`, per-arm token digest | `scripts/measure_serving_win.py` | a provider/source patch A/B must not carry `apply()` in the path (draft `:566-569`) |
| tolerance helper from dtype spacing | `flashinfer_bench/apply/config.py` or a small util | removes tolerance literals from three skills |
| strip `advice` verdicts from `ROUTES` | `scripts/profile_intel.py:20-77` | the script currently prints conclusions |
| derive sweep space from the device | `scripts/optimize_model_kernels_xpu.py:44-49` | `WORK_GROUP_SIZES`, `SUB_GROUP_SIZES` are literals; read `max_work_group_size` and `sub_group_sizes` |
| fix `init --harness` | `scripts/pull_kernel_source.py:327`, `discover-model-kernels/SKILL.md:166` | wrong CLI |
| `scripts/lint_skills.py` | new | enforces §3 mechanically after every rewrite |
| `FIB_UNITRACE` env var or path discovery | `flashinfer_bench/agents/unitrace.py:61` | a build need not be on `PATH`; the variable names it |

## 8. Order of work, and what is verifiable after each step

| Step | Work | Verifiable by |
| --- | --- | --- |
| 1 | `scripts/lint_skills.py`; run it on the current tree to record the baseline count of violations | lint output lists every violation §4 cites |
| 2 | `optimize-model-kernels/references/{read-the-numbers,mechanisms,tools,gates}.md` (from §2, §3, knowledge README, draft §10) | lint clean; every other skill can link to them |
| 3 | `optimize-onednn` rewrite (§5.1) | lint clean; a dry read produces a hypothesis for a shape by classification, not by fix number; `references/illustrations.md` holds all five former fixes |
| 4 | Machinery items: `bound_candidates.py` + `bound.log`, `build_log` capture, calibration fields, `finalize --require-win`, `benchmark --unitrace`, CLI fix, `FIB_UNITRACE` | unit tests; `calibrate_part.py` prints the new fields; Stage 0-3 run on one model without a long GPU session; `bound.log` has a line for every candidate with share and `grep REJECT` explains every absence from the worklist; a deliberately spilling SYCL trial prints `SPILLS: <n>` |
| 5 | `optimize-model-kernels/SKILL.md` (§6) + `references/{read-the-numbers,mechanisms,tools,gates,deploy-provider-patch,epilogue-fusion}.md`; delete `route-kernel-work/`, replace the draft `PLAN.md` | one Stage 4 iteration on an existing bundle in `tools/kernel-harness/pulled/` produces a strategy string in the §6.2 form and every branch decision cites a `VERDICT` key |
| 5b | Fusion route end to end (§6.7): `scripts/xefuse_trial.py`; one ACCEPT `fusion_*` row from `bound.log` taken through generate → wrap → build → layout establishment → tile sweep → `M` sweep, no wiring yet | the series shows a tile sweep with `BUILD`/`SPILLS`/`VERDICT` per tile; the layout script shows one candidate reference at dtype-spacing error and the others at a large constant; `--tile auto` is refused |
| 6 | `optimize-intel-kernels` (§5.2); `find-kernel-gaps` + merge and delete `optimize-ssm-scan` (§5.4) | lint clean; CLAUDE.md skill list updated |
| 7 | `profile-intel` (§5.3) + `ROUTES` advice removal; `onboard-model-intel` Phase 5/6 (§5.5) | lint clean; `profile_intel.py` output carries regime/class, no verdict strings |
| 8 | `discover-model-kernels`, `wrap-kernel-for-tuning`, `measure-serving-win` (§5.6-5.8) + plain arm, `ab`, digest, tolerance helper | lint clean; `measure_serving_win.py --plain-arm` runs the child arm without `FIB_*` set |
| 9 | `setup-intel-env`, `clone-repos` environment rules and tool rows (§5.9) | no `uv pip`/`uv run` remains under `.claude/skills/`; VTune and unitrace rows print expected values |
| 10 | P3 light edits; CLAUDE.md "Agent and skill workflows" list; reference-file relabelling | lint clean repo-wide |

Steps 1-3 are the owner's named example done end to end and are the template for the rest;
review them before step 4.

## 9. Where the critique is wrong, or would cost something

1. **Not every table is hardcoding.** Fact tables about tools and silicon (`xe-matrix.md`
   DPAS and 2D-block constraints, the unitrace sync that is hooked, oneDNN verbose fields,
   provider schemas, the never-`uv run` rule) and failure tables (tool error → fix) are
   closed because their subject is closed. Deleting or "opening" them costs agents the
   facts they cannot measure cheaply. The fix is labelling (§3 "Lint" exempts fenced code and
   fact tables with a stated source) and sourcing, not removal.

2. **Withdrawn.** An earlier draft proposed keeping labelled `Prior:` lines for
   performance claims. The owner rejected it, and the rejection is right: a prior is a
   verdict from some part, shape family and library version, wrong in exactly the cases
   where reasoning pays, and redundant with the `At the bound` row of §2.1, which answers
   the same question per case for one measurement. No `Prior:` mechanism remains in this
   plan. What each former prior carried is re-expressed in §5.1 and §5.6 as either a fact
   about how the system is built or an illustration.

3. **Ordering.** The only ordering any skill carries is derived in-run: the §2.1
   classification of the kernel in hand, and the §6.4 plateau rule that forces a
   re-classification. The row order of §2.1 is a validity order (a later row's arithmetic
   is undefined until the earlier row is cleared), not a ranking of remedies. No skill
   states which remedy usually wins, and no default first move exists per regime.

4. **Probed correctness findings are not measurements to discard.**
   `quantized-matmul.md:11,15-23` records a library path that returns wrong values silently.
   That is worth more than any performance number in the tree; it is kept as an
   illustration with its re-probe command, not deleted for naming a part and a version.

5. **The loop cannot be made safe by prose alone.** Several gates the owner wants (a losing
   `best` never finalizes, arms differ only in the patch, the predicted mechanism moved)
   need the §7 code changes. Accepting this plan means accepting some script work; without
   it the rewritten skills ask the agent to enforce gates the machinery should.

6. **The dataset-facing skills are out of scope for this critique.** `extract-kernel-
   definitions`, `collect-workloads`, `add-reference-tests`, `submit-onboarding-prs`,
   `track-models`, `onboard-model` encode procedures and schemas, not optimization
   verdicts. Rewriting them to the §2 template would cost time and change nothing an agent
   does. §4 marks them P3 with the specific lines worth touching.

7. **The draft pipeline plan is heavy on deployment and light on reasoning, but its
   deployment half is right.** Installation-shape detection, overlay builds, identity
   proofs and the three-arm rebuild control (draft `:59-104`, `:391-455`) are what make a
   provider patch a result instead of a guess. They move to a reference file rather than
   being cut. Its claim that VTune is absent should not survive: presence is a `--check`
   result taken per box, and VTune changes what the Reason step can see.

8. **Some current text is already the target shape** and should be kept verbatim as the
   model for the rest: `optimize-intel-kernels/SKILL.md:229-235`, `architectures.md:8-20`
   and `:131-133`, `tools/kernel-harness/knowledge/README.md`, `route-kernel-work/PLAN.md:29-40`,
   `calibration.py:1-11`, `kernel_trials.py:1-24`.
