# FlashInfer-Bench Repo Guide

This document gives agents repo-level context for working in `flashinfer-bench`.
Use it for repository structure, trace dataset conventions, and source-of-truth guidance.
Task-specific procedures live in `.claude/skills/`.

## Project Overview

FlashInfer-Bench is a GPU kernel optimization benchmarking framework for:
- Standardizing FlashInfer Trace format
- Real-time workload tracing and collection
- Automated kernel optimization and replacement
- Performance leaderboards and tracking

### Core Concepts

1. **Definition**: Specifies an operation's interface (inputs/outputs, axes, reference implementation)
2. **Solution**: Concrete implementation of a Definition (Python/Triton/CUDA/SYCL)
3. **Workload**: Specific input configuration and test case
4. **Trace**: Execution record containing correctness and performance data
5. **Model**: Hierarchical module structure mapping model components to Definitions

## Repository Structure

```
flashinfer-bench/
├── flashinfer_bench/           # Main Python package
│   ├── data/                   #   Definition, Solution, Workload, Trace data classes
│   ├── device/                 #   Accelerator abstraction (CUDA / Intel XPU / CPU), timers,
│   │                           #   per-part calibration, host power-profile checks
│   ├── bench/                  #   Benchmarking engine, evaluators, reference cross-validation
│   ├── compile/                #   Builders: Python / Triton / CUDA (TVM-FFI) / SYCL / TileLang
│   ├── apply/                  #   Kernel auto-replacement API and its min-gain gate
│   ├── serve/                  #   Benchmark orchestration service (NOT inference)
│   ├── integration/            #   FlashInfer and vLLM adapters; Intel kernel providers,
│   │                           #   upstream-kernel baselines, in-tree kernel templates
│   ├── tracing/                #   Workload tracing utilities
│   ├── agents/                 #   Agent-facing tools: FFI/SYCL prompts, ncu/unitrace/vtune, sanitizer
│   └── cli/                    #   The `flashinfer-bench` entry point
├── tests/                      # Pytest suite; tests/scripts/ covers the pipeline scripts
├── scripts/                    # Standalone entry points: onboarding (workload collection,
│                               #   sanitization) and the kernel-optimization pipeline (below).
│                               #   Each carries a module docstring; read it before use
├── tools/                      # gpu-lock; kernel-harness/ (pipeline artefacts, below);
│                               #   onednn/ (a repro); vllm-fp8-configs/ (tuned tile configs)
├── docs/                       # Documentation (model coverage, op_type schemas, hardware support)
├── web/                        # Web UI for visualization
├── examples/                   # FFI, SYCL, kernel-generator and SGLang bench examples
├── thirdparty/cutlass/         # git submodule
├── .claude/skills/             # Agent skills, two unaccepted plans, the rewrite plan and
│                               #   the lint baseline (see below)
└── tmp/                        # Gitignored. Upstream clones (SGLang, FlashInfer, sgl-cookbook,
                                #   flashinfer-trace, the Intel kernel repos) and run artefacts
                                #   (kernel-trials/, serving-win/, ...)
```

## Python Environments

Two virtualenvs are in play. Both are uv-managed, neither has `pip`, and they are not
interchangeable:

| Environment | Where | Holds |
| --- | --- | --- |
| dev | `.venv/` in this checkout: `source .venv/bin/activate` | `flashinfer_bench`, `scripts/*.py` that do not import vLLM, the Intel provider packages the baselines call (`vllm_xpu_kernels` among them), the `ninja` the SYCL builder needs |
| serving | a second uv venv outside this repo | vLLM XPU: anything importing `vllm` — discovery, resolution, the serving A/B |

The serving venv is configured nowhere in the code: the pipeline scripts run under whichever
interpreter is active (`sys.executable`) and their child processes inherit it. Find it rather
than assume it. It is a sibling directory of this checkout named for the stack it holds —
`ls -d "$(git rev-parse --show-toplevel)"/../*venv*` — and the right one is the one whose
`python -c "import vllm"` succeeds; the dev venv fails that import. `vllm_xpu_kernels` is
present in both, so it does not tell them apart.

Activate one, then call `python`, `pytest` and `flashinfer-bench` by bare name. Activating
puts the venv's `bin/` on `PATH`, which naming its `python` by absolute path does not. With
nothing activated, bare `python` is `/usr/bin/python`, which has no torch.

**Never `uv run`, `uv sync` or `uv pip` in either environment.** `pyproject.toml` pins plain
`torch` with no index override, so uv's resolver replaces the `+xpu` torch wheel with a CUDA
build and `triton-xpu` with upstream `triton`; after that nothing runs on the GPU and the
box needs a manual reinstall. There is no safe flag: never `uv run --no-sync` either.
Installing anything — a package, an editable checkout, a provider, or `pip` itself via
`ensurepip` — is the owner's action: stop, name the package and the environment it goes
into, and ask. The mechanism, the check that detects the damage, and what the repair
consists of are in `.claude/skills/setup-intel-env/SKILL.md`, in the section titled
"Never `uv run` or `uv pip` in these venvs".

## Trace Dataset

### Single source of truth: HuggingFace

The canonical (and only) trace dataset lives at
[`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)
on HuggingFace. It contains definitions, reference tests, baseline solutions, workloads,
blobs, and evaluation traces.

There is **no** in-repo `flashinfer_trace/` directory in `flashinfer-bench`. The earlier
"internal trace layer" was removed in the trace-dataset refactor (PR #418); definitions,
reference tests, and workloads no longer live here. Skills that need to read or edit trace
content do so through a local clone of the HF dataset at `tmp/flashinfer-trace/`:

```
tmp/flashinfer-trace/                  # local clone of the HuggingFace dataset
├── definitions/{op_type}/{definition_name}.json
├── tests/references/test_{definition_name}.py
├── solutions/baseline/{op_type}/{definition_name}/...
├── workloads/{op_type}/{definition_name}.jsonl
├── blob/workloads/{op_type}/{definition_name}/*.safetensors
└── traces/{author}/{op_type}/{definition_name}.jsonl
```

Browse `tmp/flashinfer-trace/definitions/` to see the current set of supported op_types
once `/clone-repos` has been run.

### Lifecycle

1. Run `/clone-repos` to ensure `tmp/flashinfer-trace/` is checked out and up to date.
2. Generate or update content under `tmp/flashinfer-trace/` and commit on a feature branch.
3. Open a PR against the HuggingFace dataset repo (PR 2 in the onboard-model flow).
4. Open a companion PR against `flashinfer-bench` that updates **only** `docs/model_coverage.mdx`
   to reflect the new coverage (PR 1 in the onboard-model flow).

The HuggingFace dataset is the primary edit surface — flashinfer-bench owns code, docs,
and the coverage doc, not the trace data itself.

### Definition JSON Structure

Each definition JSON follows a common structure:

```json
{
  "name": "...",
  "description": "...",
  "op_type": "...",
  "tags": ["stage:decode", "status:verified", "model:...", "fi_api:...", "tp:N"],
  "axes": { "batch_size": {"type": "var"}, "num_heads": {"type": "const", "value": 16} },
  "constraints": ["len_indptr == batch_size + 1"],
  "inputs": { "tensor_name": {"shape": ["axis1", "axis2"], "dtype": "bfloat16"} },
  "outputs": { "output": {"shape": ["axis1", "axis2"], "dtype": "bfloat16"} },
  "reference": "import torch\n\ndef run(...):\n    ..."
}
```

Key conventions:
- **Axes**: `type: "var"` for runtime dimensions (batch_size, seq_len); `type: "const"` with
  `value` for model-specific constants (num_heads, hidden_size)
- **Tags**: `stage:`, `status:`, `model:`, `fi_api:`, `tp:`, `ep:`, `quantization:` prefixes
- **Reference**: Plain PyTorch `run()` function serving as ground truth
- **TP/EP**: Some kernel types (attention, MoE) produce separate definitions per tensor/expert
  parallelism setting because parallelism changes constant axis values (e.g., head counts,
  local expert counts). Other kernel types (normalization, GEMM, RoPE, sampling) are
  parallelism-agnostic. See the `extract-kernel-definitions` skill for the full rules.

Refer to `docs/flashinfer-trace/definition.mdx` for the complete schema documentation.

## Documentation Structure (`docs/`)

```
docs/
├── Getting Started        # index, installation, quickstart, hardware-support
├── Tutorials              # run-benchmark, cli, server-api, bring-your-own-kernel
├── FlashInfer Trace       # definition, workload, solution, trace schemas
├── Dataset                # model_coverage
└── Op Type Reference      # per-op-type specs (gemm, gqa, mla, moe, sampling, ...)
```

Navigation is defined in `docs/docs.json`. Page files live under `docs/start/`,
`docs/tutorials/`, `docs/flashinfer-trace/`, and `docs/op-types/`.

## Kernel Optimization Pipeline

Naming a definition to optimize picks an op someone remembered, at a shape nobody checked,
under a label the serving stack may not look up. The pipeline derives the target from the
model instead; each stage's output is the next stage's input, and every threshold in it is
measured on this part rather than written down. Discovery, resolution and the serving A/B
run under the serving interpreter (they record what that stack dispatches); the rest under
the dev venv. Procedure lives in the skills named; this table says what exists.

| Stage | Question it answers | Script |
| --- | --- | --- |
| Discover | Which ops does the model run under its stack, at what shapes, with what share of device time? Emits `discovered.json` and one verified harness per (op, shape); ops that need per-step stack state get a captured `.state.pt` beside their harness | `scripts/harness_from_model.py` |
| Resolve | Which kernel implements each op: a oneDNN primitive, a provider kernel, Triton, a Python-registered op, ATen inside PyTorch, or a decomposition? Asks the dispatcher and runs the op under `ONEDNN_VERBOSE`; `--bundle` copies the kernel's own source next to its harness with a `PROVENANCE.md` | `scripts/pull_kernel_source.py` |
| Fuse | Which producer→consumer edges the model actually ran could a GEMM epilogue absorb? Presets are read from Xe-Fuse, not copied | `scripts/fusion_candidates.py` |
| Calibrate | What does this part charge: `apply()` dispatch cost, timing floor, launch floor, read bandwidth, achieved matmul throughput? Prints the record the bounds and the apply gate consume | `scripts/calibrate_part.py` |
| Rank and bound | For every op with measured share and every delivery mechanism — provider patch, Triton in place, library call, layout transform, fusion at the call site or via `apply()`, `apply()` substitution, source rewrite — is the ceiling positive after that mechanism's cost? Every gate writes one line; `worklist.json` is the survivors ordered by worth | `scripts/bound_candidates.py` |
| Optimize | Propose, measure, branch, keep the best. `benchmark` gates on correctness before timing and interleaves arms; `ab` compares two builds of one `torch.ops` symbol across processes; `finalize` refuses a best trial that is not a measured win. `init --bound --mechanism` ties a series to an ACCEPT row of the routing, and `benchmark` refuses a rejected or stale one before timing | `scripts/kernel_trials.py` |
| Prove | Tokens/sec under vLLM, A/B, with the dispatch counters that prove the substitution happened and token digests that prove the arms agree; `--plain-arm` for a provider build or source patch. A failed gate -- an arm that failed, differing digests, nothing applied, a rejected or stale routing -- halts with no throughput printed | `scripts/measure_serving_win.py` |

Skills: `discover-model-kernels` (discover, resolve, fuse), `wrap-kernel-for-tuning` and
`optimize-intel-kernels` (optimize), `measure-serving-win` (prove). The rank-and-bound stage
is described by the `route-kernel-work` plan (below), which is not yet a skill.

A failed gate halts its stage: it prints no result a reader could take as valid, names the
gate in the key contract (`VERDICT: ...`, `DONE` last) with the evidence it had, and exits
non-zero. Stages consume each other's artefacts only with provenance intact -- `bound.json`
records the digest and model of the `discovered.json` it was computed from, and a consumer
refuses it once that report has changed (`STALE_INPUT`).

Adjacent scripts, each one question:

| Script | Question |
| --- | --- |
| `scripts/rank_vs_provider.py` | From traces on disk, does a solution beat the kernel a deployment would otherwise run, net of the calibrated substitution cost? |
| `scripts/fill_serving_gaps.py` | Which definitions would close a serving run's `no-solution` shapes? |
| `scripts/profile_intel.py`, `scripts/find_kernel_gaps.py` | Where does a model's device time go by family; which hot ops have no definition at all? (`profile-intel`, `find-kernel-gaps`) |
| `scripts/optimize_model_kernels_xpu.py` | The earlier, definition-driven loop: an unattended work-group/sub-group sweep for a target already chosen |
| `scripts/tune_vllm_fp8_config.py` | Tile configs for vLLM's block-FP8 Triton GEMM on the current device; committed output lives in `tools/vllm-fp8-configs/` |
| `scripts/observe_triton_kernels.py`, `scripts/capture_triton_kernel.py` | Which Triton kernels a model launches; capture one launch's arguments. Capture does not yet verify its replay |
| `scripts/build_onednn.py`, `scripts/port_triton_solutions_to_xpu.py` | Build a chosen oneDNN into its own prefix; rewrite CUDA-bound Triton solution wrappers as new, device-agnostic solutions |

Artefacts, and what may be committed:

```
tools/kernel-harness/
├── auto/             harnesses + discovered.json per model run     gitignored; regenerate
├── pulled/<op>/      harness.py, source/, PROVENANCE.md             gitignored; third-party source
├── *.state.pt        captured serving-stack state (large)           gitignored; never commit
├── trials/           hand-written trial files and winners           tracked
├── optimized/        in-place substitution patches for a stack      tracked
├── knowledge/        rules a trial follows; no measured values      tracked
└── sycl_harness.py   inline SYCL/oneDNN source → harness contract   tracked
tmp/kernel-trials/<series>.json    trial trees written by kernel_trials.py
```

## Where To Look By Task

### Understanding data structures

`flashinfer_bench/data/` defines `Definition`, `Solution`, `Workload`, `Trace`, and
`TraceSet` — the core data classes used throughout the codebase.

### Running or writing benchmarks

`flashinfer_bench/bench/` is the engine (benchmark, evaluators, runners, and
`reference_check.py`, which cross-validates a reference on a new accelerator against the
host before it is trusted); `flashinfer_bench/compile/` builds solutions. The CLI in
`flashinfer_bench/cli/main.py` exposes `run`, `serve`, `report`, `validate`, plus `ref`
(reference cross-validation on a device), `baselines` (add upstream Intel kernels to a
dataset as competitors) and `providers` (`list`, `verify`, `install`).

### Writing kernels for Intel GPUs

SYCL is a first-class solution language (`spec.language: "sycl"`), built by `SyclBuilder`
with oneAPI DPC++. Kernels run on PyTorch's own `sycl::queue`, obtained the same way CUDA
kernels obtain their stream:

```cpp
sycl::queue* q = static_cast<sycl::queue*>(
    TVMFFIEnvGetStream(dev.device_type, dev.device_id));
```

A worked example with a Definition, Solution and workloads lives in `examples/sycl/`.
`flashinfer_bench.SYCL_PROMPT` is the agent-facing guidance, mirroring `FFI_PROMPT` for
CUDA. Upstream Intel kernels become ordinary Solutions through
`integration/xpu_kernels.py`, the project's own templates through
`integration/intree_kernels.py`, and the providers are acquired through
`integration/providers.py`.

Definitions and workloads are hardware-agnostic and are never re-collected per backend;
only solutions and traces carry hardware identity.

### Device backends and timing

`flashinfer_bench/device/` owns every backend-specific operation. Devices are addressed by
string (`cuda:0`, `xpu:0`, `cpu`) and synchronization, device selection, cache management,
timing methodology and capability reporting all go through `get_accelerator(device)`.
Benchmark, evaluator and runner code must contain no vendor branches; add a backend by
registering an `Accelerator`, and a new device within an existing backend by adding a
capability record.

Timing methodologies are not interchangeable (CUPTI device-side duration vs. device
events including launch overhead), so the timer used is recorded in every trace at
`evaluation.environment.libs.timing`.

Per-part costs are the same class of fact and are measured, never written down.
`flashinfer_bench.device.calibration.get()` measures once per (part, timer, stack), caches
the record under `FIB_CACHE_PATH`, and returns what an `apply()` substitution costs per call
(`dispatch_us`), the timer's per-region floor (`timing_floor_us`), the per-launch floor
(`launch_floor_us`), contiguous read bandwidth (`bandwidth_gbs`) and achieved matmul
throughput per native dtype (`matmul_peak_tflops`); `scripts/calibrate_part.py` prints it.
These are the thresholds behind the apply gate, the candidate bounds and the trial gates,
and they belong in no default and no document: values hand-measured on one machine had
moved, one by most of its value, when re-measured a day later. A quantity that could not be
measured is `None`, never `0.0`, and a caller must treat `None` as unknown and fail closed —
a gate set to zero admits every substitution. A record with an unmeasured field is returned
but not cached, so the next process measures again. `device/power.py` flags a host power
profile that distorts measurements; heed its warning before timing anything.

### Kernel auto-replacement at runtime

`flashinfer_bench/apply/` holds `apply(...)`, the shared entry point for both optimized
kernel dispatch and workload tracing. Substitution is gated by `ApplyConfig.min_gain_us`
(`FIB_APPLY_MIN_GAIN_US`): above zero, a key is indexed only when the best solution beats
the best provider baseline by more than that margin on the table's hardware, and keys with
no provider baseline are not indexed at all. Zero disables the gate and is the default; set
it from the calibration, and do not enable `apply()` with the gate open when the calibration
reports `dispatch_us=None`.

### Model coverage or web metadata

`web/apps/web/data/` is the web UI data layer; `docs/model_coverage.mdx` is the coverage
document.

### Benchmark service behavior

`flashinfer_bench/serve/` exposes benchmark orchestration as a service — it is **not** an
inference server.

### Dataset-facing questions

When the question is about published trace contents, workload coverage, or synced definitions,
reason against the external dataset
[`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace).

### Agent and skill workflows

`.claude/skills/` holds one directory per skill, each with a `SKILL.md`. The frontmatter
`description` is the routing text; the summaries below are orientation only.

- **onboard-model**: CUDA onboarding pipeline — discover and classify kernels, generate
  definitions, collect workloads, open the PRs; orchestrates the per-phase skills through a
  run manifest
- **discover-models**: Phase 1 of onboarding — candidate LLMs and a kernel inventory (the
  definitions a model needs, existing or new, and which backend supplies a kernel); writes
  the run manifest
- **extract-kernel-definitions**: Definition JSON by harvesting an SGLang pass with
  FlashInfer's trace dumper (CUDA), by transcription from model sources and `config.json`,
  or by module hooks on Intel; owns the naming and axis rules
- **collect-workloads**: Real workloads from an SGLang run (FlashInfer Level-10 dump),
  sanitized and verified non-synthetic. Requires NVIDIA
- **add-reference-tests**: The pytest that validates a definition's reference against
  FlashInfer or SGLang ground truth (Intel cross-validation in its Intel GPUs section)
- **submit-onboarding-prs**: The per-definition PR pair — HF dataset, then
  `docs/model_coverage.mdx` — with the pre-flight `validate` gate
- **track-models**: Maintain `docs/model_coverage.mdx`; owns its format and nothing else
- **clone-repos**: Clone or update SGLang, FlashInfer, sgl-cookbook, flashinfer-trace and
  optionally the Intel kernel repos into `tmp/`, recording their SHAs
- **setup-intel-env**: Bring an Intel box up — driver, PyTorch XPU, oneAPI DPC++, kernel
  providers, unitrace, optionally vLLM XPU — and the venv rule above. Run before any Intel work
- **onboard-model-intel**: Intel counterpart of `onboard-model` — definitions on `xpu:0`
  without CUDA, cross-validated references, profiling, kernels sourced from oneDNN /
  vllm-xpu-kernels / sgl-kernel-xpu / Xe-Fuse / SYCL, and triage of a slow or wrong op to
  reference, harness or provider
- **profile-intel**: Rank kernel families by recoverable device time on Intel and route each
  to the skill that fixes it; unitrace, VTune (`flashinfer_bench/agents/vtune.py`, with its
  root-gated prerequisites) and torch.profiler usage on XPU
- **find-kernel-gaps**: Hot ops that no definition covers; rewrite versus new kernel; a
  definition plus solution for the ones worth it
- **discover-model-kernels**, **wrap-kernel-for-tuning**, **measure-serving-win**: the
  pipeline stages above — discover/resolve/fuse, harness and tune a kernel inside its stack
  without extracting it, and the serving A/B
- **optimize-intel-kernels**: SYCL or Triton solutions for existing definitions on Intel,
  including porting CUDA Triton solutions to XPU. Carries `architectures.md` (per-part
  traps), `xe-matrix.md` (DPAS-backed kernels) and `xe-fuse.md`. Per-part *values* are
  queried from `Capabilities` and the calibration, not tabulated
- **optimize-onednn**: A slow oneDNN GEMM on Intel — `ONEDNN_VERBOSE` and dispatch output,
  resolving a rejection to the gate in oneDNN's source, the call-level fixes
- **optimize-ssm-scan**: State-space / SSD scan kernels for hybrid models on Intel; use
  when profiling reports materialised high-rank contractions

Two directories under `.claude/skills/` hold **plans, not skills**. They have no `SKILL.md`
by design and nothing routes to them until the owner accepts them:
`optimize-model-kernels/` (`PLAN.md`, the end-to-end pipeline skill the stages above are
to become) and `route-kernel-work/` (`PLAN.md`, `RUN.md`: choosing among candidates by
measured ceiling; `scripts/bound_candidates.py` implements the bounding it describes).

`.claude/skills/SKILLS-REWRITE-PLAN.md` is the accepted plan the skill rewrites follow: a
skill stores a measurement and a way to reason from it, never a conclusion reached once on
one part. `scripts/lint_skills.py` enforces its conventions mechanically — no stored
measurements, no performance expectations, no part, model or definition names outside
illustrations, no remedy orderings or closed remedy lists, no `uv run`, no broken
references. `.claude/skills/lint-baseline.json` records the pre-rewrite violations;
`--check-baseline .claude/skills/lint-baseline.json` fails only on new ones, and the baseline
is re-recorded when a skill is finished. `CLAUDE.md` is outside the linter's default path
set; run `python scripts/lint_skills.py CLAUDE.md` after editing it.

## Common Misunderstandings

### Tracing and apply are unrelated entry points

They are different runtimes, but the `apply(...)` call path is a shared entry point for both
optimized dispatch and workload collection (tracing). This matters when reading runtime
interception logic in `flashinfer_bench/apply/`.

### Benchmark only measures speed

Benchmark also validates correctness against the reference implementation and stores
evaluation results as traces.

### A per-kernel ratio is a result

It is not. `speedup_factor` in a trace is measured against the definition's PyTorch
reference, which exists to decide correctness. Deployment is decided against the kernel the
serving stack would otherwise run (`scripts/rank_vs_provider.py`), net of what the delivery
mechanism costs (`calibration.get().dispatch_us`), and proven as tokens/sec
(`scripts/measure_serving_win.py`).

### `serve/` is for generic inference traffic

It is not. The serve subsystem is a benchmark orchestration service over dataset-backed
workloads.

## Contributing New Operation Types

To add a new op_type beyond what currently exists:

1. Create operation documentation in `docs/op-types/`
2. Create Definition JSON files under `tmp/flashinfer-trace/definitions/{new_op_type}/`
   (the HuggingFace dataset clone — submit via a PR to `flashinfer-ai/flashinfer-trace`)
3. Provide a Python reference implementation in the definition's `reference` field
4. Create Solution implementations (Triton/CUDA/SYCL optimized)
5. Optionally create a FlashInfer adapter in `flashinfer_bench/integration/`

The existing op_type directories under `tmp/flashinfer-trace/definitions/` serve as templates.

## Maintenance Notes

Update `CLAUDE.md` when any of the following change:

- The internal vs external trace boundary or sync lifecycle
- Repository directory structure
- The set of supported device backends, the timing methodology per backend, or what the
  calibration measures and how callers must treat an unmeasured value
- The Python environments on the box, or the rule about how packages get into them
- The kernel-optimization pipeline: its scripts, the artefacts they write under
  `tools/kernel-harness/` and `tmp/`, and which of those are gitignored
- Which `.claude/skills/` directories are routable skills and which are plans; the skill
  lint rules or where its baseline lives
- Core concept definitions
- The definition JSON schema conventions

Update the relevant `.claude/skills/*.md` files when task procedures change.
Keep this file focused on repo-level context. Skill-specific procedures and
op_type-specific details belong in their respective skill files.

## References

- [FlashInfer Documentation](https://docs.flashinfer.ai)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [HuggingFace Hub](https://huggingface.co/models)
- [Definition Schema Documentation](docs/flashinfer-trace/definition.mdx)
- [Operation Type Schema](docs/op-types/)
