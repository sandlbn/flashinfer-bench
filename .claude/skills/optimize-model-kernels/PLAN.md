# Plan: optimize-model-kernels

Status: **plan awaiting acceptance**. Not a skill; no `SKILL.md` exists until the owner accepts
this and the gaps in §12 are closed.

One procedure, four phases in the owner's order: find the ops a model runs, find the kernel
behind each, optimize each in the trial loop, put the result back into the serving stack and
measure it. The loop is the Xe-Forge shape — reason about why it is slow, change it, measure,
feed the result or the error back, repeat — with one invariant enforced mechanically at every
exit: **a kernel that is slower or wrong never passes.**

Nothing in this file is a stored measurement. Every number a stage needs is named by the
command that produces it on this box, for this model, under this stack.

## 0. Conventions

| Item | Rule |
| --- | --- |
| Dev interpreter | `source .venv/bin/activate` — `flashinfer_bench`, `scripts/*.py` that do not import vLLM |
| Serving interpreter | `source /home/sand/Projects/vllm-xpu-venv/bin/activate` — anything importing `vllm` or `vllm_xpu_kernels`: discovery, resolution, serving A/B, provider builds |
| Never | `uv run`, `uv pip`, or `pip` resolving dependencies in either venv: both replace torch-xpu with CUDA torch. Every install below is `--no-deps --no-index` and requires the owner's approval first |
| GPU | One benchmark at a time. Before any timed stage: `fuser -v /dev/dri/renderD* 2>&1` must list only your process; `powerprofilesctl get` must read `performance` |
| Installation shape | Never assumed. §2.1 detects, per component, whether it is an editable install backed by a checkout or a wheel with no source, and whether it carries compiled extensions; the patch mechanism (§7.1) follows from that detection |
| Provenance | Every stage and every A/B arm records the revision of the stack and of the provider (§2.2). Arms whose stack provenance differs are refused as non-comparable (§10) |
| unitrace | `/home/sand/Projects/pti-gpu/tools/unitrace/build/unitrace` — not on `PATH`; export it in every shell that profiles |
| VTune | Not installed. §6.4 is optional and gated on an install the owner must authorise |
| Artefacts | `tools/kernel-harness/auto/` (discovery), `tools/kernel-harness/pulled/<op>/` (bundles), `tmp/kernel-trials/<series>.json` (trial trees), `tmp/unitrace/`, `tmp/provider-builds/<id>/` (overlays), `tmp/serving-win/`, `tmp/provenance/<run>.json` |

## 1. The stages

| # | Stage | Command | Output | Gate to the next stage |
| --- | --- | --- | --- | --- |
| 0 | Preflight | §2 checks | calibration cache, tool paths, installation shape per component, provenance record | every row of §2 prints its expected value; shape and provenance written |
| 1 | Discover | `scripts/harness_from_model.py` | `discovered.json`, verified harnesses, bound to a provenance record | at least one harness verified; `device_time_total_us > 0` |
| 2 | Resolve | `scripts/pull_kernel_source.py --bundle` + unitrace | provider per op, source bundle, kernel names and properties | every op with share has a provider class; unitrace names agree with the resolver |
| 3 | Bound and rank | `calibration.get()` + `scripts/fusion_candidates.py` | ordered worklist with mechanism per row | worklist non-empty; each row's ceiling `> 0` for its mechanism |
| 4 | Trial loop | `scripts/kernel_trials.py` | trial tree, best node, control result | best is correct, beats production outside spread, and the control attributes the win to the kernel |
| 5 | Patch and rebuild | §7 | overlay build, identity proof, diff | rebuilt kernel proven to be the one running; unpatched-rebuild control measured |
| 6 | Serving A/B | `scripts/measure_serving_win.py` | tokens/sec per arm, token digests | delta outside spread, digests agree, no `apply()` in the path |
| 7 | Promote or revert | §9 | installed wheel or restored snapshot; record | owner approved the install; revert path tested |

Stages 4 to 6 iterate. A failure at 5 or 6 returns to 4 with the failure as input, never forward.

## 2. Stage 0 — Preflight

Run in the serving interpreter unless marked.

| Check | Command | Expected | Failure means |
| --- | --- | --- | --- |
| Device | `python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_properties(0).name)"` | `True <part>` | driver or torch-xpu; stop, `/setup-intel-env` |
| Stack | `python -c "import vllm, vllm_xpu_kernels._C as m; print(vllm.__file__, m.__file__)"` | both import; paths feed §2.1 | wrong interpreter |
| Compiler (dev) | `python -c "from flashinfer_bench.compile.builders import SyclBuilder; print(SyclBuilder.is_available())"` | `True` | `source /opt/intel/oneapi/setvars.sh` or set `FIB_SYCL_COMPILER` |
| Calibration (dev) | `python scripts/calibrate_part.py` | dispatch cost, timing floor, bandwidth printed | `None` for dispatch: dataset or provider missing; the §4 bound for `apply()` is then unavailable, not zero |
| unitrace | `export PATH=/home/sand/Projects/pti-gpu/tools/unitrace/build:$PATH; unitrace --version` | a version | path wrong; rebuild per `flashinfer_bench/agents/unitrace.py` docstring (owner approval) |
| Metrics (optional) | `unitrace --metric-list` | metric groups listed | needs `sysctl dev.xe.observation_paranoid=0` — a sudo action the owner performs, or skip `-q` |
| GPU idle | `fuser -v /dev/dri/renderD*` | nothing foreign | another agent is on the GPU; do not time anything |
| Power | `powerprofilesctl get` | `performance` | set it; numbers taken otherwise are discarded |

### 2.1 Detect the installation shape of every component

Two components matter: the serving stack (`vllm`) and the kernel provider
(`vllm_xpu_kernels`). Ask each the same three questions; never read the answer off a
remembered layout.

| Question | Command (`<dist>` = `vllm` or `vllm_xpu_kernels`; `<mod>` the import name) | Reading |
| --- | --- | --- |
| Editable checkout or wheel? | `python -c "import importlib.metadata as m; d=m.distribution('<dist>'); print(d.read_text('direct_url.json'))"` | `{"url":"file://<root>","dir_info":{"editable":true}}` = editable checkout at `<root>`; `None` or no `editable` = wheel |
| Where does it import from? | `python -c "import <mod>, os; print(os.path.dirname(<mod>.__file__))"` | outside `site-packages` = checkout; inside = wheel. Must agree with the row above |
| Does it carry compiled extensions? | `find <that dir> -name '*.so' -o -name '*.pyd' \| wc -l` | `0` = pure Python: an edit or a branch switch is picked up by the next process, no build; `> 0` = compiled: a source change needs a rebuild |
| Is there source for a wheel? | `ls tmp/<dist>` (clones from `/clone-repos`) and `git -C tmp/<dist> log -1 --format=%h` | present = §7.2 can rebuild it; absent = clone first |

Mechanism selection, read from the answers:

| Shape detected | Patch mechanism | Rebuild | Revert |
| --- | --- | --- | --- |
| editable checkout, pure Python | edit in place, or `git switch <branch>` in `<root>` | none; next process picks it up (clear `~/.triton/cache` for Triton) | `git -C <root> checkout -- <file>` / `git switch <original>` |
| editable checkout, compiled | edit in place | `python setup.py build_ext --inplace` in `<root>` (same subshell rules as §7.2) | `git checkout` + rebuild |
| wheel, source clone present | edit the clone | §7.2 overlay build, `PYTHONPATH` per process | drop the `PYTHONPATH`; §9 for an installed one |
| wheel, no source | none until cloned | — | — |

Record the shape table in the provenance record (§2.2). On this box today the detection
returns "editable checkout, pure Python" for the stack and "wheel, source clone present"
for the provider; a future box may return the reverse, and the plan is unchanged.

### 2.2 Record provenance — every stage, every arm

A version string proves nothing here: an editable checkout reports whatever
`setuptools_scm` stamped at install, and a rebuilt wheel from an untagged clone reports a
dev version. Record what identifies the code that runs:

| Component shape | Record | Command |
| --- | --- | --- |
| checkout | commit, branch, dirty flag | `git -C <root> rev-parse HEAD; git -C <root> branch --show-current; git -C <root> status --porcelain \| wc -l` |
| wheel | every `.so` path, size, mtime, sha256 | `find <dir> -name '*.so' -exec sha256sum {} \; ; stat -c '%n %s %Y' <dir>/*.so` |
| overlay (§7.2) | the same, under `tmp/provider-builds/<id>/`, plus `patch.diff` and the clone's commit | as above, plus `git -C tmp/<dist> rev-parse HEAD` |
| both | the interpreter and torch build | `python -c "import sys, torch; print(sys.executable, torch.__version__)"` |

Write it to `tmp/provenance/<run>.json` before Stage 1 and again inside each A/B arm (the
worker process, not the launcher: `VLLM_ENABLE_V1_MULTIPROCESSING=0` for the proof run, or
print it from `sitecustomize`-level code). Two records are **comparable** when the stack
component's entry is identical and the provider entries differ only in the component under
test. Anything else is refused, not reported — switching the stack's branch between arms
changes the scheduler, the attention-backend selection and the Python call path, and the
delta would be booked to a kernel.

## 3. Stage 1 — Discover the ops

```bash
python scripts/harness_from_model.py --model <repo_id> --out-dir tools/kernel-harness/auto \
    --prompts <n> --out-tokens <n> --top <k>
```

Inputs: the model id, the serving stack in this interpreter. Outputs: `discovered.json`
(`ops` with shapes/dtypes/calls, `op_share`, `device_time_by_kernel`, `edges`, `triton`) and
one verified harness per (op, shape) for the top `k` ops by call count.

Read before continuing:

| Field | Use |
| --- | --- |
| `op_share` | the ranking; call count is not share |
| `device_time_by_kernel` | what unitrace must agree with in Stage 2 |
| `edges` | fusion candidates; a per-op tally cannot supply these |
| `triton` | kernels the dispatcher never sees, with `file:line` |
| discard reasons on stdout | `Forward context is not set` = needs the stack around it (`/wrap-kernel-for-tuning`); `shape ... != ... seen in the model` = a real defect in recording |

Failure modes: `device-time pass failed` → shares are absent and Stage 3 cannot rank; re-run
before proceeding. Zero Triton kernels on a stack known to JIT them → the observer was not
installed in the worker; discovery forces `VLLM_ENABLE_V1_MULTIPROCESSING=0`, check it held.

Prefill and decode shapes are different problems. Run discovery at a decode-sized and a
prefill-sized `--out-tokens`/`--prompts` pair and keep both `discovered.json` files.

`discovered.json` is valid only for the stack provenance it was recorded under: the ops,
the edges and the shares are properties of that branch's scheduler and backend selection.
Name the out-dir by it (`--out-dir tools/kernel-harness/auto/<stack commit>`) and copy the
§2.2 record beside it. A stack branch switch, a checkout edit outside the kernel under test,
or a changed dirty flag invalidates the worklist; re-run Stage 1 before any further stage.

## 4. Stage 2 — Resolve each op to the kernel that ran, and see it under unitrace

### 4.1 Resolve

```bash
python scripts/pull_kernel_source.py \
    --from-report tools/kernel-harness/auto/discovered.json \
    --from-harnesses tools/kernel-harness/auto \
    --bundle tools/kernel-harness/pulled
```

Asks `torch._C._dispatch_dump` for the registering source and runs each op once under
`ONEDNN_VERBOSE=1`. Each op lands in one class; the class decides the mechanism in Stage 3:

| Resolved as | Established by | Where the change goes | Pays substitution cost |
| --- | --- | --- | --- |
| oneDNN | a `primitive,exec` line | the **call**: layout, fpmath, post-ops, in vLLM's Python or a load-time weight transform (`/optimize-onednn`) | no |
| provider kernel | `XPU` key registered from the provider's `torch_bindings.cpp`; source found in `tmp/vllm-xpu-kernels` | the provider source, rebuilt (§7) | no |
| Triton | JIT record with `file:line` inside the stack's import root | that file, by the stack's detected shape (§2.1: pure-Python checkout = edit in place) | no |
| Python-registered op | no dispatcher entry; namespace names the package | its Python source, in place | no |
| ATen inside PyTorch | key registered from `/aten/` | none locally; measure, report upstream | — |
| decomposition | composite key only, no oneDNN line | whatever it decomposes to | — |
| GEMM → elementwise edge | `fusion_candidates.py` proposes a preset | the GEMM call (post-op or Xe-Fuse) | no |
| `apply()` substitution | last resort, any class | the dataset | **yes** |

Failure modes: a `_C::` op reported `not on this device` or `unresolved` means the
provider's extension was not imported in the resolver process — run in the serving
interpreter. `registered, source not on this box` → the clone is missing; `/clone-repos`.
`PROVENANCE.md` listing numpy or pandas as "defines it" is a scoping miss on an ATen op;
the classification stands, the source list does not.

### 4.2 unitrace — identity and kernel properties

unitrace answers two questions torch.profiler cannot: *which compiled kernel* (by its
demangled name and launch geometry) and *what the compiler did to it* (register spill, SIMD
width, GRF mode). It does not produce the share; that stays with `discovered.json`.

Per op, on its verified harness. The script under unitrace must end with a sync unitrace
hooks, or the report contains only a summary:

```bash
export PATH=/home/sand/Projects/pti-gpu/tools/unitrace/build:$PATH
unitrace -d -v -o tmp/unitrace/<op>.txt python - <<'PY'
import importlib.util, torch
spec = importlib.util.spec_from_file_location("h", "tools/kernel-harness/pulled/<op>/harness.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
model, inputs = m.Model(*m.get_init_inputs()), m.get_inputs()
for _ in range(<warm+timed calls>): model(*inputs)
torch.xpu.current_stream().synchronize()   # zeCommandListHostSynchronize -- hooked
PY
```

`torch.xpu.synchronize()` is not hooked. An empty report is this mistake, not a broken tool.

Read from `tmp/unitrace/<op>.txt.<pid>`:

| Section | Column | Meaning for the loop |
| --- | --- | --- |
| Device Timing | kernel name | the identity: `vllm::…` functor names are the provider; `gemm_kernel`/`jit:gemm` names are oneDNN; a bare function name is Triton. Must agree with 4.1 |
| Device Timing (`-v`) | global/local sizes | the launch geometry that actually ran — the first tuning axis |
| Device Timing | avg time | device-side per-kernel time, free of the host call path; the number the §6 control compares against |
| Kernel Properties | `Spill Memory Per Thread` | nonzero = the kernel is register-spilling; fix this before anything else |
| Kernel Properties | SIMD width, GRF size | whether the compiler chose the width and register mode the source asked for |

Whole-model run (optional, for cross-checking `device_time_by_kernel`): the same invocation
around a short `LLM.generate` with `VLLM_ENABLE_V1_MULTIPROCESSING=0` and the hooked sync
at the end. Kernel names from this run map each `device_time_by_kernel` row to a 4.1 class.

Hardware metrics (`-q`, `--stall-sampling`) need the sysctl from §2 and, on this part, may
list no usable metric group. If `--metric-list` is empty, that is the hole §6.4 describes.

Gate: every op carrying share has a class, and the unitrace kernel name for each harness
matches the class. Disagreement means the harness is not calling what the model called.

## 5. Stage 3 — Bound and rank before writing anything

```python
from flashinfer_bench.device import calibration
cal = calibration.get()          # dispatch_us, timing_floor_us, bandwidth_gbs; None if unmeasurable
```

Per candidate row `(op, shape, calls, share, class)` compute, from measurement only:

```
current_us    = unitrace avg device time for this harness (Stage 2)
achievable_us = bytes the kernel is obliged to touch / cal.bandwidth_gbs
              -- refine by timing a contiguous read of the same bytes in the same shape
floor_us      = cal.timing_floor_us
mechanism_us  = cal.dispatch_us if the mechanism is apply(); 0 for every in-place mechanism
ceiling_us    = current_us - max(achievable_us, floor_us) - mechanism_us
worth         = ceiling_us * calls / device_time_total_us
```

Rules:

- `ceiling_us <= 0` ⇒ unroutable **for that mechanism**. Record it with the arithmetic and
  either change mechanism or drop the row. Elementwise work is routinely unroutable through
  `apply()` and routable as a source patch or a fusion.
- `cal is None` or `cal.dispatch_us is None` ⇒ the `apply()` mechanism is unavailable, not free.
- Rank by `worth`, not by share. The largest share is often the least movable.
- Fusion rows come from `python scripts/fusion_candidates.py --report tools/kernel-harness/auto/discovered.json`;
  their `worth` is the consumer's share (a launch and a round trip removed), not a ratio.
- A GEMM row's ceiling is the call-level slack (`/optimize-onednn` Steps 1–3 on its shape),
  not a replacement kernel.

Output: the worklist, in order, each row naming its mechanism and ceiling. Stop-line: the
first row whose ceiling is inside the spread `kernel_trials.py benchmark` reports for that
harness. State the cutoff.

Failure modes: a ceiling computed from the definition's PyTorch reference instead of the
harness is not a ceiling; every `current_us` here comes from the production op.

## 6. Stage 4 — The trial loop

One series per worklist row. The baseline is the **production harness** from the bundle;
never a definition reference.

```bash
python scripts/kernel_trials.py init      <series> tools/kernel-harness/pulled/<op>/harness.py
python scripts/kernel_trials.py save      <series> <trial.py> --parent <tN> --strategy "<why>"
python scripts/kernel_trials.py benchmark <series> <trial.py> --trial <tM> [--rounds N --calls N]
python scripts/kernel_trials.py status    <series>
python scripts/kernel_trials.py best      <series>
```

`benchmark` is the only sanctioned number: it gates on correctness (exit 1, nothing timed),
warms both arms, interleaves rounds, reports median and spread, and flags a difference
inside the spread. Never write another timing script.

### 6.1 One iteration (the Xe-Forge loop)

| Step | Input | Action | Output fed forward |
| --- | --- | --- | --- |
| Reason | bundle `source/`, `PROVENANCE.md` schema, unitrace properties, last result | state one hypothesis for why it is slow, in the `--strategy` string | the strategy |
| Change | the best node's file | one change that tests the hypothesis | `<trial.py>` |
| Build | `sycl_harness.build(...)` or Triton JIT | compile | on failure: the compiler's stderr, verbatim, is the next iteration's input; save the fix as a child of the same parent |
| Measure | `benchmark --trial` | correctness then timing | `INCORRECT: <argument, max abs err, tol>` → fix in place (child of the failing trial); a ratio + spread → next hypothesis |
| Branch | `status` | improved: continue from it. Regressed: `--parent` = `best`. Inside spread: raise `--rounds`/`--calls` once, then treat as regressed | the tree |

First hypotheses come from the source's constants, checked against the device rather than
assumed: `reqd_sub_group_size`, work-group size, vector width, rows per work-group
(`torch.xpu.get_device_properties(0).sub_group_sizes`, `caps.vector_width`,
`architectures.md`). Nonzero spill in Kernel Properties outranks all of them: shrink the
tile or set the large-GRF option, measure each alone, never both at once.

### 6.2 Trial forms by class

| Class | Trial file contains | Compiled by |
| --- | --- | --- |
| provider kernel | the bundled SYCL source, edited, wrapped with a TVM-FFI entry | `tools/kernel-harness/sycl_harness.py::build` |
| Triton | the kernel function copied from its `file:line`, edited | Triton JIT; clear `~/.triton/cache` when a change appears not to take |
| oneDNN | the same GEMM with a different layout / post-op / fpmath, as a SYCL+oneDNN trial or a Python call change | `build(..., dependencies=["onednn"])` or plain Python |
| fusion | the Xe-Fuse or post-op kernel for the edge | `xe-fuse.md` |

### 6.3 The control port — required before any win is attributed

A provider trial runs through TVM-FFI while the baseline runs through `torch.ops`. The call
paths differ, and the difference can be a large part of the measured ratio. Before the
`best` node is accepted:

1. Save a **control**: the *unmodified* provider algorithm ported line for line onto the
   trial call path (`tools/kernel-harness/trials/fusedadd_control_port.py` is the form).
   `--strategy "control: faithful port of stock kernel"`.
2. `benchmark` it. Its ratio against production is the call-path component.
3. The kernel's own contribution is `best.candidate_us` against `control.candidate_us`.
   That is the number a source patch will deliver, and the number Stage 5 must reproduce.

If `best` does not beat the control outside the spread, the kernel change did nothing; the
row is closed as "call path only" and no patch is made.

### 6.4 VTune — optional, install gated on the owner

Not installed on this box. Everything above runs without it. It shortens the *Reason* step
when unitrace's Kernel Properties say a kernel is slow but not where:

| Question | unitrace | VTune (`-collect gpu-hotspots`) |
| --- | --- | --- |
| Is it spilling, and how much | yes (`Spill Memory Per Thread`) | yes, and **which instructions** spill and fill (`-knob profiling-mode=source-analysis`) |
| Where do memory stalls come from | no on this part unless `--stall-sampling` is supported | per-instruction stall attribution: global vs SLM vs L3 miss, basic-block latency |
| Occupancy achieved vs theoretical | no | yes (`-knob characterization-mode=overview`: EU active/stalled/idle, occupancy) |
| Dynamic instruction mix (did the vector loads get emitted) | no | yes (`-knob characterization-mode=instruction-count`) |

Prerequisites the owner must authorise: the oneAPI VTune component, its sampling driver or
the perf/observation sysctls. Invocation once present:

```bash
vtune -collect gpu-hotspots -knob characterization-mode=overview -r tmp/vtune/<op> -- python <runner>
vtune -collect gpu-hotspots -knob profiling-mode=source-analysis \
      -knob computing-task-of-interest=<kernel name from unitrace> -r tmp/vtune/<op>-src -- python <runner>
vtune -report summary -r tmp/vtune/<op>
```

### 6.5 Exit conditions for a series

Stop and record the reason when the first of these holds:

| Condition | Verdict |
| --- | --- |
| `best` correct, beats production and the control outside spread | proceed to Stage 5 |
| `best.candidate_us` within spread of `max(achievable_us, floor_us)` | at the bound; proceed if it also beats production |
| `K` consecutive trials from `best` without an improvement outside spread (`K` fixed at `init` by the operator) | plateau: change the algorithm once; if that also plateaus, close the row |
| trial budget spent | close the row with `status` output |
| every trial `INCORRECT` for the same reason | the harness or schema is misread; re-read `PROVENANCE.md`, do not loosen tolerance |

## 7. Stage 5 — Patch the component and prove the patched code is the one running

### 7.1 Where each class is patched — decided by the §2.1 detection

The class (Stage 2) says which component owns the kernel; the detected shape of that
component says how a change reaches a running process.

| Class | Component | Edit | Then, per detected shape (§2.1) |
| --- | --- | --- | --- |
| provider kernel (SYCL) | provider | its `csrc/...` in the checkout or the clone | compiled: editable checkout → `build_ext --inplace`; wheel → §7.2 overlay |
| Triton | stack | the `file:line` discovery recorded | pure Python → nothing to build; `rm -rf ~/.triton/cache`; next process JITs the edit |
| Python-registered op / provider call site | stack | the Python source | as above |
| oneDNN call | stack | the call site, or a load-time weight transform (`flashinfer_bench/integration/weight_layout.py` pattern) | as above |
| oneDNN catalog (Fix 5) | oneDNN | `tmp/oneDNN/.../kernel.db` via `scripts/build_onednn.py` | own prefix under `FIB_ONEDNN_DIR` — **note §12.2** |

Keep the diff of whichever tree was edited:
`git -C <root or clone> diff > tmp/provider-builds/<id>/patch.diff`. Revert is the §2.1
table's last column. If the shape detected for a component is "wheel", nothing edited in a
clone reaches a process until §7.2 has run and `PYTHONPATH` selects the result — that is the
mistake the detection exists to prevent, in both directions.

### 7.1a Running the stack from a branch

The owner may want the stack on a branch other than the one it was installed from. When
§2.1 reports the stack as an editable, pure-Python checkout:

```bash
git -C <root> status --porcelain          # must be empty, or stash: a dirty tree is not a branch
git -C <root> switch <branch>             # that is the whole procedure; no build, no reinstall
python -c "import vllm; print(vllm.__file__)"   # still <root>: the .pth points at the directory
```

A worktree elsewhere is **not** picked up — the editable `.pth` names one directory. Switch
in place, or point a second venv at the worktree (an install, owner approval).

Consequences the plan enforces:

| Effect | Rule |
| --- | --- |
| provenance changes | re-record §2.2 |
| `discovered.json` is stale | re-run Stage 1 on the branch; the worklist from the old branch is void |
| A/B arms | both arms on the same branch and the same commit, or refused (§10). A branch-vs-branch comparison is a *stack* A/B, run with the provider held fixed, and is reported as such — never as a kernel result |
| provider compatibility | the branch may call provider ops the installed wheel lacks; the Stage 0 `Stack` row and Stage 2 resolution surface this as `not on this device` |

If the stack is a wheel instead, a branch cannot be selected without cloning it and
installing editable — an owner-approved install, outside this procedure.

### 7.2 Build the provider without touching the environment

Serving interpreter. The compiler comes from oneAPI; keep it in a subshell so `setvars.sh`
does not leak into the venv's shell.

```bash
( source /opt/intel/oneapi/setvars.sh >/dev/null
  export MAX_JOBS=<n>                                   # see below
  export VLLM_XPU_AOT_DEVICES=<caps.sycl_target> VLLM_XPU_XE2_AOT_DEVICES=<caps.sycl_target>
  cd tmp/vllm-xpu-kernels && python setup.py bdist_wheel )
python -m zipfile -e tmp/vllm-xpu-kernels/dist/<wheel>.whl tmp/provider-builds/<id>/
```

- `MAX_JOBS`: `setup.py` defaults to `min(cores, total_memory / per-job estimate)`. Set it
  lower when another agent shares the host; watch `free -g` during the first build. The
  build directory under the clone is reused, so later builds recompile only changed TUs.
- AOT list: restrict to this part's target from the capability record; the stock wheel
  compiles for several parts and that time buys nothing here.
- `setuptools`, `wheel`, `cmake`, `ninja` are already in the venv; `build` and `pip` are
  not. `bdist_wheel` needs neither. The venv's `setuptools` is newer than the project's
  build pin (§12.9); a failure that names it is that mismatch.
- No install happens. The unpacked wheel is an **overlay** selected per process with
  `PYTHONPATH=tmp/provider-builds/<id>`, which shadows site-packages.

### 7.3 Prove which kernel runs — never assume

Run each in the arm being measured (same env, same `PYTHONPATH`):

| Proof | Command | Stock wheel says | Rebuilt overlay says |
| --- | --- | --- | --- |
| loaded `.so` | `python -c "import vllm_xpu_kernels._C as m; print(m.__file__)"` | `.../site-packages/vllm_xpu_kernels/_C.abi3.so` | `tmp/provider-builds/<id>/vllm_xpu_kernels/_C.abi3.so` |
| registration | `python -c "import torch, vllm_xpu_kernels._C; print(torch._C._dispatch_dump('_C::<op>'))"` | `registered at /workspace/vllm_xpu_kernel/csrc/torch_bindings.cpp` (the CI path) | `registered at <this clone>/csrc/torch_bindings.cpp` |
| kernel identity | unitrace `-d` kernel name | stock functor name | rename the functor in the patch (suffix) and the name in the Device Timing table is the proof |
| inside the vLLM worker | `VLLM_ENABLE_V1_MULTIPROCESSING=0` for the proof run, or `grep _C.abi3.so /proc/<worker pid>/maps` | site-packages path | overlay path |
| pure-Python edit (stack checkout) | `python -c "import vllm, subprocess; print(vllm.__file__); print(subprocess.run(['git','-C','<root>','rev-parse','HEAD'],capture_output=True,text=True).stdout)"` plus the dirty flag | original commit, clean | same commit, dirty, and the `git diff` equals `patch.diff` |

All applicable proofs must agree, and the §2.2 record from inside the arm must match the
one taken at Stage 0 for every component not under test, before a number from that arm is
recorded.

### 7.4 The rebuilt-unpatched control

Build the clone **unmodified** once (`tmp/provider-builds/stock-rebuild/`). The clone's
commit and the wheel's release differ, and so may the compiler. Three arms, in order:

| Arm | Measures |
| --- | --- |
| stock wheel | what runs today; the deploy baseline |
| unpatched rebuild | drift from source revision and toolchain — not your work |
| patched rebuild | your work = patched vs unpatched; deploy value = patched vs stock |

Per-kernel comparison across arms is device-side: the Stage 2 unitrace runner under each
`PYTHONPATH`, same call count, average device time per kernel, repeated alternating.
Two builds of one `torch.ops` symbol cannot coexist in a process, which is why the in-loop
comparison used the control port and this one uses processes (§12.4 proposes closing that).

Gate: patched beats unpatched outside the run-to-run spread, and the patched-vs-unpatched
gain is consistent with the §6.3 kernel contribution. A gain that appears only against the
stock wheel and not against the unpatched rebuild is drift, and is reported as such.

Failure modes: a build that succeeds but the registration still says `/workspace/…` means
`PYTHONPATH` did not reach the process (vLLM worker spawned without it, or the overlay
directory is the wheel file rather than its unpacked contents). `ImportError` on `_C` with a
message about `libsycl` means the oneAPI runtime is not on the loader path for that process;
the venv carries `intel_sycl_rt` — check the interpreter, not the build.

## 8. Stage 6 — Serving A/B

```bash
python scripts/measure_serving_win.py --model <repo_id> --repeats <n> --prompts <n> --out-tokens <n> \
    --env PYTHONPATH=$PWD/tmp/provider-builds/<id> --json tmp/serving-win/<id>.json
```

Arms must differ **only** in the overlay. The script's patched arm today also installs the
`apply()` integration (`FIB_VLLM_INTEGRATION`, `FIB_ENABLE_APPLY`), which puts interception
in the path and contaminates a provider A/B — §12.6 names the flag to add. Until it exists,
run the script's child arm (`--_run-arm`) by hand, alternating, with and without the
`PYTHONPATH` overlay and no `FIB_*` variables set in either, and compute median and spread
from the `FIB_RESULT` lines yourself; say that you did.

Gate, all four:

| Check | Pass |
| --- | --- |
| comparable | the §2.2 records printed from inside each arm's worker are identical for the stack and differ only in the provider entry under test; otherwise the run is refused, not reported |
| throughput | `delta` outside the reported spread, on a window long enough that doubling it does not change the sign |
| correctness | greedy token ids per arm agree (digest per arm; §12.6 adds the print). A divergence is a correctness finding to explain before any number is kept |
| identity | §7.3 proofs hold inside the worker of the patched arm |

A regression here with a per-kernel win in Stage 5 is a real result: the shapes serving
presents are not the harness shape. Return to Stage 1 with the serving `discovered.json`
at that batch, not to the kernel.

## 9. Stage 7 — Promote or revert

Promotion is an install; the owner approves it explicitly, per build.

```bash
# snapshot first, once
SP=/home/sand/Projects/vllm-xpu-venv/lib/python3.12/site-packages
tar -C $SP -czf tmp/provider-builds/stock-wheel.tgz vllm_xpu_kernels vllm_xpu_kernels-*.dist-info
# pip is absent from this venv; bootstrapping it is itself an install the owner approves
python -m ensurepip
python -m pip install --no-deps --no-index --force-reinstall tmp/vllm-xpu-kernels/dist/<wheel>.whl
```

`--no-deps --no-index` is what keeps the resolver away from torch. Revert:

```bash
python -m pip uninstall -y vllm-xpu-kernels && tar -C $SP -xzf tmp/provider-builds/stock-wheel.tgz
```

then re-run the §7.3 proofs and expect the stock answers. Record the result as a trace
(`flashinfer-bench run --save-results` on the definition if one exists) or, for a patch with
no definition, the `patch.diff`, the three-arm table and the serving JSON together. Offer
the diff upstream to the provider; a source patch that lives only in `tmp/` is lost on the
next `/clone-repos`.

## 10. The gate, mechanically

| Gate | Enforced by | What passes | What a failure means |
| --- | --- | --- | --- |
| harness is production | `harness_from_model.py verify` + unitrace name match | shape, dtype, finite, same kernel name | you would be tuning a different function |
| correctness | `kernel_trials.py benchmark` exits 1 before timing; compares mutated arguments for in-place ops | `max abs err <= atol + rtol*max` at the dtype's tolerance | fix in place; never loosen tolerance to pass |
| faster than production | `benchmark` ratio with spread; `best` selects max among correct | `speedup - 1 > spread` | inside spread = unmeasured, not a win; below 1 = a regression the tree records and never promotes |
| win is the kernel's | §6.3 control port | `best` beats control outside spread | call-path artefact; no patch |
| worklist valid for this stack | §3, §2.2 | `discovered.json` provenance equals the current stack provenance | branch switch or edit since discovery; Stage 1 again |
| arms comparable | §2.2 records from inside each arm | stack entries identical; provider entries differ only in the component under test | non-comparable: refused, not reported — a stack change would be booked to a kernel |
| patched code is running | §7.3 proofs | all applicable proofs agree | the number belongs to the stock code |
| win survives rebuild | §7.4 three arms | patched > unpatched > spread | drift, or the patch did not compile in |
| win survives serving | §8 four checks | delta > spread, digests equal, identity holds | shape mismatch or correctness; back to Stage 4 or 1 |
| nothing installed unasked | §0, §9 | owner's explicit approval per install | procedure violation regardless of result |

`kernel_trials.py finalize` today copies `best` even when `best` is a regression; until
§12.5 lands, the operator refuses to run `finalize` unless `best` clears the third row.

## 11. Who decides what

| Decision | Agent | Operator / owner | Machinery |
| --- | --- | --- | --- |
| which op to work on next | no — reads the worklist order | may reorder with a stated reason | Stage 3 ranking |
| hypothesis and code change per trial | **yes** | — | — |
| whether a number counts | no | no | `benchmark` (spread), unitrace (identity) |
| tolerances | no | only by changing the definition's dtype rule | `benchmark --atol/--rtol` defaults per dtype |
| skip the control port | no | no | Stage 4 gate |
| `K`, trial budget | no — reads them | sets them at `init` | — |
| close a row | yes, when §6.5 fires, with the reason recorded | may close earlier | — |
| rebuild the provider | yes (no install) | — | §7.2 |
| switch the stack's branch | no — only on instruction; then re-records provenance and re-runs Stage 1 | decides | §7.1a |
| run `sudo`, install anything, `ensurepip`, promote | **never** | approves each | §9 |
| declare a deployment win | no | reads §8 and §10 | `measure_serving_win.py` |

## 12. Gaps and conflicts with the owner's stated order — for the owner to decide

1. **"Patch in place" depends on the installation shape, which differs per component and
   per box.** Detected here: the stack is an editable, pure-Python checkout (edit or branch
   switch, no build); the provider is a wheel with a source clone at another revision
   (overlay build, §7.2, or an owner-approved install, §9). The plan detects the shape
   (§2.1) rather than assuming either layout, so a box with the provider editable and the
   stack pinned follows the same text.
2. **The dominant GEMM is unreachable by the provider patch.** `F.linear` runs the oneDNN
   bundled inside torch, not the provider and not `FIB_ONEDNN_DIR`. Call-level fixes go
   into vLLM's Python or a load-time weight transform. A rebuilt oneDNN (Fix 5) affects only
   SYCL solutions unless torch's copy is shadowed on the loader path — an experiment the
   owner should sanction separately, not a step here.
3. **unitrace's place.** The owner puts it at "find where they are". Existing machinery
   takes share from torch.profiler because unitrace on this driver returned only a summary;
   the cause is the sync call. The plan uses unitrace for identity and Kernel Properties and
   keeps share in `discovered.json`; both are run, and a summary-only report is a procedure
   error, not evidence.
4. **`kernel_trials.py` cannot A/B two builds of one `torch.ops` op in a process.** Hence the
   control port in-loop and unitrace per-process for the rebuild. Proposed: an `ab`
   subcommand that launches the two arms as subprocesses with per-arm env, interleaved, and
   reports through the same gate.
5. **`finalize` does not refuse a losing `best`.** Proposed: `finalize --require-win`
   refusing when `speedup - 1 <= spread`, default on.
6. **`measure_serving_win.py` has no plain arm.** Its patched arm always enables the
   `apply()` integration. Proposed: a `--plain-arm` mode where `--env` is the only
   difference between arms, and a per-arm greedy token digest printed beside `FIB_RESULT`.
7. **Doc mismatch.** `PROVENANCE.md` and `discover-model-kernels` Step 4 print
   `kernel_trials.py init --harness <path>`; the CLI is `init <name> <baseline.py>`.
8. **VTune is absent** and requires an install plus sysctls the owner performs. §6.4 stays
   optional; no stage depends on it.
9. **Provider build pin.** The project pins `setuptools<80`; the venv has a newer one and no
   `pip`/`build`. `setup.py bdist_wheel` is the no-install path; if it fails on the pin, the
   fix is a `--no-deps --no-index` install of a pinned setuptools — owner approval again.
10. **Fusion is not per-kernel.** The owner's step 3 is per kernel; the highest-ceiling rows
    on a dense transformer are edges (GEMM → elementwise), which replace *two* launches and
    pay no substitution. Stage 3 admits them as rows; the loop form is `xe-fuse.md`.
11. **The clone has no tags.** `setuptools_scm` will stamp a dev version on the rebuilt
    wheel; do not use the version string as identity — use the `.so` path and the
    registration path (§7.3).

## Sources

- `scripts/harness_from_model.py`, `scripts/pull_kernel_source.py`, `scripts/fusion_candidates.py`, `scripts/kernel_trials.py`, `scripts/measure_serving_win.py`, `scripts/calibrate_part.py`
- `flashinfer_bench/device/calibration.py`, `flashinfer_bench/agents/unitrace.py`
- `tools/kernel-harness/sycl_harness.py`, `tools/kernel-harness/trials/fusedadd_control_port.py`, `tools/kernel-harness/optimized/patch_unified_attention.py`, `tools/kernel-harness/knowledge/README.md`
- `.claude/skills/discover-model-kernels`, `wrap-kernel-for-tuning`, `optimize-intel-kernels` (+ `xe-fuse.md`, `architectures.md`, `xe-matrix.md`), `optimize-onednn`, `profile-intel`, `measure-serving-win`, `route-kernel-work/PLAN.md`, `RUN.md`
- `tmp/vllm-xpu-kernels/setup.py`, `tools/envs.py`, `CMakeLists.txt` (`MAX_JOBS`, `VLLM_XPU_AOT_DEVICES`)
