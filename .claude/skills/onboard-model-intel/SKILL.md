---
name: onboard-model-intel
description: End-to-end pipeline for supporting a new LLM on Intel GPUs — definitions acquired on Intel without CUDA, references cross-validated on xpu:0, kernels sourced from oneDNN / vllm-xpu-kernels / sgl-kernel-xpu / Xe-Fuse / hand-written SYCL, and provider bugs triaged. Use when someone says "support model X on Intel", or when an Intel deployment is slow or wrong and you need to find out whose fault it is.
---

# Onboard a model on Intel

Take a model nobody has run here, end up with definitions in the dataset, references proven
correct on `xpu:0`, a profile that says where the time goes, and a benchmarked solution for
the ops that matter.

Everything works on an Intel-only box. No CUDA anywhere in this path.

| Phase | Produces | Skip when |
| --- | --- | --- |
| 0 Bring up the box | a working toolchain | already set up |
| 1 Inventory | which definitions this model needs, and which exist | — |
| 2 Acquire definitions | definition JSON + workloads | all definitions already exist |
| 3 Cross-validate | references proven on `xpu:0` | **never skip** |
| 4 Profile and rank | ordered list of what is worth optimizing | you already know the target |
| 5 Source solutions | baselines + your own solution | — |
| 6 Triage | whose fault a slow or wrong op is | nothing is wrong |
| 7 Benchmark and publish | traces + PRs | — |

Phases 0-3 are the zero-day path: they establish that the model *runs correctly* on Intel
and can be measured, before any kernel work.

## Phase 0: Bring up the box

Run `/setup-intel-env`. It covers the driver, PyTorch XPU, oneAPI DPC++, providers,
unitrace, and the environment variables, each with a verification command.

The three checks this phase depends on:

```bash
python -c "from flashinfer_bench.device import list_devices; print(list_devices())"   # ['xpu:0']
python -c "from flashinfer_bench.compile.builders import SyclBuilder; print(SyclBuilder.is_available())"
flashinfer-bench providers list
```

Also run `/clone-repos` for `tmp/flashinfer-trace` (the dataset) and the upstream sources you
will grep in Phase 5.

## Phase 1: Inventory the model's kernels

Read `config.json` and work out which definitions the model needs and which already exist:

```bash
python -c "
import json, urllib.request
c = json.load(urllib.request.urlopen('https://huggingface.co/<repo_id>/raw/main/config.json'))
print('architectures', c.get('architectures'))
# Multimodal configs (*ForConditionalGeneration) nest the language model under text_config.
# Reading only top-level keys returns nothing for them, silently.
t = c.get('text_config', c)
for k in ('hidden_size','intermediate_size','moe_intermediate_size','num_attention_heads',
          'num_key_value_heads','head_dim','num_hidden_layers','vocab_size',
          'num_experts','num_experts_per_tok','torch_dtype','dtype'):
    if k in t: print(f'{k:24} {t[k]}')
if 'vision_config' in c:
    print('NOTE: vision tower present -- scope it explicitly; this pipeline covers text ops')
"
ls tmp/flashinfer-trace/definitions/*/ | grep -E "h<hidden_size>|d<head_dim>"
```

Naming formulas per op_type are owned by `/extract-kernel-definitions` (section B2). Use
`/discover-models` for the full classification when the model is new to the project.

Record, per definition: does it exist in the dataset, does it have workloads, and which
provider is likely to supply a baseline (Phase 5 table). That list is the plan for the rest
of the run.

## Phase 2: Acquire definitions

**Check first.** Definitions are hardware-agnostic — one that already exists is reused
unchanged, never re-derived:

```bash
ls tmp/flashinfer-trace/definitions/*/<name>.json
```

### Path C (primary on Intel): module hooks on a live `transformers` run

Runs the real model on `xpu:0` and records the shapes that actually executed.

```bash
python scripts/extract_model_kernels_xpu.py \
    --model <repo_id> --output tmp/extract-<slug> \
    --device xpu:0 --max-new-tokens 24 --top-linear 8
```

`--dtype` overrides the extraction dtype; **by default it reads the model config's own**,
which is what you want. A definition extracted in the wrong precision collides by name with
the right one and there is nothing in the name to tell them apart — see the dtype trap in
`../optimize-intel-kernels/architectures.md`.

The script emits dataset conventions directly — `rmsnorm_h{H}` with
`(hidden_states, weight) -> output`, `gemm_n{N}_k{K}` with `(A, B) -> C` on axes M/N/K — so
no renaming is needed. Verify before staging rather than assuming either way:

```bash
python -c "
import json,sys,pathlib
for f in sorted(pathlib.Path(sys.argv[1]).rglob('*.json')):
    d=json.load(open(f)); print(d['name'], d['op_type'], list(d['inputs']), '->', list(d['outputs']))
" tmp/extract-<slug>/definitions
```

Then move each file to `tmp/flashinfer-trace/definitions/{op_type}/` — the `op_type` field
inside the JSON decides the subdirectory. **Check for an existing definition of the same
name first**; an existing one is reused, never overwritten.

Path C emits workloads alongside its definitions, at TP=1 shapes.

### Path B (fallback): transcribe from sources

For kernels Path C does not reach (attention, MoE, KV cache) or for TP/EP variants. The
model's HuggingFace modeling file is the ground truth for what the op computes; `config.json`
plus the TP/EP rules give the constant axes. Procedure and naming: `/extract-kernel-definitions`
section B.

### Path A (optional): the CUDA harvest

If an NVIDIA box happens to be available, `/onboard-model` Phase 2 harvests definitions
faster. Not required, and nothing downstream depends on it.

### Writing the `reference`

Plain PyTorch, float32 accumulation, constants asserted, epsilon as a module-level `EPS`.
Copy the nearest sibling definition rather than starting blank.

## Phase 3: Cross-validate the references on `xpu:0`

**Never skip this.** Correctness is judged against the reference *running on this device*, so
a reference that misbehaves on XPU silently validates a wrong kernel.

```bash
flashinfer-bench validate-references --local tmp/flashinfer-trace \
    --device xpu:0 --definitions <name>
```

| Status | Meaning | Action |
| --- | --- | --- |
| `PASSED` | Agrees across devices | Continue |
| `MISMATCH` | The PyTorch reference itself differs on XPU | Reduce to a minimal repro, run `run()` on `cpu` and `xpu:0`, `torch.testing.assert_close`, report to pytorch/pytorch with `torch.__version__` and the driver version. Do not work around it |
| `UNSUPPORTED_DTYPE` | The part lacks a dtype the definition needs | Pick a different target; Battlemage has no FP8 |
| `NO_WORKLOAD` | Nothing to run it on | Attach a workload (Phase 2) |
| `TARGET_ERROR` | The reference uses an op with no XPU implementation | Rewrite it in primitive ops and re-run |
| `BUILD_ERROR` / `BASELINE_ERROR` | A solution or baseline failed to build | Phase 5; not a reference problem |

## Phase 4: Profile, then rank

Run **`/profile-intel`** — it ranks families by recoverable time and routes each to the
skill that fixes it.

Then run **`/find-kernel-gaps`**, which answers the question the router cannot: what is
burning time that *no definition covers*. On a model whose architecture is not a plain
transformer this is where the large wins are, and they are usually not new kernels. On
Zamba2-1.2B it found a single `aten::sum` at 27.8% of device time that was a batched GEMM
written as broadcast-multiply-then-sum; rewriting it was 107x with no kernel written.

Do both before optimising anything. The rest of this phase is how to read the output.

```bash
unitrace --device-timing --chrome-kernel-logging \
    python scripts/extract_model_kernels_xpu.py \
        --model <repo_id> --output tmp/profile-scratch --device xpu:0 --max-new-tokens 64
```

`--output` is required and must be a directory the script can create: it calls
`mkdir(parents=True, exist_ok=True)` on it, so `/dev/null` raises `FileExistsError` before
any profiling happens. The definitions written there are a by-product of this pass -- the
profile is what you came for -- but they are the same ones Path A produces, so keep them.

Aggregate kernel names into op families and rank by share of device time. Rank by
**share × expected speedup**, not by either alone: a 3x win on 2% of time is worth less than
a 1.15x win on 50%.

Two validity gates:

- Confirm the run does what you think — greedy vs sampled decode changes the mix.
- A `transformers` profile **cannot** rank attention or KV-cache kernels the way a serving
  stack would. For those, profile vLLM-XPU or SGLang-XPU instead.

`flashinfer_bench/agents/unitrace.py` wraps invocation and documents the build.

## Phase 5: Source the solution

### Decision table

| op_type family | Try first | Then | Last resort |
| --- | --- | --- | --- |
| `gemm` (dense projections) | **oneDNN** — already the path torch takes | oneDNN **post-ops** for the epilogue | SYCL, only to fix a bad *call* (`/optimize-onednn`) |
| GEMM + norm/activation fusion | **oneDNN post-ops** — gate on M ≥ 2048 | Xe-Fuse | hand-written SYCL |
| `rmsnorm`, `rope`, activations | **`vllm-xpu-kernels`** — broadest elementwise coverage | `sgl-kernel-xpu` | SYCL — winnable, but small; check the deployment traps first |
| `gqa_paged`, `gqa_ragged`, `mla_paged`, `dsa_paged` | **`sgl-kernel-xpu`** — the only Intel attention kernels | torch SDPA as a correctness floor | SYCL — very large effort |
| `moe`, GroupGemm, W4A16/W8A16 | **`sgl-kernel-xpu`** | — | — |
| `gdn`, `mamba_ssu`, any SSD scan | **`/optimize-ssm-scan`** — check for a materialised contraction first | `sgl-kernel-xpu` `gdn_attention` | SYCL |
| quantization, KV-cache ops | **`vllm-xpu-kernels`** | — | — |
| anything else | PyTorch eager reference — correct, slow, honest | — | SYCL |

Cost and risk, which the table deliberately does not encode:

| Source | Cost | Risk |
| --- | --- | --- |
| oneDNN / oneMKL | none — ships with oneAPI | lowest; versioned with the toolchain |
| `vllm-xpu-kernels` | **prebuilt wheel, seconds** | low; plain torch custom ops |
| `sgl-kernel-xpu` | source build, tens of minutes, multi-GiB | medium; `bmg`/`cri` only, no integrated-Xe3 build |
| Xe-Fuse | one TU per kernel | **high — IntelLabs marks it not stable** |
| hand-written SYCL | seconds to build | yours forever |

### Add baselines

Beating PyTorch eager means nothing. The bar is the kernel a real Intel deployment runs.

```bash
# In-tree first — needs no provider at all.
flashinfer-bench add-baselines --local tmp/flashinfer-trace --in-tree --definitions <name>

# Then the vendor kernels.
flashinfer-bench add-baselines --local tmp/flashinfer-trace \
    --providers vllm-xpu,sgl-kernel-xpu --definitions <name>

# Prove they are built — an import is not proof.
flashinfer-bench providers verify --local tmp/flashinfer-trace --device xpu:0
```

### Wiring a new upstream kernel in as a baseline

The main extension point, and the place where a small mistake produces silence rather than an
error. `flashinfer_bench/integration/xpu_kernels.py` holds a `REGISTRY` of `BaselineKernel`
records. Matching requires **all three** to hold exactly:

```python
definition.op_type == kernel.op_type
tuple(definition.inputs)  == kernel.inputs      # names, in declaration order
tuple(definition.outputs) == kernel.outputs
```

Names, not shapes or dtypes. A definition whose input is `hidden_states` does not match a
kernel declaring `x`, and a near-miss produces **no baseline at all** — logged under "No
upstream kernel matches" and skipped. That is deliberate (a wrongly bound baseline would give
a confidently wrong comparison), but it means a typo looks exactly like "not registered yet".

**1. Read the real signature.** Do not infer it from docs.

```python
import torch, vllm_xpu_kernels._C  # noqa: F401
print(torch.ops._C.rms_norm.default._schema)
# _C::rms_norm(Tensor! out, Tensor input, Tensor weight, float epsilon) -> ()

import sgl_kernel, inspect
print(inspect.signature(sgl_kernel.rmsnorm))
```

`Tensor! out` first means destination-passing and in-place.

**2. Check semantics, not arity.** `gemma_rms_norm` takes byte-identical arguments to
`rms_norm` and scales by `(1 + weight)`. Registered against a plain RMSNorm definition it ran
cleanly and computed the wrong thing. A kernel belongs in the registry only when it computes
the same function as the definition's reference.

**3. Write the wrapper** — plain Python source as a string, defining `run(...)` with the
definition's inputs in order. Allocate outputs yourself for destination-passing ops, and
**clone in-place inputs**: the benchmark reuses tensors across trials, so an op that mutates
them makes trial 2 measure different data.

`run(...)` takes the definition's **inputs** and nothing else. Scalars the kernel needs but
the definition does not declare as inputs — epsilon, `is_neox` — are `constants`: the
template is `str.format`-ed with them before it is compiled, so they become module-level
literals, not parameters. Naming one as a `run` argument is the single most common reason a
registry entry silently matches nothing.

```python
_MY_KERNEL_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)

EPS = {eps!r}


def run(hidden_states, weight):          # exactly the definition's inputs, in order
    out = torch.empty_like(hidden_states)
    torch.ops._C.my_kernel(out, hidden_states, weight, EPS)
    return out
"""
```

Then declare what the template needs: `constants=("eps",)` on the registry entry.
`definition_eps()` resolves it from the definition's reference source, warning if the
definition states none rather than guessing silently.

**4. Add the registry entry**, with `inputs`/`outputs` copied from the **definition**, not
from the upstream signature.

**5. Verify it matched.** The failure mode is silence, so never assume:

```python
from flashinfer_bench.data import TraceSet
from flashinfer_bench.integration import available_providers, find_baselines

ts = TraceSet.from_path("tmp/flashinfer-trace")
d = ts.definitions["<name>"]
print("providers:", available_providers())
print("signature:", d.op_type, tuple(d.inputs), "->", tuple(d.outputs))
print("matched:", [k.solution_name for k in find_baselines(d)])
```

Empty `matched` is almost always: op_type spelled differently, an input the definition
declares that the kernel does not take (`eps` as an input vs a constant), or outputs in the
other order. **Fix the entry to match reality — never rename a dataset definition to make a
baseline match.** The definition describes the operation; the registry describes the kernel.

### Writing your own solution

`/optimize-intel-kernels` owns this — SYCL and Triton, with the levers for each.

## Phase 6: Triage a slow or wrong op

### Whose problem is it?

Three suspects — reference, harness, provider — separated by cheap tests **in this order**.
A wrong reference explains every downstream symptom and costs an afternoon if assumed away.

```
1. flashinfer-bench validate-references --local tmp/flashinfer-trace --device xpu:0 --definitions <name>
       MISMATCH -> the reference is wrong on XPU. Not a provider problem. Stop.
       PASSED   -> continue

2. Run the provider kernel and the reference on the SAME workload and compare:
       flashinfer-bench add-baselines --local tmp/flashinfer-trace --definitions <name>
       flashinfer-bench run --local tmp/flashinfer-trace --definitions <name> --num-trials 1
       flashinfer-bench report summary --local tmp/flashinfer-trace
       INCORRECT -> provider correctness bug -> ladder rung 3/4
       PASSED    -> continue

3. Time three things on the same workload:
       (a) the provider kernel   (b) the reference (eager)   (c) a naive SYCL kernel, untuned
       (a) ~ (c), both near bandwidth -> nobody is at fault; the op is bandwidth-bound
       (a) >> (c)                     -> the provider picked a bad implementation -> /optimize-onednn
       (a) << (c) but still slow      -> genuinely hard; needs fusion, not tuning
```

Step 3(c) converts "oneDNN is slow" from an opinion into a bound: copy
`examples/sycl/rmsnorm_sycl.cpp`, change the body, do not tune. Fifteen minutes.

### The escalation ladder

**Rung 1 always exists** — because the Solution layer sits between the definition and any
library, you are never blocked waiting on an upstream fix.

| Rung | Action | Ships | Cost |
| --- | --- | --- | --- |
| 1 | Change the call, not the library: memory layout, fpmath mode, post-ops, splitting one primitive into two. Encoded as a Solution | immediately | hours |
| 2 | Override with a SYCL solution for the shape range where the library loses; a solution competes per workload, so the library keeps the rest | immediately | days |
| 3 | Patch the provider locally, pinned to a commit | your box only | days + carrying cost |
| 4 | Report upstream with a minimal repro built from the definition + workload | months | hours |

Rungs 3 and 4 are not alternatives to 1 and 2 — do 1 or 2 **and** 4.

Nothing in the trace format records "this baseline used a locally patched provider", so if
you patch one, say so in the solution's `description` and the PR body or the number is not
reproducible.

## Phase 7: Benchmark, validate, publish

```bash
flashinfer-bench run --local tmp/flashinfer-trace --definitions <name> --save-results
flashinfer-bench validate --dataset tmp/flashinfer-trace --definitions <name>
```

Traces record `hardware_id` and `environment.libs.timing`, so results group by device and
are never ranked across devices. That is provenance, not a caveat — an Intel speedup and an
NVIDIA speedup are not the same quantity.

Then `/submit-onboarding-prs`.

## Source map

| Question | Read |
| --- | --- |
| What do I install, and what does each package offer? | `/setup-intel-env` |
| What kernels does each provider contain, and how do I read a signature? | `providers.md` |
| Why is this GEMM slow? | `/optimize-onednn` |
| What is hot that nothing covers? | `/find-kernel-gaps` |
| A hybrid/Mamba model is slow | `/optimize-ssm-scan` |
| How do I write a fast SYCL or Triton kernel? | `../optimize-intel-kernels/SKILL.md` |
| What is special about this Intel part? | `../optimize-intel-kernels/architectures.md` |
| GEMM epilogue fusion | `../optimize-intel-kernels/xe-fuse.md` |
| How are in-tree kernels generated per definition? | `flashinfer_bench/integration/intree_kernels.py` — one template per language |
| Definition naming and axis rules | `/extract-kernel-definitions` section B2 |
