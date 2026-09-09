---
name: setup-intel-env
description: Bring an Intel GPU box from nothing to running flashinfer-bench — driver, PyTorch XPU, oneAPI DPC++, kernel providers (vllm-xpu-kernels, sgl-kernel-xpu, oneDNN, Xe-Fuse), unitrace, and optionally vLLM XPU. Use before any Intel onboarding, optimization or benchmarking work.
---

# Set up an Intel GPU box

Each step ends in a command that prints the expected value or says what is missing. Do them
in order. Installing anything — system packages (oneAPI, drivers) and Python packages alike —
is the owner's action: ask first, naming the package and the environment it goes into
(`CLAUDE.md`, "Python Environments").

## Step 1: GPU and driver

```bash
xpu-smi discovery 2>/dev/null || ls /dev/dri/
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.device_count())"
python -c "import torch; p=torch.xpu.get_device_properties(0); print(p.name, 'IP', p.version)"
```

The IP version decides what you can build — `optimize-intel-kernels/architectures.md`
(IP 20 = Battlemage `bmg`, 30 = Xe3.0 integrated, 35 = Crescent Island `cri`).

If `torch.xpu.is_available()` is False, stop — `docs/start/hardware-support.mdx`.

## Step 2: PyTorch with XPU support

torch-xpu is installed by the owner into the dev venv (`.venv`). This step verifies it and
installs nothing:

```bash
source .venv/bin/activate
python -c "import torch, triton; print(torch.__version__, torch.xpu.is_available(), list(triton.backends.backends))"
# want: <ver>+xpu True ['amd', 'intel', 'nvidia']
```

Any other output — no `+xpu` suffix, `False`, or `intel` missing from the backends — means
the venv never had the XPU wheels or has been re-synced (below). Stop and ask the owner to
install `torch==<ver>+xpu` and `triton-xpu==<ver>` from
`https://download.pytorch.org/whl/xpu` into `.venv`. Neither venv has `pip`, so verify with
`python -c "import <module>"`, never `pip show`.

### Never `uv run` or `uv pip` in these venvs

The rule is in `CLAUDE.md`, "Python Environments"; this is the mechanism. `pyproject.toml`
pins plain `torch` with no index override, so `uv run` (and `uv sync`) re-syncs the venv and
replaces the XPU wheels with CUDA ones, and installs upstream `triton` over `triton-xpu`
(they share the `triton` namespace, so `triton.backends.intel` stops importing and every
Triton solution reports `COMPILE_ERROR`). `uv pip install` resolves a package's
dependencies against PyPI and does the same whenever a requirement is not met by the
installed `+xpu` wheels. Activate the venv and call tools directly; there is no safe flag,
so never `uv run --no-sync` either. Activating also puts `.venv/bin` on `PATH`, which the
SYCL builder needs for `ninja`.

The check above detects the damage. The repair is an install and therefore the owner's:
report the check's output and stop. What the owner reinstalls into `.venv`, in this order:
`torch==<ver>+xpu`; then upstream `triton` removed (it owns the shared `triton/_C/`); then
`triton-xpu==<ver>` — both wheels from `https://download.pytorch.org/whl/xpu`. Re-run the
check afterwards.

## Step 3: oneAPI DPC++ (the SYCL compiler)

```bash
source /opt/intel/oneapi/setvars.sh
# or point the builder straight at it:
export FIB_SYCL_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx

python -c "from flashinfer_bench.compile.builders import SyclBuilder; print(SyclBuilder.is_available())"
```

Must print `True` before Step 4's source builds or any SYCL work.

## Step 4: Kernel providers

The live inventory — installed, version, what each offers, install cost — is a command:

```bash
flashinfer-bench providers list
```

| Provider | What it offers | Kind | Source |
| --- | --- | --- | --- |
| `vllm-xpu` | vLLM's Intel kernels: norms, RoPE, activations, quantization, KV cache | **wheel**, no compiler needed | [vllm-project/vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels) |
| `sgl-kernel-xpu` | SGLang's Intel kernels: FMHA, MLA, GroupGemm, low-bit GEMM, GdnAttn | **source**, long and memory-hungry | [sgl-project/sgl-kernel-xpu](https://github.com/sgl-project/sgl-kernel-xpu) |
| `onednn` | GEMM, conv, and the post-op mechanism used to fuse epilogues | **system**, ships with oneAPI | `/opt/intel/oneapi/dnnl/latest` |
| `xe-fuse` | GEMM epilogue fusion on CUTLASS-SYCL (not stable, per IntelLabs) | **checkout** | [IntelLabs/Xe-Fuse](https://github.com/IntelLabs/Xe-Fuse) |
| `sycl-tla` | CUTLASS with SYCL bindings; required by Xe-Fuse | **checkout** | [intel/sycl-tla](https://github.com/intel/sycl-tla) |

`providers install` is an install, so it is the owner's action. It runs `python -m pip`
in the active interpreter and nothing else by default. On an interpreter without `pip` —
both venvs here — it does not run: it prints `Declined to install <package> into <venv>
…` naming the package, the venv and interpreter, the reason (a uv resolve can replace the
XPU torch with a CUDA build), the manual recipe, and the opt-in, then exits 1. There is no
automatic fallback to uv; `--installer uv` (or `FIB_PROVIDER_INSTALLER=uv`) is the only
route to it, and that is the command `CLAUDE.md`, "Python Environments" forbids an agent to
run. So preview, and hand the command to the owner naming the venv it targets:

```bash
source .venv/bin/activate
flashinfer-bench providers install vllm-xpu --dry-run                       # pip command, or the refusal
flashinfer-bench providers install vllm-xpu --installer uv --dry-run        # the uv command; printed, not run
flashinfer-bench providers install sgl-kernel-xpu --target bmg --dry-run    # or --device xpu:0
```

Without `--dry-run`, `--installer uv` runs uv. Never pass it yourself; it exists so the
owner can make that choice explicitly.

`install` handles the `wheel` and `source` kinds only. The checkout providers are used from
a source tree:

```bash
git clone https://github.com/IntelLabs/Xe-Fuse tmp/Xe-Fuse
git clone https://github.com/intel/sycl-tla tmp/sycl-tla
export FIB_XE_FUSE_DIR=$PWD/tmp/Xe-Fuse FIB_SYCL_TLA_DIR=$PWD/tmp/sycl-tla
```

`/clone-repos` does both clones. `sgl-kernel-xpu` builds per architecture (`bmg`, `cri`
only) and has an OOM guard that aborts rather than thrash — cap parallelism on a small
host. On an integrated Xe3.0 part use `vllm-xpu` plus in-tree kernels
(`flashinfer-bench add-baselines --in-tree`).

**Installed is not the same as built.** Prove it by calling:

```bash
flashinfer-bench providers verify --local tmp/flashinfer-trace --device xpu:0
```

## Step 4a: Choosing a oneDNN version

| Which | Selected by | Used for |
| --- | --- | --- |
| oneAPI's install | `FIB_ONEDNN_DIR` default | what SYCL solutions link against |
| torch's bundled copy | fixed by the torch wheel | what `torch.matmul` executes — i.e. a GEMM definition's `reference` |
| an upstream release you build | `FIB_ONEDNN_DIR` | new capability |

```bash
python -c "from flashinfer_bench.integration.providers import \
    onednn_link_version, onednn_runtime_version; \
    print('link   :', onednn_link_version()); print('runtime:', onednn_runtime_version())"
```

A mismatch means a solution and the reference it is measured against use different oneDNN
builds. It is recorded in every trace as `env:onednn_version_mismatch` and logged. To build
and select a specific release, side by side and without touching the oneAPI tree:

```bash
python scripts/build_onednn.py --list
python scripts/build_onednn.py --version <tag>
export FIB_ONEDNN_DIR=$HOME/.cache/flashinfer_bench/onednn/<tag>
```

The build enables `ONEDNN_BUILD_GRAPH=ON`; the Graph API carries the `gated_mlp` fusion
pattern. Match torch's version for clean comparisons; take a newer one for capability, and
let the trace record it.

## Step 4b: Source checkouts for diagnosis

`/clone-repos` places these under `tmp/`:

| Checkout | Needed for |
| --- | --- |
| `tmp/oneDNN` (pin to the running version) | resolving a dispatch rejection to the gate that caused it |
| `tmp/sgl-kernel-xpu` | registered ops and their C++ signatures |
| `tmp/vllm-xpu-kernels` | torch custom-op schemas |
| `tmp/Xe-Fuse`, `tmp/sycl-tla` | **required at build time** — set `FIB_XE_FUSE_DIR` and `FIB_SYCL_TLA_DIR` |

## Step 5: unitrace (profiler)

From [intel/pti-gpu](https://github.com/intel/pti-gpu); not on PyPI, build from source.
`flashinfer_bench/agents/unitrace.py` documents the build and wraps invocation. It gives
device-side kernel timing and the Kernel Properties section that reports register spill.
Flags: `-d` for device timing, `--chrome-kernel-logging` for a timeline.

## Step 6: vLLM XPU (only to benchmark a live server)

Not needed for kernel work. It pins a different torch version, so install it in its **own**
virtualenv. The integration lives in `flashinfer_bench/integration/vllm/` and is inert
unless `FIB_VLLM_INTEGRATION=1`. Upstream instructions: vLLM's
`docs/getting_started/installation/gpu.md`, XPU section.

## Step 7: Verify the whole chain

```bash
python -c "from flashinfer_bench.device import list_devices; print(list_devices())"          # ['xpu:0']
python -c "from flashinfer_bench.compile.builders import SyclBuilder; print(SyclBuilder.is_available())"
python -c "from flashinfer_bench.device import get_accelerator; print(get_accelerator('xpu:0').capabilities('xpu:0'))"
flashinfer-bench providers list
flashinfer-bench providers verify --local tmp/flashinfer-trace --device xpu:0
powerprofilesctl set performance     # measure under a performance profile, always
python scripts/calibrate_part.py     # per-part constants the deploy gates read
```

## Environment variables

| Variable | Purpose |
| --- | --- |
| `FIB_SYCL_COMPILER` | Path to `icpx` when oneAPI is not on PATH |
| `FIB_ONEDNN_DIR` | oneDNN root, when not at a default location |
| `FIB_XE_FUSE_DIR` | Xe-Fuse checkout |
| `FIB_SYCL_TLA_DIR` | CUTLASS-SYCL checkout |
| `FIB_SYCL_AOT_DEVICE` | Override the AOT device name (`bmg`, `cri`) |
| `FIB_SYCL_LARGE_GRF` | Build with large-register-file mode |
| `FIB_DEVICE_BACKEND` | Force a backend (`xpu`) on a mixed-vendor host |
| `FIB_DATASET_PATH` | Trace dataset root |
| `FIB_CACHE_PATH` | Build and calibration cache root |
| `FIB_ENABLE_APPLY` | Turn on runtime kernel substitution |
| `FIB_ENABLE_TRACING` | Turn on workload tracing |
| `FIB_VLLM_INTEGRATION` | Install the vLLM adapters in every interpreter |
| `FIB_VLLM_MLP_FUSION` | Opt in to fused-MLP substitution (off by default) |
| `FIB_VLLM_MLP_MIN_TOKENS` | Token threshold above which the fused MLP is used |

`flashinfer_bench/compile/builders/sycl_builder.py` is authoritative for the build ones.

## Failure table

| Symptom | Fix |
| --- | --- |
| `torch.xpu.is_available()` False | `docs/start/hardware-support.mdx`; stop until fixed |
| `SyclBuilder.is_available()` False | `source setvars.sh` or set `FIB_SYCL_COMPILER` |
| `cannot find -ldnnl` | `export FIB_ONEDNN_DIR=/opt/intel/oneapi/dnnl/latest` |
| `pip show` says a package is missing but it imports | Trust the import; use `providers list` |
| `torch.__version__` lacks `+xpu`, or `intel` missing from `triton.backends.backends` | The venv was re-synced; the owner reinstalls (the Step 2 check names the wheels); stop until it passes |
| `sgl-kernel-xpu` build aborts | Reduce build parallelism, add swap, or build elsewhere |
| `sgl-kernel-xpu` refuses to build | Part is not `bmg`/`cri`; use `vllm-xpu` and `--in-tree` baselines |
| Provider imports but a kernel call fails | `providers verify`; treat as not installed |
| Benchmarks vary run to run by more than the effect | `powerprofilesctl set performance` |
| Wrong backend chosen on a mixed host | `export FIB_DEVICE_BACKEND=xpu` |

## Sources

- `flashinfer_bench/integration/providers.py` — `SPECS` is the authority on every provider
- `docs/start/hardware-support.mdx` — driver setup, tolerances, timing methodology
- `optimize-intel-kernels/architectures.md` — parts table and per-part capabilities
- `onboard-model-intel/providers.md` — per-provider kernel inventories and signatures
