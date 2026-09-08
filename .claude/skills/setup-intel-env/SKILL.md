---
name: setup-intel-env
description: Bring an Intel GPU box from nothing to running flashinfer-bench — driver, PyTorch XPU, oneAPI DPC++, kernel providers (vllm-xpu-kernels, sgl-kernel-xpu, oneDNN, Xe-Fuse), unitrace, and optionally vLLM XPU. Says what each package offers, where it lives, and what installing it costs. Use before any Intel onboarding, optimization or benchmarking work.
---

# Set up an Intel GPU box

Everything below is verifiable: each step ends in a command that either prints the expected
value or tells you what is missing. Do them in order — later steps assume earlier ones.

Ask before installing system packages (oneAPI, drivers); the Python-level steps are safe.

## Step 1: GPU and driver

```bash
xpu-smi discovery 2>/dev/null || ls /dev/dri/
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.device_count())"
python -c "import torch; p=torch.xpu.get_device_properties(0); print(p.name, 'IP', p.version)"
```

The IP version decides what you can build — see `optimize-intel-kernels/architectures.md`
for the parts table (IP 20 = Battlemage `bmg`, 30 = Xe3.0 integrated, 35 = Crescent Island
`cri`).

If `torch.xpu.is_available()` is False, stop here — nothing downstream works. See
`docs/start/hardware-support.mdx`.

## Step 2: PyTorch with XPU support

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/xpu
```

Use the same environment manager throughout. Mixing pip and uv in one venv leaves `pip`
absent, which makes `pip show` report packages as missing when they are installed — verify
with `python -c "import <module>"`, never with `pip show`.

### `uv run` will silently convert this box to CUDA

`pyproject.toml` pins plain `torch>=2.8.0` with no index override, so **`uv run` re-syncs
the venv and replaces the XPU wheels with CUDA ones.** Observed: `torch 2.14.0+xpu` →
`torch 2.14.0` (`torch.xpu.is_available()` False), plus upstream `triton` installed
alongside `triton-xpu` — and since they share the `triton` package namespace, the CUDA
`libtriton.so` wins and `triton.backends.intel` stops importing. Every Triton solution then
reports `COMPILE_ERROR` with no mention of torch, which is a long way from the cause.

Use `uv run --no-sync`, or activate the venv and call tools directly. Activating also puts
`.venv/bin` on `PATH`, which the SYCL builder needs — it shells out to `ninja`, and
`.venv/bin/python -m ...` does not provide it (`[Errno 2] ... 'ninja'`).

To repair a venv this has already hit:

```bash
uv pip install --index-url https://download.pytorch.org/whl/xpu "torch==<ver>+xpu"
uv pip uninstall triton                     # removes the shared triton/_C/ ...
uv pip install --force-reinstall --index-url https://download.pytorch.org/whl/xpu \
    "triton-xpu==<ver>"                     # ... so triton-xpu must be reinstalled after
```

Check both together, since fixing one does not fix the other:

```bash
python -c "import torch, triton; print(torch.__version__, torch.xpu.is_available(), list(triton.backends.backends))"
# want: 2.14.0+xpu True ['amd', 'intel', 'nvidia']
```

## Step 3: oneAPI DPC++ (the SYCL compiler)

Needed to build any SYCL solution. If oneAPI is installed but not on PATH:

```bash
source /opt/intel/oneapi/setvars.sh
# or point the builder straight at it:
export FIB_SYCL_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx

python -c "from flashinfer_bench.compile.builders import SyclBuilder; print(SyclBuilder.is_available())"
```

Must print `True` before Step 4's source builds or any SYCL work.

## Step 4: Kernel providers

These supply the baselines you benchmark against and the libraries your kernels link. The
live inventory — what is installed, at which version, what each offers, and what installing
it costs — is a command, not a table to maintain:

```bash
flashinfer-bench providers list
```

| Provider | What it offers | Kind / cost | Source |
| --- | --- | --- | --- |
| `vllm-xpu` | vLLM's Intel kernels: norms, RoPE, activations, quantization, KV cache | **wheel**, ~81 MB, seconds, no compiler needed | [vllm-project/vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels) |
| `sgl-kernel-xpu` | SGLang's Intel kernels: FMHA, MLA, GroupGemm, low-bit GEMM, GdnAttn | **source**, tens of minutes, multi-GiB | [sgl-project/sgl-kernel-xpu](https://github.com/sgl-project/sgl-kernel-xpu) |
| `onednn` | GEMM, conv, and the post-op mechanism used to fuse epilogues | **system**, ships with oneAPI | `/opt/intel/oneapi/dnnl/latest` |
| `xe-fuse` | GEMM epilogue fusion on CUTLASS-SYCL (not stable, per IntelLabs) | **checkout**, ~25 s per generated kernel | [IntelLabs/Xe-Fuse](https://github.com/IntelLabs/Xe-Fuse) |
| `sycl-tla` | CUTLASS with SYCL bindings; required by Xe-Fuse | **checkout**, clone only | [intel/sycl-tla](https://github.com/intel/sycl-tla) |

```bash
flashinfer-bench providers install vllm-xpu
flashinfer-bench providers install sgl-kernel-xpu --target bmg   # or --device xpu:0
flashinfer-bench providers install <name> --dry-run              # print the command only
```

`install` only handles the `wheel` and `source` kinds. The two **checkout** providers are
used from a source tree and have no install step -- asking for one is an error, not a
build:

```bash
git clone https://github.com/IntelLabs/Xe-Fuse tmp/Xe-Fuse
git clone https://github.com/intel/sycl-tla tmp/sycl-tla
export FIB_XE_FUSE_DIR=$PWD/tmp/Xe-Fuse FIB_SYCL_TLA_DIR=$PWD/tmp/sycl-tla
```

`/clone-repos` does both clones; the env vars are what makes them findable.
`flashinfer-bench providers verify` confirms each one resolves.

Only `sgl-kernel-xpu` is a real source build — cap parallelism there if the host is small;
its build has an OOM guard that aborts rather than thrash. `vllm-xpu` is a prebuilt
`cp38-abi3` wheel and needs no compiler at all.

`sgl-kernel-xpu` builds per architecture and supports `bmg` and `cri` only. On an integrated
Xe3.0 part it cannot be installed; use `vllm-xpu` plus in-tree kernels
(`flashinfer-bench add-baselines --in-tree`).

**Installed is not the same as built.** A partially-built package imports fine and then
fails when a kernel is called. Prove it by calling:

```bash
flashinfer-bench providers verify --local tmp/flashinfer-trace --device xpu:0
```

## Step 4a: Choosing a oneDNN version

Three oneDNN builds are usually in play and they are rarely the same:

| Which | Selected by | Used for |
| --- | --- | --- |
| oneAPI's install | `FIB_ONEDNN_DIR` default | what SYCL solutions link against |
| torch's bundled copy | fixed by the torch wheel | what `torch.matmul` executes — i.e. a GEMM definition's `reference` |
| an upstream release you build | `FIB_ONEDNN_DIR` | new capability |

Check what you actually have; they differ more often than not:

```bash
python -c "from flashinfer_bench.integration.providers import \
    onednn_link_version, onednn_runtime_version; \
    print('link   :', onednn_link_version()); print('runtime:', onednn_runtime_version())"
```

A mismatch means a solution and the reference it is measured against use different oneDNN
builds, so the ratio is cross-library rather than a kernel result. It is recorded in every
trace as `env:onednn_version_mismatch` and logged as a warning.

To build and select a specific release:

```bash
python scripts/build_onednn.py --list                  # newest stable tags
python scripts/build_onednn.py --version v3.13.2       # ~15 min, own prefix
export FIB_ONEDNN_DIR=$HOME/.cache/flashinfer_bench/onednn/v3.13.2
```

Versions install side by side and the oneAPI tree is never touched. The build enables
`ONEDNN_BUILD_GRAPH=ON`, which matters: the Graph API carries the `gated_mlp` fusion pattern.

**Newer is not automatically the right choice.** v3.13 adds an optimized `gated_mlp`
(`micro_horz`) where v3.12 has only a reference one — real new capability. But pointing
`FIB_ONEDNN_DIR` at a version *further* from torch's widens the mismatch above. Match torch's
version for clean comparisons; take a newer one for capability, and let the trace record it.

## Step 4b: Source checkouts for diagnosis

Installing a provider gives you the binary; reading its source is a separate step and some
skills depend on it. `/clone-repos` places all of these under `tmp/`:

| Checkout | Needed for |
| --- | --- |
| `tmp/oneDNN` (pin to the running version) | resolving a dispatch rejection to the gate that caused it |
| `tmp/sgl-kernel-xpu` | the 108 registered ops and their C++ signatures |
| `tmp/vllm-xpu-kernels` | the 49 torch custom-op schemas |
| `tmp/Xe-Fuse`, `tmp/sycl-tla` | **required at build time** — set `FIB_XE_FUSE_DIR` and `FIB_SYCL_TLA_DIR` |

## Step 5: unitrace (profiler)

Intel's counterpart to Nsight Compute, from [intel/pti-gpu](https://github.com/intel/pti-gpu).
It gives device-side kernel timing and the Kernel Properties section that reports register
spill — which you need before trusting any SYCL benchmark.

Not on PyPI; build from source. `flashinfer_bench/agents/unitrace.py` documents the build
and wraps invocation. Two flags that matter: `-d` for device timing, and
`--chrome-kernel-logging` for a timeline.

## Step 6: vLLM XPU (only to benchmark a live server)

Not needed for kernel work. It pins a different torch version, so install it in its **own**
virtualenv rather than alongside flashinfer-bench. The integration lives in
`flashinfer_bench/integration/vllm/` and is inert unless `FIB_VLLM_INTEGRATION=1`.

Upstream instructions: vLLM's `docs/getting_started/installation/gpu.md`, XPU section.

## Step 7: Verify the whole chain

```bash
python -c "from flashinfer_bench.device import list_devices; print(list_devices())"          # ['xpu:0']
python -c "from flashinfer_bench.compile.builders import SyclBuilder; print(SyclBuilder.is_available())"
python -c "from flashinfer_bench.device import get_accelerator; print(get_accelerator('xpu:0').capabilities('xpu:0'))"
flashinfer-bench providers list
flashinfer-bench providers verify --local tmp/flashinfer-trace --device xpu:0
powerprofilesctl set performance     # measuring under power-save invalidates results
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
| `FIB_CACHE_PATH` | Build cache root |
| `FIB_ENABLE_APPLY` | Turn on runtime kernel substitution |
| `FIB_ENABLE_TRACING` | Turn on workload tracing |
| `FIB_VLLM_INTEGRATION` | Install the vLLM adapters in every interpreter |
| `FIB_VLLM_MLP_FUSION` | Opt in to fused-MLP substitution (off by default) |
| `FIB_VLLM_MLP_MIN_TOKENS` | Token threshold above which the fused MLP is used |

`flashinfer_bench/compile/builders/sycl_builder.py` is authoritative for the build ones.

## Failure table

| Symptom | Cause | Action |
| --- | --- | --- |
| `torch.xpu.is_available()` False | Driver or wrong torch build | `docs/start/hardware-support.mdx`; stop until fixed |
| `SyclBuilder.is_available()` False | `icpx` not on PATH | `source setvars.sh` or set `FIB_SYCL_COMPILER` |
| `cannot find -ldnnl` | oneDNN not discoverable | `export FIB_ONEDNN_DIR=/opt/intel/oneapi/dnnl/latest` |
| `pip show` says a package is missing but it imports | pip absent in a uv venv | Trust the import; use `providers list` |
| `sgl-kernel-xpu` build aborts | Host memory | Reduce build parallelism, add swap, or build elsewhere |
| `sgl-kernel-xpu` refuses to build | Part is not `bmg`/`cri` | Use `vllm-xpu` and `--in-tree` baselines |
| Provider imports but a kernel call fails | Partial build | `providers verify` — treat as not installed |
| Benchmarks vary 40%+ run to run | Power-saving profile | `powerprofilesctl set performance` |
| Wrong backend chosen on a mixed host | Auto-detection | `export FIB_DEVICE_BACKEND=xpu` |

## Sources

- `flashinfer_bench/integration/providers.py` — `SPECS` is the authority on every provider
- `docs/start/hardware-support.mdx` — driver setup, tolerances, timing methodology
- `optimize-intel-kernels/architectures.md` — parts table and per-part capabilities
- `onboard-model-intel/providers.md` — per-provider kernel inventories and signatures
