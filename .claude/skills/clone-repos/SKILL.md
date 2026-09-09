---
name: clone-repos
description: Clone or update the upstream repositories the other skills read — SGLang, FlashInfer, sgl-cookbook, the flashinfer-trace HuggingFace dataset, and optionally the Intel kernel repos — into tmp/, and record their SHAs. Use when setting up, or before any skill that greps upstream sources.
---

# Clone repositories

Every other skill greps `tmp/`. This puts it there and records what was checked out.

## Clone or update

One loop, not one block per repo. Re-running is safe: existing checkouts are hard-reset to
the branch, missing ones are cloned.

```bash
mkdir -p tmp

# name             url                                                        branch  submodules
read -r -d '' REPOS <<'LIST'
sglang             https://github.com/sgl-project/sglang.git                  main    yes
flashinfer         https://github.com/flashinfer-ai/flashinfer.git            main    yes
sgl-cookbook       https://github.com/sgl-project/sgl-cookbook.git            main    no
LIST

while read -r name url branch subs; do
  [ -z "$name" ] && continue
  if [ -d "tmp/$name/.git" ]; then
    echo "updating $name"
    ( cd "tmp/$name" && git fetch origin && git checkout "$branch" \
      && git reset --hard "origin/$branch" \
      && { [ "$subs" = yes ] && git submodule update --init --recursive || true; } )
  else
    echo "cloning $name"
    if [ "$subs" = yes ]; then
      git clone --recurse-submodules -b "$branch" "$url" "tmp/$name"
    else
      git clone -b "$branch" "$url" "tmp/$name"
    fi
  fi
done <<< "$REPOS"
```

The trace dataset is a **HuggingFace dataset repo**, not GitHub, and needs LFS for the
workload blobs:

```bash
git lfs install
[ -d tmp/flashinfer-trace/.git ] \
  && git -C tmp/flashinfer-trace pull \
  || git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace tmp/flashinfer-trace
git -C tmp/flashinfer-trace lfs pull
```

Everything written by workload collection goes into this clone.

## Intel repositories (optional)

Needed only for Intel work. These are **sources to read** — installing the providers is
`/setup-intel-env`, which uses `flashinfer-bench providers install`. Xe-Fuse and sycl-tla are
the exception: they are consumed as checkouts, so the builder needs their paths.

```bash
git clone --depth 1 https://github.com/sgl-project/sgl-kernel-xpu.git    tmp/sgl-kernel-xpu
git clone --depth 1 https://github.com/vllm-project/vllm-xpu-kernels.git tmp/vllm-xpu-kernels
git clone --depth 1 https://github.com/IntelLabs/Xe-Fuse.git             tmp/Xe-Fuse
git clone --depth 1 https://github.com/intel/sycl-tla.git                tmp/sycl-tla

# Xe-Fuse kernels are compiled from these checkouts, so the builder must be told where:
export FIB_XE_FUSE_DIR=$PWD/tmp/Xe-Fuse
export FIB_SYCL_TLA_DIR=$PWD/tmp/sycl-tla
```

| Repo | Read it for | Skill |
| --- | --- | --- |
| `sgl-kernel-xpu` | registered ops and their C++ signatures | `onboard-model-intel/providers.md` |
| `vllm-xpu-kernels` | torch custom-op schemas | `onboard-model-intel/providers.md` |
| `Xe-Fuse` | `autotune/generate_kernel.py`, presets, EVT vocabulary | `optimize-intel-kernels/xe-fuse.md` |
| `sycl-tla` | CUTLASS-SYCL headers Xe-Fuse compiles against | `optimize-intel-kernels/xe-fuse.md` |
| `oneDNN` | dispatch gates, below | `/optimize-onednn` |

## oneDNN source (optional, for diagnosis)

Needed only to resolve a dispatch rejection to the gate that caused it. **Pin it to the
version you are actually running** — a gate moves between releases, so source from a
different version points at the wrong line.

```bash
V=$(ONEDNN_VERBOSE=1 python -c "
import torch
a=torch.randn(8,64,dtype=torch.bfloat16,device='xpu:0')
torch.matmul(a, a.T[:64]); torch.xpu.synchronize()" 2>&1 |
    grep -oE 'oneDNN v[0-9.]+' | head -1 | sed 's/oneDNN v//')
git clone --depth 1 --branch "v$V" https://github.com/uxlfoundation/oneDNN.git tmp/oneDNN
```

For reading, never for building — the runtime comes from oneAPI. Tuning oneDNN's own GEMM
is not the goal; the wins are in how it is called (`/optimize-onednn`).

## Install from source — only on CUDA

**Skip this section on an Intel-only box.** Building FlashInfer from source requires `nvcc`;
the clone is still useful, because the Intel skills only *read* these trees (to grep
signatures and reference implementations) and never import them.

Both lines are installs: the owner approves them first, and they go into the activated dev
venv (`.venv`) — `CLAUDE.md`, "Python Environments".

```bash
source .venv/bin/activate
( cd tmp/flashinfer && python -m pip install --no-build-isolation -e . -v )
( cd tmp/sglang    && python -m pip install -e python )
```

Subshells keep the working directory unchanged. A uv-managed venv has no `pip` module
(`No module named pip`); bootstrapping one with `python -m ensurepip` is itself an install
the owner approves, and never `uv pip install` in its place. Verify an install with
`python -c "import <module>"` — `pip show` reports installed packages as missing.

## Record the SHAs

`/discover-models` and the PR provenance fields expect these:

```bash
for r in sglang flashinfer sgl-cookbook flashinfer-trace \
         sgl-kernel-xpu vllm-xpu-kernels Xe-Fuse sycl-tla oneDNN; do
  printf "%-18s %s\n" "$r" "$(git -C tmp/$r rev-parse --short HEAD 2>/dev/null || echo missing)"
done
```

## Verify

```bash
ls tmp/sglang/python/sglang/srt/models/     # model implementations
ls tmp/flashinfer/flashinfer/               # package is at repo root, not python/
ls tmp/flashinfer/tests/                    # how each wrapper is really called
ls tmp/sgl-cookbook/data/models/generated/  # serving configs, newest dir wins
ls tmp/flashinfer-trace/definitions/        # the dataset

# CUDA only:
python -c "import sglang, flashinfer; print(sglang.__version__, flashinfer.__version__)"
```

## Failure table

| Symptom | Cause | Action |
| --- | --- | --- |
| Submodule checkout incomplete | Interrupted clone | `git -C tmp/<name> submodule update --init --recursive` |
| Blobs are text pointers, not tensors | LFS not pulled | `git -C tmp/flashinfer-trace lfs pull` |
| HF clone asks for credentials | Not logged in | `hf auth login` |
| FlashInfer build fails on `nvcc` | Intel-only box | Expected — skip the install step; the clone is enough |
| `python -m pip: No module named pip` | uv-managed venv; `pip` is absent by design | Ask the owner to install, naming the package and `.venv`; never `uv pip install` |
