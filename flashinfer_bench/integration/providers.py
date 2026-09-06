"""Acquiring and compiling the Intel kernel providers.

Five sources of Intel GPU kernels, four genuinely different acquisition mechanisms: a
prebuilt wheel, a source build that must be told which GPU architecture to target, a
header-only checkout pointed at by an environment variable, and a system library that
arrives with oneAPI. Each is a page of instructions in a document somewhere; this module
makes them a table, so ``flashinfer-bench providers install <name>`` works the same way
whichever one you name.

Two things fall out of having the table. Availability stops being a guess -- a provider is
present when its import resolves, and its version comes from installed distribution
metadata rather than a ``__version__`` attribute the package may not define. And the build
configuration stops being something the user has to know: ``sgl-kernel-xpu`` must be
compiled for a specific architecture, and the architecture is already recorded in the
device's capability record, so the install command derives it instead of asking.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """A provider cannot be installed as asked, with the reason."""


@dataclass(frozen=True)
class ProviderSpec:
    """How one kernel source is obtained, detected and reported.

    Parameters
    ----------
    name : str
        Identifier used by the CLI and by ``BaselineKernel.provider``.
    summary : str
        One line on what the provider covers.
    module : Optional[str]
        Import name that proves it is present. ``None`` for providers that are not Python
        packages -- oneDNN and Xe-Fuse are found on disk, not imported.
    distribution : Optional[str]
        Installed distribution name, for reading a version out of package metadata.
    kind : str
        ``wheel``, ``source``, ``checkout`` or ``system``. Determines which acquisition
        path applies and what it costs.
    cost : str
        Honest expectation for how long acquisition takes and what it needs.
    env_var : Optional[str]
        Environment variable that points at an on-disk provider, when ``kind`` is
        ``checkout`` or ``system``.
    search_paths : Tuple[str, ...]
        Where an on-disk provider is looked for when ``env_var`` is unset.
    probe : Optional[str]
        Path relative to a root that proves a real installation rather than an empty
        directory -- a header, because a runtime-only prefix cannot build against.
    repo : Optional[str]
        Upstream source, shown when acquisition is manual.
    needs_sycl_target : bool
        Whether the build must be told the GPU architecture. Derived from the device
        rather than asked for.
    build_requires : Tuple[str, ...]
        Packages the build backend needs present *before* it runs. A source build here
        uses ``--no-build-isolation`` -- required, because the build must link against the
        torch already installed rather than a fresh one pulled into an isolated
        environment -- and that also means nothing installs the backend for it.
    """

    name: str
    summary: str
    kind: str
    cost: str
    module: Optional[str] = None
    distribution: Optional[str] = None
    env_var: Optional[str] = None
    search_paths: Tuple[str, ...] = ()
    probe: Optional[str] = None
    repo: Optional[str] = None
    needs_sycl_target: bool = False
    build_requires: Tuple[str, ...] = ()


SPECS: Tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="vllm-xpu",
        summary="vLLM's Intel kernels: norms, RoPE, activations, quantization, KV cache.",
        kind="wheel",
        cost="~81 MB, seconds. Prebuilt cp38-abi3 wheel; no compiler needed.",
        module="vllm_xpu_kernels",
        distribution="vllm-xpu-kernels",
        repo="https://github.com/vllm-project/vllm-xpu-kernels",
    ),
    ProviderSpec(
        name="sgl-kernel-xpu",
        summary="SGLang's Intel kernels: FMHA, MLA, GroupGemm, low-bit GEMM, GdnAttn.",
        kind="source",
        cost="Tens of minutes, multi-GiB. Hundreds of CUTLASS translation units.",
        module="sgl_kernel",
        # The repo is sgl-kernel-xpu; the distribution it installs is "sgl-kernel". Using
        # the repo name here silently loses the version from trace provenance.
        distribution="sgl-kernel",
        repo="https://github.com/sgl-project/sgl-kernel-xpu",
        needs_sycl_target=True,
        # scikit-build-core is the build backend; with build isolation off it has to be
        # installed first or the build fails with "No module named scikit_build_core".
        build_requires=("scikit-build-core", "cmake", "ninja", "setuptools", "wheel"),
    ),
    ProviderSpec(
        name="onednn",
        summary="oneDNN: GEMM, conv, and the post-op mechanism used to fuse epilogues.",
        kind="system",
        cost="None. Ships with oneAPI.",
        env_var="FIB_ONEDNN_DIR",
        search_paths=("/opt/intel/oneapi/dnnl/latest", "/usr"),
        probe="include/oneapi/dnnl/dnnl.hpp",
    ),
    ProviderSpec(
        name="xe-fuse",
        summary="GEMM epilogue fusion on CUTLASS-SYCL. Marked not-stable by IntelLabs.",
        kind="checkout",
        cost="Clone, then one translation unit (~25 s) per generated kernel.",
        env_var="FIB_XE_FUSE_DIR",
        probe="include/xe-fuse/builder/epilogue_builder.hpp",
        repo="https://github.com/IntelLabs/Xe-Fuse",
    ),
    ProviderSpec(
        name="sycl-tla",
        summary="CUTLASS with SYCL bindings. Required by Xe-Fuse kernels.",
        kind="checkout",
        cost="Clone only; built as part of whatever includes it.",
        env_var="FIB_SYCL_TLA_DIR",
        probe="include/cutlass/cutlass.h",
        repo="https://github.com/intel/sycl-tla",
    ),
)
"""Every kernel source, in the order a new machine should acquire them."""

_BY_NAME: Dict[str, ProviderSpec] = {s.name: s for s in SPECS}


def get_spec(name: str) -> ProviderSpec:
    """The spec for ``name``, or a ``ProviderError`` naming what is available."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ProviderError(
            f"Unknown provider '{name}'. Known: {', '.join(sorted(_BY_NAME))}."
        ) from None


def find_root(spec: ProviderSpec) -> Optional[str]:
    """Where an on-disk provider lives, or ``None``.

    ``env_var`` wins, then the standard prefixes. The probe is what distinguishes a real
    installation from a directory that merely exists -- for oneDNN specifically, a prefix
    carrying only the runtime library will not build.
    """
    if spec.env_var and os.environ.get(spec.env_var):
        root = os.environ[spec.env_var]
        if spec.probe is None or (Path(root) / spec.probe).exists():
            return root
        return None
    for root in spec.search_paths:
        if spec.probe is None or (Path(root) / spec.probe).exists():
            return root
    return None


def provider_version(spec: ProviderSpec) -> Optional[str]:
    """Installed version, from distribution metadata.

    Read from metadata rather than a ``__version__`` attribute: ``vllm_xpu_kernels``
    defines no such attribute but does ship a version in its distribution.
    """
    if not spec.distribution:
        return None
    try:
        return importlib.metadata.version(spec.distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def is_installed(spec: ProviderSpec) -> bool:
    """Whether this provider is usable here."""
    if spec.module:
        try:
            return importlib.util.find_spec(spec.module) is not None
        except (ImportError, ValueError):
            return False
    return find_root(spec) is not None


def status(spec: ProviderSpec) -> Dict[str, Optional[str]]:
    """Everything worth reporting about one provider's presence."""
    installed = is_installed(spec)
    return {
        "name": spec.name,
        "kind": spec.kind,
        "installed": "yes" if installed else "no",
        "version": provider_version(spec) if installed else None,
        "location": find_root(spec) if installed and not spec.module else None,
        "summary": spec.summary,
        "cost": spec.cost,
    }


def all_status() -> List[Dict[str, Optional[str]]]:
    """Presence of every known provider, in acquisition order."""
    return [status(s) for s in SPECS]


def provider_provenance() -> Dict[str, str]:
    """Installed providers and their versions, for recording in a trace.

    Without this a trace cannot distinguish a locally patched provider from the upstream
    release, which is what makes a measured provider fix reportable.
    """
    out: Dict[str, str] = {}
    for spec in SPECS:
        if not is_installed(spec):
            continue
        out[spec.name] = provider_version(spec) or (find_root(spec) or "present")
    return out


def _resolve_sycl_target(device: Optional[str]) -> str:
    """The architecture a source build must target, read from the device.

    The build needs a name like ``bmg``; the capability record already holds it, so this
    is derived rather than asked for. A part with no published target cannot be built for,
    and saying so before a multi-GiB build is the point.
    """
    from flashinfer_bench.device import get_accelerator, list_devices

    if device is None:
        devices = [d for d in list_devices() if d.startswith("xpu")]
        if not devices:
            raise ProviderError(
                "No Intel GPU visible, so the build architecture cannot be determined. "
                "Pass --target explicitly, or check torch.xpu.is_available()."
            )
        device = devices[0]

    caps = get_accelerator(device).capabilities(device)
    if not caps.sycl_target:
        raise ProviderError(
            f"{caps.canonical_id} has no published SYCL target name, so this provider "
            f"cannot be built for it. Supported targets are bmg (Battlemage) and cri "
            f"(Crescent Island); an integrated Xe3.0 part is neither. Use SYCL solutions "
            f"or vllm-xpu instead."
        )
    return caps.sycl_target


def _pip_install(args: List[str]) -> List[str]:
    """An install command that works in this interpreter's environment.

    A uv-managed virtualenv has no ``pip`` module at all -- ``python -m pip`` there fails
    with "No module named pip" -- so uv is preferred when present and told explicitly which
    interpreter to install into, rather than letting it guess from the working directory.
    """
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", *args]
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "pip", "install", "--python", sys.executable, *args]
    raise ProviderError(
        f"Neither pip nor uv is available to install into {sys.executable}. Install pip "
        f"into this environment, or put uv on PATH."
    )


def install_command(
    spec: ProviderSpec, target: Optional[str] = None, device: Optional[str] = None
) -> List[str]:
    """The command that acquires ``spec``, or a ``ProviderError`` explaining why not."""
    if spec.kind == "system":
        raise ProviderError(
            f"'{spec.name}' is a system library and is not installed by this command. It "
            f"ships with oneAPI; set {spec.env_var} if it lives somewhere non-standard."
        )
    if spec.kind == "wheel":
        return _pip_install([spec.distribution or spec.name])
    if spec.kind == "checkout":
        raise ProviderError(
            f"'{spec.name}' is used from a source checkout, not installed. Clone "
            f"{spec.repo} and point {spec.env_var} at it."
        )
    if spec.kind == "source":
        arch = target or _resolve_sycl_target(device)
        if not shutil.which("git"):
            raise ProviderError("git is required to build a source provider.")
        return _pip_install(
            [
                "-v",
                "--no-build-isolation",
                f"--config-settings=cmake.define.DPCPP_SYCL_TARGET={arch}",
                f"git+{spec.repo}",
            ]
        )
    raise ProviderError(f"Unknown acquisition kind '{spec.kind}' for '{spec.name}'.")


def install(
    name: str,
    target: Optional[str] = None,
    device: Optional[str] = None,
    dry_run: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], int]:
    """Acquire one provider. Returns the command run and its exit status.

    ``dry_run`` reports the command without running it, which matters here because one of
    these builds costs tens of minutes and several GiB.
    """
    spec = get_spec(name)
    command = install_command(spec, target=target, device=device)
    if dry_run:
        return command, 0

    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    if spec.kind == "source":
        # Individual SYCL translation units can need several GiB; sgl-kernel-xpu's build
        # carries an OOM guard that aborts rather than letting the host thrash, and an
        # uncapped parallel build is the usual way to trip it.
        run_env.setdefault("MAX_JOBS", "2")

    if spec.build_requires:
        prereq = _pip_install(list(spec.build_requires))
        logger.info("Installing build requirements: %s", ", ".join(spec.build_requires))
        completed = subprocess.run(prereq, env=run_env)
        if completed.returncode != 0:
            return prereq, completed.returncode

    logger.info("Running: %s", " ".join(command))
    completed = subprocess.run(command, env=run_env)
    return command, completed.returncode
