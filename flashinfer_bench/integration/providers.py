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

import ctypes
import importlib.metadata
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
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


class _DnnlVersion(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("patch", ctypes.c_int),
        ("hash", ctypes.c_char_p),
        ("cpu_runtime", ctypes.c_int),
        ("gpu_runtime", ctypes.c_int),
    ]


@lru_cache(maxsize=1)
def onednn_link_version() -> Optional[str]:
    """Version and commit of the oneDNN a SYCL solution will *link against*.

    Recording the directory is not enough. `/opt/intel/oneapi/dnnl/latest` is a symlink that
    moves, and a locally rebuilt oneDNN -- the only way to change the GEMM kernel catalog --
    sits at the same path as the stock one. Without the version and commit, a trace measured
    against a patched library is indistinguishable from one measured against a released one,
    which is exactly the claim a reader needs to check.
    """
    from flashinfer_bench.compile.builders.sycl_builder import find_onednn_root

    root = find_onednn_root()
    if root is None:
        return None
    for sub in ("lib", "lib64", "lib/intel64"):
        lib_path = Path(root) / sub / "libdnnl.so"
        if not lib_path.exists():
            continue
        try:
            lib = ctypes.CDLL(str(lib_path))
            lib.dnnl_version.restype = ctypes.POINTER(_DnnlVersion)
            v = lib.dnnl_version().contents
            commit = v.hash.decode()[:12] if v.hash else "unknown"
            real = os.path.realpath(str(lib_path))
            return f"{v.major}.{v.minor}.{v.patch}+{commit} ({real})"
        except Exception:
            continue
    return None


@lru_cache(maxsize=1)
def onednn_runtime_version() -> Optional[str]:
    """Version of the oneDNN that torch itself executes.

    Not necessarily the one solutions link against: torch-xpu bundles its own oneDNN, and it
    has been observed a minor version ahead of the oneAPI install. That matters because a
    definition's `reference` for a GEMM *is* torch's oneDNN, so a solution linking a
    different build is being compared across two libraries, not one.
    """
    import subprocess
    import sys

    code = (
        "import torch;a=torch.randn(8,64,dtype=torch.bfloat16,device='xpu:0');"
        "torch.matmul(a,a.T);torch.xpu.synchronize()"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "ONEDNN_VERBOSE": "1"},
        )
    except Exception:
        return None
    m = re.search(r"oneDNN v([0-9.]+) \(commit ([0-9a-f]+)\)", out.stderr + out.stdout)
    return f"{m.group(1)}+{m.group(2)[:12]}" if m else None


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

    # oneDNN gets version+commit rather than a path, and both sides of it. A GEMM
    # definition's reference is `torch.matmul`, which runs torch's own bundled oneDNN, while
    # a SYCL solution links whatever FIB_ONEDNN_DIR points at. When those differ the
    # comparison spans two libraries, and the trace has to say so or the number is not
    # interpretable later.
    link = onednn_link_version()
    runtime = onednn_runtime_version()
    if link:
        out["onednn"] = link
    if runtime:
        out["env:onednn_runtime"] = runtime
    if link and runtime and link.split(" ")[0].split("+")[0] != runtime.split("+")[0]:
        out["env:onednn_version_mismatch"] = (
            f"solutions link {link.split(' ')[0]}, torch runs {runtime}"
        )
        logger.warning(
            "oneDNN version mismatch: solutions link %s but torch runs %s. A SYCL solution "
            "and the reference it is measured against are using different oneDNN builds; "
            "set FIB_ONEDNN_DIR to the matching install or treat the comparison as "
            "cross-library.",
            link.split(" ")[0],
            runtime,
        )
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


INSTALLER_ENV_VAR = "FIB_PROVIDER_INSTALLER"
"""Environment variable naming the installer to use when no ``installer`` is passed."""

INSTALLERS: Tuple[str, ...] = ("pip", "uv")
"""The installers this module knows how to spell a command for."""


def _has_pip() -> bool:
    """Whether ``python -m pip`` exists in this interpreter."""
    return importlib.util.find_spec("pip") is not None


def _choose_installer(installer: Optional[str]) -> str:
    """Which installer the caller asked for: the argument, else the environment, else pip.

    There is deliberately no "auto". Falling back from the installer an environment was
    built with to a different one is the hazard this module used to have: on a box whose
    torch is an Intel XPU wheel, ``uv pip install`` resolves the provider's dependencies
    against PyPI and can replace that torch with the default CUDA build. Nothing about that
    is visible at the time, and undoing it is a long manual reinstall. So uv is only used
    when named -- by ``installer`` or by ``FIB_PROVIDER_INSTALLER`` -- and never by default.
    """
    choice = installer or os.environ.get(INSTALLER_ENV_VAR) or "pip"
    if choice not in INSTALLERS:
        raise ProviderError(
            f"Unknown installer '{choice}'. Known: {', '.join(INSTALLERS)}. "
            f"(Given by --installer or {INSTALLER_ENV_VAR}.)"
        )
    return choice


def _declined_message(packages: List[str]) -> str:
    """What a caller is told when the default installer is not there.

    A refusal is only useful if it says exactly what to run instead: the packages, the
    environment they belong in (a box can hold several venvs and they are not
    interchangeable), why this command did not do it, and how to opt in if that is the
    intended decision.
    """
    pkgs = " ".join(packages)
    return (
        f"Declined to install {pkgs} into {sys.prefix} (interpreter {sys.executable}): "
        f"that environment has no `pip` module, and this command does not substitute "
        f"another installer on its own. `uv pip install` would resolve dependencies against "
        f"PyPI and can replace a torch built for Intel XPU with the default CUDA build; "
        f"recovering from that is a manual reinstall. To install it yourself, activate "
        f"{sys.prefix} and run an installer of your choosing for: {pkgs} (with pip that is "
        f"`python -m ensurepip` once, then `python -m pip install --no-deps {pkgs}`). To let "
        f"this command use uv for that environment anyway, pass --installer uv (or set "
        f"{INSTALLER_ENV_VAR}=uv); preview the command first with --installer uv --dry-run."
    )


def _pip_install(args: List[str], installer: Optional[str] = None) -> List[str]:
    """The install command for ``args`` using the installer the caller chose.

    ``pip`` (the default) is ``python -m pip install`` in this interpreter and nothing else:
    when the interpreter has no pip module the result is a ``ProviderError`` telling the
    caller what to run, not a different package manager. ``uv`` is used only when named,
    and is told explicitly which interpreter to install into rather than left to infer it
    from the working directory.
    """
    packages = [a for a in args if not a.startswith("-")]
    choice = _choose_installer(installer)
    if choice == "pip":
        if not _has_pip():
            raise ProviderError(_declined_message(packages))
        return [sys.executable, "-m", "pip", "install", *args]
    uv = shutil.which("uv")
    if uv is None:
        raise ProviderError(
            f"--installer uv was requested but no `uv` executable is on PATH, so "
            f"{' '.join(packages)} was not installed into {sys.prefix}. Put uv on PATH, or "
            f"install the package into that environment yourself."
        )
    return [uv, "pip", "install", "--python", sys.executable, *args]


def install_command(
    spec: ProviderSpec,
    target: Optional[str] = None,
    device: Optional[str] = None,
    installer: Optional[str] = None,
) -> List[str]:
    """The command that acquires ``spec``, or a ``ProviderError`` explaining why not.

    ``installer`` is ``"pip"`` or ``"uv"``; ``None`` reads ``FIB_PROVIDER_INSTALLER`` and
    otherwise means pip. See :func:`_choose_installer` for why there is no automatic choice.
    """
    if spec.kind == "system":
        raise ProviderError(
            f"'{spec.name}' is a system library and is not installed by this command. It "
            f"ships with oneAPI; set {spec.env_var} if it lives somewhere non-standard."
        )
    if spec.kind == "wheel":
        return _pip_install([spec.distribution or spec.name], installer)
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
            ],
            installer,
        )
    raise ProviderError(f"Unknown acquisition kind '{spec.kind}' for '{spec.name}'.")


def install(
    name: str,
    target: Optional[str] = None,
    device: Optional[str] = None,
    dry_run: bool = False,
    env: Optional[Dict[str, str]] = None,
    installer: Optional[str] = None,
) -> Tuple[List[str], int]:
    """Acquire one provider. Returns the command run and its exit status.

    ``dry_run`` reports the command without running it, which matters here because one of
    these builds costs tens of minutes and several GiB. ``installer`` selects the package
    manager (``"pip"`` or ``"uv"``); left unset it is ``FIB_PROVIDER_INSTALLER`` or pip, and
    an environment with no pip gets a refusal saying what to run rather than a substitute.
    """
    spec = get_spec(name)
    command = install_command(spec, target=target, device=device, installer=installer)
    prereq = _pip_install(list(spec.build_requires), installer) if spec.build_requires else None
    if dry_run:
        if prereq is not None:
            logger.info("Would first install build requirements: %s", " ".join(prereq))
        return command, 0

    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    if spec.kind == "source":
        # Individual SYCL translation units can need several GiB; sgl-kernel-xpu's build
        # carries an OOM guard that aborts rather than letting the host thrash, and an
        # uncapped parallel build is the usual way to trip it.
        run_env.setdefault("MAX_JOBS", "2")

    if prereq is not None:
        logger.info("Installing build requirements: %s", ", ".join(spec.build_requires))
        completed = subprocess.run(prereq, env=run_env)
        if completed.returncode != 0:
            return prereq, completed.returncode

    logger.info("Running: %s", " ".join(command))
    completed = subprocess.run(command, env=run_env)
    return command, completed.returncode
