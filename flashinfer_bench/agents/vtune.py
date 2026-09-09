"""Intel VTune profiling for Intel GPUs, alongside :mod:`flashinfer_bench.agents.unitrace`.

unitrace answers *how long each kernel took*. VTune answers *where a kernel's time went*:
XVE active / stalled / idle, thread occupancy, stall reasons sampled per instruction,
L3 and memory bytes per kernel, instruction mix. Those come from the GPU's hardware
counters, which a wall-clock timer cannot see, and they are what decides between two
explanations of the same slowdown.

VTune reaches a kernel through two independent channels, and each has its own
prerequisite. Both are checked before anything is launched, because VTune itself reports
these failures late and indirectly (``Cannot stop collection of GPU events`` is what a
refused counter stream looks like from the console):

* **API tracing** (kernel names and per-launch device time) injects a tool into the
  process with Pin. Pin needs its private runtime under ``<vtune>/lib64/pinruntime`` to be
  loadable and ``kernel.yama.ptrace_scope=0``. Attach mode goes through the same path.
* **Hardware counters** (everything in the list above) open an observation stream on the
  GPU driver. On the ``xe`` driver that is gated by ``dev.xe.observation_paranoid``, on
  ``i915`` by ``dev.i915.perf_stream_paranoid``; a non-zero value refuses a process
  without ``CAP_PERFMON``. Memory bandwidth additionally needs VTune's sampling driver.

The binary is found through :data:`VTUNE_ENV`, then ``vtune`` on ``PATH``, then the
vendor's default install prefix (see :func:`find_vtune`).

Command line::

    python -m flashinfer_bench.agents.vtune --check --device xpu:0
    python -m flashinfer_bench.agents.vtune --mode timing --harness path/to/harness.py --seconds 5
    python -m flashinfer_bench.agents.vtune --mode characterization -- python script.py
"""

from __future__ import annotations

import argparse
import ctypes.util
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from flashinfer_bench.data import Solution, Workload

logger = logging.getLogger(__name__)

VTUNE_ENV = "FIB_VTUNE"
"""Environment variable naming the ``vtune`` binary. Consulted before ``PATH``."""

_VENDOR_DEFAULTS = (Path("/opt/intel/oneapi/vtune/latest/bin64/vtune"),)
"""The oneAPI installer's documented default location. Tried after ``PATH``; anything
else is named with :data:`VTUNE_ENV` rather than guessed at."""

_ONEAPI_ROOT_ENV = "ONEAPI_ROOT"

BDF = Tuple[int, int, int, int]
"""PCI address as ``(domain, bus, device, function)``."""


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------


def find_vtune(vtune_path: Optional[str] = None) -> Optional[str]:
    """Locate the ``vtune`` binary, or return None when nothing names one.

    Order: ``vtune_path`` if given, then :data:`VTUNE_ENV`, then ``vtune`` on ``PATH``, then
    ``$ONEAPI_ROOT/vtune/latest/bin64/vtune``, then :data:`_VENDOR_DEFAULTS`.

    An explicit path or a set environment variable that does not name an executable raises
    :class:`FileNotFoundError` rather than falling through, as with unitrace: quietly using
    some other binary is the outcome the variable exists to prevent.
    """
    for source, value in ((" vtune_path", vtune_path), (VTUNE_ENV, os.environ.get(VTUNE_ENV))):
        if not value:
            continue
        found = shutil.which(value)
        if found is None:
            raise FileNotFoundError(
                f"{source.strip()}={value!r} does not name an executable vtune binary. Point it "
                "at <VTune install>/bin64/vtune."
            )
        return found
    found = shutil.which("vtune")
    if found:
        return found
    candidates = list(_VENDOR_DEFAULTS)
    oneapi = os.environ.get(_ONEAPI_ROOT_ENV)
    if oneapi:
        candidates.insert(0, Path(oneapi) / "vtune" / "latest" / "bin64" / "vtune")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def vtune_missing_message() -> str:
    """What to do when :func:`find_vtune` returns None, naming every place it looked."""
    searched = ", ".join(str(p) for p in _VENDOR_DEFAULTS)
    return (
        f"vtune not found: {VTUNE_ENV} is unset, `vtune` is not on PATH, {_ONEAPI_ROOT_ENV} is "
        f"unset, and there is no binary at {searched}. Install Intel VTune Profiler (part of "
        f"the oneAPI Base Toolkit, or a standalone package) and set {VTUNE_ENV}=/path/to/vtune, "
        "or put its bin64 directory on PATH."
    )


def is_vtune_available(vtune_path: Optional[str] = None) -> bool:
    """Whether a vtune binary can be found (a misconfigured name counts as not found)."""
    try:
        return find_vtune(vtune_path) is not None
    except FileNotFoundError:
        return False


def vtune_root(binary: str) -> Path:
    """The install prefix for a ``<prefix>/bin64/vtune`` binary."""
    return Path(binary).resolve().parent.parent


# --------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------

NEED_PIN = "pin"
NEED_COUNTERS = "counters"
NEED_SAMPLING_DRIVER = "sampling-driver"


@dataclass(frozen=True)
class Mode:
    """One VTune analysis shape this wrapper knows how to launch and read."""

    name: str
    collect: str
    knobs: Tuple[Tuple[str, str], ...]
    needs: frozenset
    gives: str


MODES: Dict[str, Mode] = {
    "timing": Mode(
        "timing",
        "gpu-offload",
        (("enable-characterization-insights", "false"),),
        frozenset({NEED_PIN}),
        "Per-kernel device time and launch count from API tracing, with the host-side "
        "picture (what the CPU did between launches). No hardware counters; this is the "
        "mode that works without the counter sysctl.",
    ),
    "characterization": Mode(
        "characterization",
        "gpu-hotspots",
        (("gpu-profiling-mode", "characterization"),),
        frozenset({NEED_PIN, NEED_COUNTERS}),
        "Per-kernel hardware counters: XVE active/stalled/idle, thread occupancy, L3 and "
        "GPU memory bytes, instruction mix per pipe; the metric group is selectable.",
    ),
    "stall": Mode(
        "stall",
        "gpu-hotspots",
        (("gpu-profiling-mode", "source-analysis"), ("source-analysis", "stall-sampling")),
        frozenset({NEED_PIN, NEED_COUNTERS}),
        "Stall reasons sampled per instruction (EU stall sampling), attributed to kernel "
        "source and assembly.",
    ),
    "bb-latency": Mode(
        "bb-latency",
        "gpu-hotspots",
        (("gpu-profiling-mode", "source-analysis"), ("source-analysis", "bb-latency")),
        frozenset({NEED_PIN}),
        "Basic-block latency by binary instrumentation (GTPin); no counters needed.",
    ),
}

METRIC_GROUPS = (
    "overview",
    "global-memory-accesses",
    "compute-extended",
    "lsc-slm",
    "hdc",
    "full-compute",
    "instruction-count",
)
"""Values of the ``characterization-mode`` knob this wrapper passes through."""


def flashinfer_bench_list_vtune_modes() -> str:
    """List the VTune modes this wrapper supports and what each reports.

    Returns
    -------
    str
        One mode per line with what it gives and which prerequisites it needs.
    """
    lines = []
    for mode in MODES.values():
        needs = ", ".join(sorted(mode.needs))
        lines.append(f"{mode.name:<18} needs: {needs}\n{'':<18} {mode.gives}")
    lines.append(f"{'metric groups':<18} {', '.join(METRIC_GROUPS)} (characterization only)")
    lines.append(
        f"{'bandwidth=True':<18} adds memory bandwidth to characterization; needs the "
        "sampling driver"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Prerequisites
# --------------------------------------------------------------------------------------


@dataclass
class Prerequisite:
    """One condition a mode depends on, with the fix when it does not hold."""

    name: str
    ok: bool
    detail: str
    fix: str = ""
    needs_root: bool = False
    satisfies: frozenset = field(default_factory=frozenset)

    def describe(self) -> str:
        state = "ok" if self.ok else ("NEEDS ROOT" if self.needs_root else "MISSING")
        text = f"[{state}] {self.name}: {self.detail}"
        if not self.ok and self.fix:
            text += f"\n    fix: {self.fix}"
        return text


def _read_first_line(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def check_pin_runtime(binary: str) -> Prerequisite:
    """Pin's private runtime must be loadable, or no kernel name is ever recorded.

    Pin loads ``lib64/pinruntime`` with its own linker, which honours neither
    ``LD_LIBRARY_PATH`` nor ``LD_PRELOAD``, so the files must carry their exact names in
    that directory. A package that percent-encoded the ``+`` in ``libc++.so`` leaves Pin
    unable to start, and VTune reports it only as the application exiting at once. The
    fix is a symlink per mangled file, which needs write access to the install.
    """
    root = vtune_root(binary)
    fixes: List[str] = []
    seen = False
    for libdir in ("lib64", "lib32"):
        runtime = root / libdir / "pinruntime"
        if not runtime.is_dir():
            continue
        seen = True
        for entry in sorted(runtime.iterdir()):
            decoded = urllib.parse.unquote(entry.name)
            if decoded != entry.name and not (runtime / decoded).exists():
                fixes.append(f"sudo ln -s '{entry.name}' '{runtime / decoded}'")
    if not seen:
        return Prerequisite(
            "pin-runtime",
            True,
            f"no lib64/pinruntime under {root}; cannot verify Pin's runtime",
            satisfies=frozenset({NEED_PIN}),
        )
    if fixes:
        return Prerequisite(
            "pin-runtime",
            False,
            f"{len(fixes)} file(s) under {root}/lib*/pinruntime carry percent-encoded names, so "
            "Pin cannot link them and every traced application exits immediately "
            "(console: `pinbin: error while loading shared libraries` or "
            "`CANNOT LINK EXECUTABLE DEPENDENCIES`)",
            fix="; ".join(fixes),
            needs_root=not os.access(root / "lib64" / "pinruntime", os.W_OK),
            satisfies=frozenset({NEED_PIN}),
        )
    return Prerequisite(
        "pin-runtime", True, f"{root}/lib64/pinruntime resolves", satisfies=frozenset({NEED_PIN})
    )


def check_ptrace_scope(proc_root: Union[str, Path] = "/proc") -> Prerequisite:
    """Pin injects with ptrace; Yama scope 1 or above stops it (VTune says so up front)."""
    path = Path(proc_root) / "sys" / "kernel" / "yama" / "ptrace_scope"
    value = _read_first_line(path)
    ok = value is None or value == "0"
    return Prerequisite(
        "ptrace-scope",
        ok,
        f"kernel.yama.ptrace_scope={value if value is not None else 'absent'}",
        fix="sudo sysctl kernel.yama.ptrace_scope=0" if not ok else "",
        needs_root=not ok,
        satisfies=frozenset({NEED_PIN}),
    )


def gpu_driver_for_bdf(bdf: BDF, sys_root: Union[str, Path] = "/sys") -> Optional[str]:
    """Kernel driver bound to a PCI address (``xe``, ``i915``), from sysfs."""
    domain, bus, dev, fn = bdf
    link = Path(sys_root) / "bus" / "pci" / "devices" / f"{domain:04x}:{bus:02x}:{dev:02x}.{fn}"
    try:
        return os.path.basename(os.readlink(link / "driver"))
    except OSError:
        return None


_OBSERVATION_SYSCTL = {
    "xe": "dev.xe.observation_paranoid",
    "i915": "dev.i915.perf_stream_paranoid",
}


def check_gpu_observation(
    bdf: Optional[BDF],
    proc_root: Union[str, Path] = "/proc",
    sys_root: Union[str, Path] = "/sys",
    driver: Optional[str] = None,
) -> Prerequisite:
    """The driver must let an unprivileged process open a counter stream on this GPU.

    Both OA metrics (characterization) and EU stall sampling go through this gate; VTune
    does not check it before starting, and the refusal surfaces as ``OpenIoStream returned
    error`` in the collection log and ``Cannot stop collection of GPU events`` on the
    console, after which the whole GPU plugin is disabled for that run.
    """
    if driver is None and bdf is not None:
        driver = gpu_driver_for_bdf(bdf, sys_root)
    sysctl = _OBSERVATION_SYSCTL.get(driver or "")
    if sysctl is None:
        return Prerequisite(
            "gpu-observation",
            True,
            f"driver {driver!r} for {format_bdf(bdf) if bdf else '?'}: no known counter gate",
            satisfies=frozenset({NEED_COUNTERS}),
        )
    value = _read_first_line(Path(proc_root) / "sys" / Path(*sysctl.split(".")))
    ok = value == "0"
    conf = f"/etc/sysctl.d/60-{driver}-observation.conf"
    return Prerequisite(
        "gpu-observation",
        ok,
        f"{sysctl}={value if value is not None else 'absent'} on the {driver} device "
        f"{format_bdf(bdf) if bdf else ''}".rstrip()
        + ("" if ok else "; the driver refuses counter streams to a process without CAP_PERFMON"),
        fix=(
            f"sudo sysctl {sysctl}=0   # until reboot; to persist: "
            f"echo '{sysctl} = 0' | sudo tee {conf}"
        )
        if not ok
        else "",
        needs_root=not ok,
        satisfies=frozenset({NEED_COUNTERS}),
    )


def check_metrics_library() -> Prerequisite:
    """Intel's Metrics Discovery library decodes the counter stream; without it VTune
    refuses hardware metrics before starting."""
    found = ctypes.util.find_library("igdmd") or ctypes.util.find_library("md")
    return Prerequisite(
        "metrics-discovery",
        found is not None,
        f"found {found}" if found else "neither libigdmd.so nor libmd.so is on the loader path",
        fix="install intel-metrics-discovery (libigdmd.so) from your distribution or "
        "https://github.com/intel/metrics-discovery"
        if not found
        else "",
        needs_root=found is None,
        satisfies=frozenset({NEED_COUNTERS}),
    )


def check_sampling_driver(binary: str, proc_root: Union[str, Path] = "/proc") -> Prerequisite:
    """Memory bandwidth comes from uncore counters through VTune's own kernel driver."""
    modules = _read_first_line(Path(proc_root) / "modules") or ""
    loaded = {line.split(" ", 1)[0] for line in modules.splitlines()}
    ok = bool(loaded & {"sep5", "sep4_1", "pax"})
    sepdk = vtune_root(binary) / "sepdk" / "src"
    return Prerequisite(
        "sampling-driver",
        ok,
        "sep/pax sampling driver loaded"
        if ok
        else "sep/pax sampling driver not loaded "
        "(console: `Failed to connect to PMU reservation service (PAX)`)",
        fix=f"cd {sepdk} && sudo ./build-driver -ni && sudo ./insmod-sep" if not ok else "",
        needs_root=not ok,
        satisfies=frozenset({NEED_SAMPLING_DRIVER}),
    )


def check_prerequisites(
    binary: str, mode: str, bdf: Optional[BDF], bandwidth: bool = False
) -> List[Prerequisite]:
    """Every prerequisite the requested mode depends on, checked, in report order."""
    needs = set(MODES[mode].needs)
    if bandwidth:
        needs.add(NEED_SAMPLING_DRIVER)
    checks: List[Prerequisite] = []
    if NEED_PIN in needs:
        checks.append(check_pin_runtime(binary))
        checks.append(check_ptrace_scope())
    if NEED_COUNTERS in needs:
        checks.append(check_metrics_library())
        checks.append(check_gpu_observation(bdf))
    if NEED_SAMPLING_DRIVER in needs:
        checks.append(check_sampling_driver(binary))
    return checks


def prerequisites_report(checks: Sequence[Prerequisite]) -> str:
    return "\n".join(check.describe() for check in checks)


# --------------------------------------------------------------------------------------
# Which GPU: from a torch device to the BDF VTune's target-gpu knob wants
# --------------------------------------------------------------------------------------


def bdf_from_uuid(uuid: str) -> Optional[Tuple[int, int, BDF]]:
    """Decode the Level Zero device UUID Intel's compute runtime hands out.

    Layout (bytes): vendor id (2, LE), device id (2, LE), revision (2), PCI domain (2,
    LE), bus (1), device (1), function (1), sub-device (1), reserved. Returns
    ``(vendor_id, device_id, (domain, bus, device, function))`` or None when the string is
    not sixteen bytes of hex.
    """
    raw = re.sub(r"[^0-9a-fA-F]", "", uuid)
    if len(raw) != 32:
        return None
    b = bytes.fromhex(raw)
    vendor = b[0] | (b[1] << 8)
    device = b[2] | (b[3] << 8)
    domain = b[6] | (b[7] << 8)
    return vendor, device, (domain, b[8], b[9], b[10])


def intel_gpus_from_sysfs(sys_root: Union[str, Path] = "/sys") -> List[Tuple[BDF, int, str]]:
    """Intel display-class PCI functions as ``(bdf, device_id, driver)``."""
    found: List[Tuple[BDF, int, str]] = []
    devices = Path(sys_root) / "bus" / "pci" / "devices"
    if not devices.is_dir():
        return found
    for entry in sorted(devices.iterdir()):
        try:
            cls = int((entry / "class").read_text().strip(), 16)
            vendor = int((entry / "vendor").read_text().strip(), 16)
            device = int((entry / "device").read_text().strip(), 16)
        except (OSError, ValueError):
            continue
        if vendor != 0x8086 or (cls >> 16) != 0x03:
            continue
        m = re.fullmatch(r"([0-9a-f]{4}):([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])", entry.name)
        if not m:
            continue
        bdf = tuple(int(x, 16) for x in m.groups())
        try:
            driver = os.path.basename(os.readlink(entry / "driver"))
        except OSError:
            driver = ""
        found.append((bdf, device, driver))  # type: ignore[arg-type]
    return found


def device_bdf(device: str = "xpu:0", sys_root: Union[str, Path] = "/sys") -> Optional[BDF]:
    """PCI address of a torch XPU device, from its UUID, cross-checked against sysfs.

    Returns None when torch has no XPU, the UUID does not decode, or the decoded address
    does not carry the device id torch reports (which would mean the layout assumption
    above is wrong for this runtime, and guessing is worse than asking).
    """
    try:
        import torch

        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return None
        index = int(device.split(":")[1]) if ":" in device else 0
        props = torch.xpu.get_device_properties(index)
    except Exception:
        return None
    decoded = bdf_from_uuid(str(getattr(props, "uuid", "")))
    device_id = getattr(props, "device_id", None)
    if decoded is not None:
        _, uuid_device_id, bdf = decoded
        if device_id is None or uuid_device_id == device_id:
            sysfs = {b: d for b, d, _ in intel_gpus_from_sysfs(sys_root)}
            if not sysfs or sysfs.get(bdf) in (None, device_id):
                return bdf
    if device_id is not None:
        matches = [b for b, d, _ in intel_gpus_from_sysfs(sys_root) if d == device_id]
        if len(matches) == 1:
            return matches[0]
    return None


def format_bdf(bdf: Optional[BDF]) -> str:
    """``domain:bus:device.function`` in decimal, the form VTune prints."""
    if bdf is None:
        return "?"
    domain, bus, dev, fn = bdf
    return f"{domain}:{bus}:{dev}.{fn}"


def parse_vtune_adapters(text: str) -> List[Tuple[str, str]]:
    """``(bdf, name)`` pairs from ``gpuAdapterNameList: 0:0:2.0|Name;0:4:0.0|Name;``."""
    m = re.search(r"gpuAdapterNameList:\s*(.*)", text)
    if not m:
        return []
    pairs = []
    for item in m.group(1).split(";"):
        item = item.strip()
        if not item:
            continue
        bdf, _, name = item.partition("|")
        pairs.append((bdf.strip(), name.strip()))
    return pairs


def vtune_gpu_adapters(binary: str, timeout: int = 60) -> List[Tuple[str, str]]:
    """The GPUs VTune itself can target, as ``(bdf, name)``; empty if it cannot say."""
    runss = Path(binary).resolve().parent / "amplxe-runss"
    if not runss.is_file():
        return []
    try:
        out = subprocess.run(
            [str(runss), "--context-value-list"], capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_vtune_adapters(out.stdout + out.stderr)


def match_vtune_bdf(bdf: BDF, candidates: Sequence[str]) -> Optional[str]:
    """Pick the VTune adapter string naming ``bdf``.

    VTune prints the address with unpadded fields; whether they are decimal or hex only
    matters from bus 10 upward. A hex digit anywhere in the list settles it as hex;
    otherwise decimal is tried first and hex second, and a result is returned only when
    exactly one candidate matches under a base.
    """
    bases = (16,) if any(re.search(r"[a-fA-F]", c) for c in candidates) else (10, 16)
    for base in bases:
        hits = []
        for cand in candidates:
            m = re.fullmatch(
                r"\s*([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-9a-fA-F]+)\s*", cand
            )
            if not m:
                continue
            try:
                parsed = tuple(int(x, base) for x in m.groups())
            except ValueError:
                continue
            if parsed == tuple(bdf):
                hits.append(cand.strip())
        if len(hits) == 1:
            return hits[0]
    return None


def select_target_gpu(
    device: str, binary: str, target_gpu: Optional[str] = None
) -> Tuple[Optional[str], Optional[BDF], str]:
    """Resolve the ``target-gpu`` knob value for a torch device.

    Returns ``(knob value or None, decoded bdf or None, explanation)``. An explicit
    ``target_gpu`` is used as given. Otherwise the torch device's PCI address is matched
    against the adapters VTune lists; with several Intel GPUs in the box, profiling
    without pinning one would sample counters on both and attribute the idle one.
    """
    adapters = vtune_gpu_adapters(binary)
    names = dict(adapters)
    if target_gpu:
        m = re.fullmatch(r"\s*(\d+):(\d+):(\d+)\.(\d+)\s*", target_gpu)
        bdf = tuple(int(x) for x in m.groups()) if m else None  # type: ignore[assignment]
        return (
            target_gpu.strip(),
            bdf,
            f"target-gpu given: {target_gpu} {names.get(target_gpu.strip(), '')}",
        )  # type: ignore[return-value]
    bdf = device_bdf(device)
    if bdf is None:
        if len(adapters) == 1:
            return (
                adapters[0][0],
                None,
                (
                    f"could not read {device}'s PCI address from torch; VTune lists one adapter, "
                    f"{adapters[0][0]} {adapters[0][1]}"
                ),
            )
        listed = ", ".join(f"{b} ({n})" for b, n in adapters) or "none"
        return (
            None,
            None,
            (
                f"could not read {device}'s PCI address from torch and VTune lists {len(adapters)} "
                f"adapters ({listed}); pass target_gpu explicitly"
            ),
        )
    if not adapters:
        return (
            format_bdf(bdf),
            bdf,
            f"{device} is {format_bdf(bdf)} (VTune adapter list unavailable)",
        )
    chosen = match_vtune_bdf(bdf, [b for b, _ in adapters])
    if chosen is None:
        listed = ", ".join(f"{b} ({n})" for b, n in adapters)
        return (
            None,
            bdf,
            (
                f"{device} is PCI {format_bdf(bdf)} but VTune lists {listed}; pass target_gpu "
                "explicitly"
            ),
        )
    return chosen, bdf, f"{device} is {chosen} {names.get(chosen, '')}".rstrip()


# --------------------------------------------------------------------------------------
# Command and report
# --------------------------------------------------------------------------------------


def build_vtune_command(
    binary: str,
    mode: str,
    target_gpu: Optional[str],
    result_dir: Union[str, Path],
    command: Sequence[str],
    *,
    metric_group: Optional[str] = None,
    bandwidth: bool = False,
    sampling_interval_ms: Optional[float] = None,
    kernels_of_interest: Optional[str] = None,
) -> List[str]:
    """The ``vtune -collect`` line for a mode. ``target_gpu`` is a knob, not an option:
    ``vtune -target-gpu`` is rejected by the configuration manager."""
    m = MODES[mode]
    cmd: List[str] = [binary, "-collect", m.collect]
    knobs = list(m.knobs)
    if mode == "characterization":
        group = metric_group or "overview"
        if group not in METRIC_GROUPS:
            raise ValueError(f"metric_group must be one of {METRIC_GROUPS}, got {group!r}")
        knobs.append(("characterization-mode", group))
    if bandwidth:
        if m.collect != "gpu-hotspots":
            raise ValueError("bandwidth is a gpu-hotspots knob; use mode='characterization'")
        knobs.append(("collect-memory-bandwidth", "true"))
    if target_gpu:
        knobs.append(("target-gpu", target_gpu))
    if sampling_interval_ms is not None and m.collect == "gpu-hotspots":
        knobs.append(("gpu-sampling-interval", str(sampling_interval_ms)))
    if kernels_of_interest and m.collect == "gpu-hotspots":
        knobs.append(("computing-tasks-of-interest", kernels_of_interest))
    for name, value in knobs:
        cmd.extend(["-knob", f"{name}={value}"])
    cmd.extend(["-r", str(result_dir), "--", *command])
    return cmd


_FAILURE_SIGNATURES: Tuple[Tuple[str, str], ...] = (
    (
        r"pinbin: error while loading shared libraries|CANNOT LINK EXECUTABLE DEPENDENCIES",
        "Pin could not start, so the application was killed before it ran a kernel: "
        "VTune's lib64/pinruntime is not loadable (see the pin-runtime prerequisite; on a "
        "package that percent-encoded `+` in file names the fix is a symlink per file, "
        "which needs write access to the install).",
    ),
    (
        r"OpenIoStream returned error|Cannot open IO stream|Cannot stop collection of GPU events",
        "The GPU driver refused the hardware-counter stream, and VTune disabled its GPU "
        "plugin for the run: set the driver's observation sysctl to 0 "
        "(dev.xe.observation_paranoid on xe, dev.i915.perf_stream_paranoid on i915; root), "
        "or use mode='timing', which does not open a counter stream.",
    ),
    (
        r"PMU reservation service \(PAX\)|Failed to connect to PMU",
        "Memory bandwidth needs VTune's sampling driver (sep/pax), which is not loaded: "
        "build and insmod it from <vtune>/sepdk/src (root), or drop bandwidth=True.",
    ),
    (
        r"ptrace_scope",
        "Pin needs kernel.yama.ptrace_scope=0 (root).",
    ),
    (
        r"neither libigdmd\.so nor libmd\.so",
        "Hardware metrics need Intel's Metrics Discovery library (libigdmd.so); install it.",
    ),
    (
        r"ThisTargetTypeNotWorking",
        "VTune rejected the target configuration; select the GPU with `-knob target-gpu=`, "
        "not `-target-gpu`.",
    ),
    (
        r"0x40000024 \(No data\)|Empty request output",
        "The result holds no GPU data. Either the application ran no kernel on the "
        "targeted GPU, or one of the failures above removed the GPU plugin.",
    ),
)


def explain_vtune_failure(console: str, log_text: str = "") -> List[str]:
    """Translate VTune's console and collection-log text into the causes seen so far."""
    haystack = console + "\n" + log_text
    return [text for pattern, text in _FAILURE_SIGNATURES if re.search(pattern, haystack)]


def read_collection_logs(result_dir: Union[str, Path]) -> str:
    """Error-level lines from ``<result>/log/perfrun-*.log`` and the application exit code.

    VTune's console shows one line per failure; the reason is in these logs.
    """
    log_dir = Path(result_dir) / "log"
    if not log_dir.is_dir():
        return ""
    kept: List[str] = []
    for path in sorted(log_dir.glob("perfrun-*.log")):
        try:
            for line in path.read_text(errors="replace").splitlines():
                if " ERROR " in line or "exit code" in line:
                    kept.append(re.sub(r"^\d+ \[\d+\] ", "", line))
        except OSError:
            continue
    return "\n".join(dict.fromkeys(kept))


def report_vtune_result(binary: str, result_dir: Union[str, Path], mode: str, timeout: int) -> str:
    """Per-kernel table (``-group-by=computing-task``), plus the summary for counter modes."""
    reports = [["-report", "hotspots", "-group-by=computing-task"]]
    if NEED_COUNTERS in MODES[mode].needs:
        reports.append(["-report", "summary", "-report-knob", "show-issues=false"])
    chunks = []
    for args in reports:
        try:
            out = subprocess.run(
                [binary, *args, "-r", str(result_dir)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            chunks.append(f"vtune {' '.join(args)} timed out after {timeout}s")
            continue
        text = "\n".join(
            line
            for line in (out.stdout + out.stderr).splitlines()
            if "Executing actions" not in line and line.strip()
        )
        chunks.append(f"=== vtune {' '.join(args)} ===\n{text}")
    return "\n\n".join(chunks)


def _truncate_output(output: str, max_lines: int) -> str:
    lines = output.split("\n")
    if len(lines) <= max_lines:
        return output
    kept = lines[:max_lines]
    kept.append(f"... [{len(lines) - max_lines} more lines truncated]")
    return "\n".join(kept)


# --------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------


def profile_command_with_vtune(
    command: Sequence[str],
    *,
    device: str = "xpu:0",
    mode: str = "timing",
    target_gpu: Optional[str] = None,
    metric_group: Optional[str] = None,
    bandwidth: bool = False,
    sampling_interval_ms: Optional[float] = None,
    kernels_of_interest: Optional[str] = None,
    vtune_path: Optional[str] = None,
    result_dir: Optional[Union[str, Path]] = None,
    check: bool = True,
    timeout: int = 600,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """Run ``command`` under VTune and return the per-kernel report or a diagnosis.

    Prerequisites for ``mode`` are checked first (``check=False`` skips this); a missing
    one is reported with its fix and nothing is launched, because VTune would run the
    whole workload and then report a late, indirect failure.

    ``result_dir`` keeps the VTune result for the GUI (``vtune-gui <dir>``); otherwise a
    temporary directory is used and removed.
    """
    try:
        binary = find_vtune(vtune_path)
    except FileNotFoundError as e:
        return str(e)
    if binary is None:
        return vtune_missing_message()
    if mode not in MODES:
        return f"Unsupported VTune mode {mode!r}. Supported: {', '.join(MODES)}"

    knob, bdf, how = select_target_gpu(device, binary, target_gpu)
    if knob is None:
        return f"Cannot choose the GPU for VTune: {how}"

    if check:
        checks = check_prerequisites(binary, mode, bdf, bandwidth)
        blocking = [c for c in checks if not c.ok]
        if blocking:
            return (
                f"VTune mode {mode!r} cannot run on this machine yet ({how}):\n"
                + prerequisites_report(checks)
                + "\n"
                + (
                    "Modes that need none of the missing items: "
                    + (", ".join(_runnable_modes(checks)) or "none")
                )
            )

    def _run(out_dir: Path) -> str:
        try:
            cmd = build_vtune_command(
                binary,
                mode,
                knob,
                out_dir,
                command,
                metric_group=metric_group,
                bandwidth=bandwidth,
                sampling_interval_ms=sampling_interval_ms,
                kernels_of_interest=kernels_of_interest,
            )
        except ValueError as e:
            return str(e)
        logger.debug("Running: %s", " ".join(cmd))
        run_env = dict(os.environ)
        if env:
            run_env.update(env)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
                env=run_env,
            )
        except subprocess.TimeoutExpired:
            return f"vtune timed out after {timeout}s"
        except OSError as e:
            return f"Failed to run vtune: {type(e).__name__}: {e}"
        console = result.stdout + result.stderr
        logs = read_collection_logs(out_dir)
        report = report_vtune_result(binary, out_dir, mode, timeout)
        causes = explain_vtune_failure(console, logs + "\n" + report)
        header = f"GPU: {how}\nvtune exit code {result.returncode}; result: {out_dir}\n"
        if causes:
            header += "Diagnosis:\n" + "\n".join(f"  - {c}" for c in causes) + "\n"
            header += "Collection log (errors):\n" + (logs or "  (none)") + "\n"
        return header + "\n" + report

    if result_dir is not None:
        out = Path(result_dir)
        if out.exists():
            shutil.rmtree(out)
        return _run(out)
    with tempfile.TemporaryDirectory(prefix="fib-vtune-") as work_dir:
        return _run(Path(work_dir) / "result")


def _runnable_modes(checks: Sequence[Prerequisite]) -> List[str]:
    missing = set()
    for c in checks:
        if not c.ok:
            missing |= c.satisfies
    return [name for name, m in MODES.items() if not (m.needs & missing)]


def flashinfer_bench_run_vtune(
    solution: Union[Solution, str],
    workload: Union[Workload, str],
    *,
    device: str = "xpu:0",
    trace_set_path: Optional[str] = None,
    mode: str = "timing",
    metric_group: Optional[str] = None,
    bandwidth: bool = False,
    iterations: int = 500,
    target_gpu: Optional[str] = None,
    vtune_path: Optional[str] = None,
    result_dir: Optional[str] = None,
    timeout: int = 600,
    tmpdir: Optional[str] = None,
    max_lines: Optional[int] = 200,
) -> str:
    """Profile a solution on an Intel GPU with VTune.

    All inputs and outputs are JSON-serializable, matching
    :func:`flashinfer_bench.agents.flashinfer_bench_run_unitrace`.

    Parameters
    ----------
    solution : Solution or str
        The solution to profile, or a path to its JSON file.
    workload : Workload or str
        The workload giving input dimensions, or a path to its JSON file.
    device : str
        Torch device to run on; VTune's target GPU is derived from it.
    trace_set_path : Optional[str]
        Dataset root. Falls back to ``FIB_DATASET_PATH``.
    mode : str
        ``timing`` (per-kernel device time, no counters), ``characterization`` (hardware
        counters per kernel), ``stall`` (stall reasons per instruction) or ``bb-latency``.
        See :func:`flashinfer_bench_list_vtune_modes`.
    metric_group : Optional[str]
        Counter preset for ``characterization``; one of :data:`METRIC_GROUPS`.
    bandwidth : bool
        Also collect memory bandwidth (needs VTune's sampling driver).
    iterations : int
        How many times the kernel is launched. Sampled counters need many launches of a
        short kernel before any sample lands in it.
    target_gpu : Optional[str]
        VTune adapter address (``domain:bus:device.function``) to pin, when the automatic
        choice from ``device`` is not wanted.
    vtune_path : Optional[str]
        Path to the vtune binary. When omitted it is discovered; see :func:`find_vtune`.
    result_dir : Optional[str]
        Keep VTune's result here for the GUI. A temporary directory is used otherwise.
    timeout : int
        Seconds before the profiling run is abandoned.
    tmpdir : Optional[str]
        Working directory for staging the solution.
    max_lines : Optional[int]
        Truncate the report to this many lines. ``None`` keeps everything.

    Returns
    -------
    str
        The per-kernel report, or a message saying which prerequisite is missing and how
        to satisfy it (root-only steps are marked).
    """
    from flashinfer_bench.agents.solution_handler import extract_solution_to_files

    try:
        binary = find_vtune(vtune_path)
    except FileNotFoundError as e:
        return str(e)
    if binary is None:
        return vtune_missing_message()
    if mode not in MODES:
        return f"Unsupported VTune mode {mode!r}. Supported: {', '.join(MODES)}"

    with tempfile.TemporaryDirectory(dir=tmpdir) as work_dir:
        data_dir = Path(work_dir)
        try:
            extract_solution_to_files(solution, workload, data_dir)
        except Exception as e:
            return f"Failed to stage solution for profiling: {type(e).__name__}: {e}"
        runner = [
            sys.executable,
            "-u",
            "-m",
            "flashinfer_bench.agents._solution_runner",
            "--data-dir",
            str(data_dir),
            "--device",
            device,
            "--iterations",
            str(iterations),
        ]
        if trace_set_path:
            runner.extend(["--trace-set-path", trace_set_path])
        report = profile_command_with_vtune(
            runner,
            device=device,
            mode=mode,
            target_gpu=target_gpu,
            metric_group=metric_group,
            bandwidth=bandwidth,
            vtune_path=binary,
            result_dir=result_dir,
            timeout=timeout,
            cwd=work_dir,
        )
    if max_lines is not None:
        report = _truncate_output(report, max_lines)
    return report


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m flashinfer_bench.agents.vtune",
        description="Profile a command or a kernel harness on an Intel GPU with VTune.",
    )
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--mode", default="timing", choices=sorted(MODES))
    parser.add_argument("--metric-group", choices=METRIC_GROUPS)
    parser.add_argument("--bandwidth", action="store_true")
    parser.add_argument("--target-gpu", help="VTune adapter address to pin (domain:bus:dev.fn)")
    parser.add_argument("--result-dir", help="keep the VTune result here")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-check", action="store_true", help="launch even if a check fails")
    parser.add_argument("--check", action="store_true", help="only report prerequisites")
    parser.add_argument("--list-modes", action="store_true")
    parser.add_argument("--harness", help="a Model/get_inputs file to loop under the profiler")
    parser.add_argument("--seconds", type=float, default=5.0, help="how long to loop --harness")
    parser.add_argument("command", nargs="*", help="command to profile (after --)")
    args = parser.parse_args(argv)

    if args.list_modes:
        print(flashinfer_bench_list_vtune_modes())
        return 0
    try:
        binary = find_vtune()
    except FileNotFoundError as e:
        print(e)
        return 2
    if binary is None:
        print(vtune_missing_message())
        return 2
    knob, bdf, how = select_target_gpu(args.device, binary, args.target_gpu)
    print(f"vtune: {binary}")
    if args.check:
        print(f"GPU: {how}")
        print(prerequisites_report(check_prerequisites(binary, args.mode, bdf, args.bandwidth)))
        return 0
    if knob is None:
        print(f"Cannot choose the GPU for VTune: {how}")
        return 2
    if args.harness:
        command = [
            sys.executable,
            "-u",
            "-m",
            "flashinfer_bench.agents._harness_runner",
            args.harness,
            "--seconds",
            str(args.seconds),
            "--device",
            args.device,
        ]
    elif args.command:
        command = list(args.command)
    else:
        parser.error("give --harness FILE, or a command after --")
    print(
        profile_command_with_vtune(
            command,
            device=args.device,
            mode=args.mode,
            target_gpu=args.target_gpu,
            metric_group=args.metric_group,
            bandwidth=args.bandwidth,
            vtune_path=binary,
            result_dir=args.result_dir,
            check=not args.no_check,
            timeout=args.timeout,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
