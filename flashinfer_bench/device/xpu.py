"""Intel GPU (XPU) backend -- Battlemage today, Xe3P next.

Both parts share one software stack (``torch.xpu`` / SYCL / Level Zero), so they share
one accelerator. What differs between them is data, not code: :data:`_PART_PROFILES`
maps a canonical device id to its capability overrides. Enabling a new Intel part means
adding a row there.

A part with no row still works. It gets the conservative defaults below, and its SYCL
target stays ``None``, which makes native kernels compile to SPIR-V and JIT at load
time -- the path that lets an unreleased device run on the day it arrives.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, ClassVar, Dict, List, Optional

from .accelerator import Accelerator, canonicalize_device_name, parse_device, register_accelerator
from .capabilities import DEFAULT_DTYPES, FP8_DTYPES, Capabilities
from .timer import EventTimer, Timer

logger = logging.getLogger(__name__)

_MIB = 1024 * 1024

_SLOW_WARMUP_SECONDS = 5.0
"""Above this, warmup is worth explaining rather than leaving as an unexplained stall."""

_RECOMMENDED_WARMUP_RUNS = 200
"""Warmup iterations before an Intel GPU measurement settles.

Discrete Arc parts ramp clocks from idle, and the first couple of hundred iterations run
measurably slower than the steady state. Integrated parts show no such ramp -- latency is
flat from the first iteration -- but they do take large sporadic excursions while the
package shifts power between CPU and GPU, which no amount of warmup removes.
The higher default costs milliseconds and protects the discrete case; it is only a
default, so any config layer or CLI flag overrides it.
"""

_CONSERVATIVE_L2_BYTES = 64 * _MIB
"""Cache-flush size of last resort, when neither the driver nor a profile reports one.

Overestimating costs a little benchmark time; underestimating silently produces
warm-cache latencies, so unknown parts round up.
"""

_PART_PROFILES: Dict[str, Dict[str, object]] = {
    # Battlemage (Xe2-HPG), validated hardware.
    #
    # FP8 is emulated, not native. CUTLASS-SYCL guards every fp8 and block-scaled-MX
    # (BDPAS) atom behind SYCL_INTEL_TARGET == 35 (Crescent Island), so on Xe2 the fp8
    # path upconverts to fp16 and runs the fp16 DPAS. torch._scaled_mm is bit-exact
    # against an fp32-upcast reference and runs at a fraction of bf16 throughput. It runs
    # and it is correct, so refusing it would block onboarding an FP8 model for no reason
    # -- but a latency measured here is the emulation's, not the format's, and the
    # fraction is something to benchmark, not to quote.
    "INTEL_ARC_B580": {
        "l2_bytes": 18 * _MIB,
        "sycl_target": "bmg",
        "emulated_dtypes": FP8_DTYPES,
        "extra": {"architecture": "Xe2-HPG", "codename": "Battlemage", "device_ip_version": 20},
    },
    "INTEL_ARC_B570": {
        "l2_bytes": 18 * _MIB,
        "sycl_target": "bmg",
        "emulated_dtypes": FP8_DTYPES,
        "extra": {"architecture": "Xe2-HPG", "codename": "Battlemage", "device_ip_version": 20},
    },
    "INTEL_ARC_PRO_B50": {
        "l2_bytes": 18 * _MIB,
        "sycl_target": "bmg",
        "emulated_dtypes": FP8_DTYPES,
        "extra": {"architecture": "Xe2-HPG", "codename": "Battlemage", "device_ip_version": 20},
    },
    "INTEL_ARC_PRO_B60": {
        "l2_bytes": 18 * _MIB,
        "sycl_target": "bmg",
        "emulated_dtypes": FP8_DTYPES,
        "extra": {"architecture": "Xe2-HPG", "codename": "Battlemage", "device_ip_version": 20},
    },
    # Xe3P "Crescent Island", device IP version 35. Pre-silicon: sgl-kernel-xpu builds
    # for it behind a SGL_PRE_SILICON flag. Cache size and dtype support are still
    # unpublished, so the defaults stay conservative; only the target name is known.
    "INTEL_XE3P_CRESCENT_ISLAND": {
        "sycl_target": "cri",
        "extra": {
            "architecture": "Xe3P",
            "codename": "Crescent Island",
            "device_ip_version": 35,
            "status": "pre-silicon",
        },
    },
}
"""Per-part capability overrides, keyed by canonical device id.

``sycl_target`` holds the device name oneAPI's offline compiler accepts
(``-fsycl-targets=spir64_gen`` together with ``-device <name>``), matching the targets
sgl-kernel-xpu builds for.

Intel identifies GPU generations by a device IP version, readable from
``torch.xpu.get_device_properties(i).version``: 20 is Xe2 (Battlemage), 30 is Xe3.0
(Panther Lake / Wildcat Lake integrated graphics), 35 is Xe3.5 (Crescent Island).
"""


@register_accelerator
class XpuAccelerator(Accelerator):
    """Intel GPUs via the PyTorch XPU backend."""

    type: ClassVar[str] = "xpu"

    @staticmethod
    def is_available() -> bool:
        try:
            import torch
        except ImportError:
            return False
        xpu = getattr(torch, "xpu", None)
        if xpu is None:
            return False
        try:
            return bool(xpu.is_available())
        except Exception:
            return False

    def device_count(self) -> int:
        import torch

        if not self.is_available():
            return 0
        return int(torch.xpu.device_count())

    def list_devices(self) -> List[str]:
        return [f"xpu:{i}" for i in range(self.device_count())]

    def _index(self, device: str) -> int:
        index = parse_device(device)[1]
        if index is not None:
            return index
        import torch

        return int(torch.xpu.current_device())

    def set_device(self, device: str) -> None:
        import torch

        torch.xpu.set_device(self._index(device))

    def synchronize(self, device: Optional[str] = None) -> None:
        import torch

        if device is None:
            torch.xpu.synchronize()
        else:
            torch.xpu.synchronize(device=device)

    def empty_cache(self) -> None:
        import torch

        torch.xpu.empty_cache()

    def device_name(self, device: str) -> str:
        import torch

        return str(torch.xpu.get_device_name(self._index(device)))

    def capabilities(self, device: str) -> Capabilities:
        return _xpu_capabilities(self._index(device))

    def make_timer(self, device: str) -> Timer:
        # Tier 1 of the timing plan: device events, cache-flushed between iterations.
        # Includes launch overhead, which is why every trace records its timer name.
        # Tiers 2 and 3 (kineto/PTI, then PTI directly) land once measured on hardware.
        import torch

        return EventTimer(torch.xpu, l2_bytes=self.capabilities(device).l2_bytes)

    def warmup(self, device: str) -> None:
        """Compile the kernels evaluation depends on, before anything is timed.

        Intel's runtime compiles SPIR-V to device code on first use and caches it under
        ``~/.cache/neo_compiler_cache``. On a cold cache that first compile takes minutes
        -- long enough to blow past the default evaluation timeout and report a working
        solution as TIMEOUT. This runs the operations the evaluator itself performs
        (elementwise math, matmul, reductions, the comparisons behind the correctness
        check) so the cost is paid once, at worker startup, outside any timed region.
        """
        import time

        import torch

        started = time.perf_counter()
        try:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                a = torch.ones(256, dtype=dtype, device=device)
                b = torch.ones(256, dtype=dtype, device=device)
                c = a + b
                c = c * b
                c.sum()
                c.max()
                # The correctness check runs in fp32 regardless of the tensor dtype.
                f = c.to(torch.float32)
                err = torch.abs(f - f)
                ((err > 1e-3) & (err > 1e-3)).sum()
                torch.isnan(f).any()
                torch.isinf(f).any()
                m = torch.ones(32, 32, dtype=dtype, device=device)
                m @ m
                c.to("cpu")
            self.synchronize(device)
        except Exception as e:  # pragma: no cover - warmup must never fail a run
            logger.debug("XPU warmup skipped (%s)", e)
            return

        elapsed = time.perf_counter() - started
        if elapsed > _SLOW_WARMUP_SECONDS:
            logger.info(
                "Compiled XPU kernels in %.0fs on first use. This is a one-time cost per "
                "machine, cached under ~/.cache/neo_compiler_cache; later runs start in "
                "about a second.",
                elapsed,
            )

    def lib_versions(self) -> Dict[str, str]:
        libs: Dict[str, str] = {}
        try:
            import torch.version as tv

            for attr in ("xpu", "sycl"):
                value = getattr(tv, attr, None)
                if value:
                    libs[attr] = str(value)
        except Exception:
            pass
        try:
            import torch

            props = torch.xpu.get_device_properties(torch.xpu.current_device())
            driver = getattr(props, "driver_version", None)
            if driver:
                libs["level_zero_driver"] = str(driver)
        except Exception:
            pass
        try:
            # Which upstream kernel libraries were present, and at what version. A trace
            # measuring a provider's kernel is otherwise indistinguishable from one
            # measuring a locally patched build of it, which is what makes a reported
            # provider bug checkable by someone else.
            from flashinfer_bench.integration.providers import provider_provenance

            for name, version in provider_provenance().items():
                libs[name] = version
        except Exception:
            pass
        return libs


_GENERIC_NAMES = frozenset({"INTEL", "INTEL_GRAPHICS", "GRAPHICS", "UNKNOWN"})
"""Device names that identify no particular part.

Integrated Intel GPUs commonly report just "Intel(R) Graphics", which every such part
would share. Those get disambiguated by PCI device id so traces from different machines
do not collide under one identifier.
"""

_ISA_PROPERTIES = (
    "architecture",
    "device_id",
    "gpu_eu_count",
    "gpu_subslice_count",
    "max_compute_units",
    "max_work_group_size",
    "max_num_sub_groups",
    "sub_group_sizes",
    "local_mem_size",
    "total_memory",
    "memory_bus_width",
    "is_integrated_gpu",
    "has_fp16",
    "has_fp64",
    "has_atomic64",
    "has_bfloat16_conversions",
    "has_subgroup_matrix_multiply_accumulate",
    "has_subgroup_matrix_multiply_accumulate_tensor_float32",
    "has_subgroup_2d_block_io",
    "vendor",
    "version",
    "platform_name",
)
"""Driver-reported properties worth recording.

These are the details a kernel author needs -- sub-group widths, shared local memory
budget, whether the part has XMX/DPAS matrix units -- and the ones that distinguish one
Intel generation from the next. Recorded verbatim rather than modelled, so a new part's
specifics land in a trace before anyone has decided how to represent them.
"""


def _canonical_xpu_id(name: str, props: object) -> str:
    """Stable identifier for an Intel GPU, disambiguated when the name is generic."""
    base = canonicalize_device_name(name)
    if base in _GENERIC_NAMES:
        device_id = getattr(props, "device_id", None)
        if isinstance(device_id, int) and device_id > 0:
            return f"{base}_{device_id:04X}"
    return base


@lru_cache(maxsize=None)
def _xpu_capabilities(index: int) -> Capabilities:
    import torch

    name = torch.xpu.get_device_name(index)
    try:
        props: Any = torch.xpu.get_device_properties(index)
    except Exception:
        props = None

    canonical = (
        _canonical_xpu_id(name, props) if props is not None else (canonicalize_device_name(name))
    )
    profile = _PART_PROFILES.get(canonical, {})

    extra: Dict[str, object] = dict(profile.get("extra", {}))  # type: ignore[arg-type]
    if props is not None:
        for attr in _ISA_PROPERTIES:
            value = getattr(props, attr, None)
            if value is not None:
                extra.setdefault(attr, value)
    if canonical not in _PART_PROFILES:
        extra.setdefault("profile", "driver-reported")

    # The driver knows the cache size; the profile table is only a fallback for parts
    # whose runtime does not report one. Guessing high wastes benchmark time, guessing
    # low silently produces warm-cache latencies.
    l2_bytes = int(getattr(props, "last_level_cache_size", 0) or 0) if props else 0
    if l2_bytes <= 0:
        l2_bytes = int(profile.get("l2_bytes", _CONSERVATIVE_L2_BYTES))  # type: ignore[arg-type]

    # The driver enumerates the sub-group widths this part supports; take the widest,
    # which is what a group reduction is cheapest over. Read rather than tabulated, so a
    # new part needs no entry to get this right.
    #
    # This is the *elementwise* width and only that. Matrix kernels must pin 16 on every
    # Xe part -- DPAS has execution size 16 and CUTLASS-SYCL hardcodes
    # `constexpr int sg_size = 16` with no 32-lane path anywhere -- so a caller sizing a
    # tl.dot or XMX kernel against this field is reading the wrong number. See
    # `.claude/skills/optimize-intel-kernels/architectures.md`.
    sub_groups = extra.get("sub_group_sizes") or []
    try:
        preferred_sub_group = max(int(v) for v in sub_groups) if sub_groups else None
    except (TypeError, ValueError):
        preferred_sub_group = None

    return Capabilities(
        canonical_id=canonical,
        # Conservative until measured: a dtype that torch accepts but lowers to an
        # emulated path would otherwise be benchmarked as if it were native.
        supported_dtypes=frozenset(profile.get("supported_dtypes", DEFAULT_DTYPES)),  # type: ignore[arg-type]
        # Correct but not at native rate. Kept out of supported_dtypes so a result is
        # never read as a native measurement of that format.
        emulated_dtypes=frozenset(profile.get("emulated_dtypes", frozenset())),  # type: ignore[arg-type]
        l2_bytes=l2_bytes,
        sycl_target=profile.get("sycl_target"),  # type: ignore[arg-type]
        supports_graphs=False,
        # torch has no XPU tensor IPC: pickling one to a worker process raises
        # "_share_fd_: only available on CPU". Baselines route through host memory.
        supports_ipc_tensors=False,
        recommended_warmup_runs=_RECOMMENDED_WARMUP_RUNS,
        preferred_sub_group_size=preferred_sub_group,
        # 16 bytes is the widest single access on every Intel Xe part; the win from using
        # it is large and the cost of a narrower access is silent, so this is not
        # per-part data today.
        vector_bytes=16,
        # Large GRF has been available since Xe-HPG. A pre-silicon part with no capability
        # record still gets it, which is correct: the flag only matters when a kernel
        # spills, and refusing it there would hide the fix.
        supports_large_grf=True,
        extra=extra,
    )
