"""Per-device capability records.

A ``Capabilities`` instance describes what a concrete accelerator can do. It is the
seam that lets FlashInfer-Bench support a new device without new backend code: an
unreleased part is enabled by adding a capability record and a device-name mapping,
not by adding a code path.

Fields are intentionally descriptive rather than prescriptive. Benchmarking code asks
"does this device support ``float8_e4m3fn``?" instead of "is this device a B580?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional

_MIB = 1024 * 1024

DEFAULT_DTYPES: FrozenSet[str] = frozenset(
    {"float32", "float16", "bfloat16", "int64", "int32", "int16", "int8", "bool"}
)
"""Dtypes every supported accelerator is assumed to handle."""

FP8_DTYPES: FrozenSet[str] = frozenset({"float8_e4m3fn", "float8_e5m2"})
"""Narrow float dtypes that require explicit hardware/runtime support."""

FP4_DTYPES: FrozenSet[str] = frozenset({"float4_e2m1"})
"""Sub-byte float dtypes that require explicit hardware/runtime support."""


@dataclass(frozen=True)
class Capabilities:
    """What a concrete device can do.

    Parameters
    ----------
    canonical_id : str
        Stable identifier used in traces and leaderboards (e.g. ``"NVIDIA_H100"``,
        ``"INTEL_ARC_B580"``). Distinct from the raw vendor device name, which varies
        with driver version and is unsuitable as a grouping key.
    supported_dtypes : FrozenSet[str]
        Definition-schema dtype strings the device can execute. Workloads requiring a
        dtype outside this set are reported as unsupported rather than crashing.
    l2_bytes : int
        Size of the last-level cache to defeat when timing with a cold cache. A
        conservative overestimate costs a little benchmark time; an underestimate
        silently produces warm-cache numbers, so err high when unknown.
    sycl_target : Optional[str]
        Ahead-of-time SYCL target triple for this device (e.g.
        ``"intel_gpu_bmg_g21"``). ``None`` means "no AOT target known" and callers
        should fall back to SPIR-V JIT, which is how an unreleased part is targeted
        before its triple is published.
    supports_graphs : bool
        Whether the device supports captured graph replay (CUDA graphs and
        equivalents). Gates the graph capture path in tracing.
    supports_ipc_tensors : bool
        Whether a tensor on this device can be sent to another process directly. CUDA has
        IPC handles and CPU has shared memory; Intel XPU has neither today, so its tensors
        must be routed through host memory to reach a worker process.
    recommended_warmup_runs : Optional[int]
        Warmup iterations this device needs before a measurement settles, when nothing
        else specifies a value. Devices that ramp clocks from idle need considerably more
        than the generic default; ``None`` means the generic default is fine.
    preferred_sub_group_size : Optional[int]
        Sub-group (warp/wavefront) width a kernel should ask for on this device. Intel
        parts commonly offer several -- Battlemage reports {16, 32} -- and the widest is
        normally what a group reduction is cheapest over. ``None`` means the device has no
        preference worth expressing, and a kernel should not pin one.
    vector_bytes : int
        Widest single memory access, in bytes. A memory-bound kernel that reads one
        element per work-item leaves most of the pipe idle: widening RMSNorm's loads from
        2 bytes to this value measured 1.35x at prefill batch sizes on Battlemage. Use
        :meth:`vector_width` to turn it into an element count for a dtype.
    supports_large_grf : bool
        Whether the device can run kernels in a large register-file mode (256 registers
        per thread instead of 128). Only meaningful when a kernel actually spills: on
        Battlemage a CUTLASS tile spilling 8576 bytes/thread went from 8.67 ms to 0.62 ms
        with it, while a kernel that already fits gains nothing and loses occupancy.
    extra : Dict[str, Any]
        Device-specific details that have no cross-vendor meaning -- ISA revision,
        subgroup sizes, shared-local-memory budget, EU/SM counts. Kept untyped on
        purpose so a new part's specifics can be recorded before they are modeled.
    """

    canonical_id: str
    supported_dtypes: FrozenSet[str] = DEFAULT_DTYPES
    l2_bytes: int = 64 * _MIB
    sycl_target: Optional[str] = None
    supports_graphs: bool = False
    supports_ipc_tensors: bool = True
    recommended_warmup_runs: Optional[int] = None
    preferred_sub_group_size: Optional[int] = None
    vector_bytes: int = 16
    supports_large_grf: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def supports_dtype(self, dtype: str) -> bool:
        """Whether ``dtype`` (a definition-schema dtype string) can run on this device."""
        return dtype in self.supported_dtypes

    def unsupported_dtypes(self, dtypes: Any) -> FrozenSet[str]:
        """Return the subset of ``dtypes`` this device cannot execute."""
        return frozenset(d for d in dtypes if d not in self.supported_dtypes)

    def vector_width(self, itemsize: int) -> int:
        """Elements of ``itemsize`` bytes that fit in one widest-possible access.

        At least one, so a dtype wider than the access width still produces a usable
        loop rather than a zero-length one.
        """
        if itemsize <= 0:
            raise ValueError(f"itemsize must be positive, got {itemsize}")
        return max(1, self.vector_bytes // itemsize)
