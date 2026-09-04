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
    extra: Dict[str, Any] = field(default_factory=dict)

    def supports_dtype(self, dtype: str) -> bool:
        """Whether ``dtype`` (a definition-schema dtype string) can run on this device."""
        return dtype in self.supported_dtypes

    def unsupported_dtypes(self, dtypes: Any) -> FrozenSet[str]:
        """Return the subset of ``dtypes`` this device cannot execute."""
        return frozenset(d for d in dtypes if d not in self.supported_dtypes)
