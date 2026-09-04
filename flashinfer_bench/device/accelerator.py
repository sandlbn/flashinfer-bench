"""Accelerator abstraction and registry.

FlashInfer-Bench addresses devices by string (``"cuda:0"``, ``"xpu:1"``, ``"cpu"``).
Everything backend-specific behind those strings -- stream synchronization, device
selection, cache management, timing methodology, capability reporting -- lives behind
:class:`Accelerator`, so benchmark, evaluator, and runner code contains no vendor
branches.

Resolve a backend with :func:`get_accelerator`, which dispatches on the device string's
type prefix::

    get_accelerator("xpu:0").synchronize("xpu:0")
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

from flashinfer_bench.env import get_fib_device_backend

from .capabilities import Capabilities
from .timer import Timer

_TRADEMARK_RE = re.compile(r"\((?:R|TM|C)\)", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")


def canonicalize_device_name(name: str) -> str:
    """Normalize a raw vendor device name into a stable identifier.

    Raw names carry trademark marks and vary with driver version, which makes them poor
    grouping keys for leaderboards. This collapses them to the convention already used in
    the trace docs (``NVIDIA_H100``).

    >>> canonicalize_device_name("Intel(R) Arc(TM) B580 Graphics")
    'INTEL_ARC_B580'
    """
    cleaned = _TRADEMARK_RE.sub(" ", name).upper()
    token = _NON_ALNUM_RE.sub("_", cleaned).strip("_")
    # Vendor suffixes that add no identity and would fragment grouping.
    for noise in ("_GRAPHICS", "_GPU"):
        if token.endswith(noise):
            token = token[: -len(noise)]
    return token or "UNKNOWN"


def parse_device(device: Any) -> Tuple[str, Optional[int]]:
    """Split a device designator into ``(type, index)``.

    Accepts device strings (``"cuda:0"``, ``"cpu"``) and anything exposing ``.type`` and
    ``.index`` (``torch.device``). Unlike ``device.split(":")[1]``, a device string with
    no index is valid and yields ``None``.
    """
    if device is None:
        raise ValueError("device is None")

    dev_type = getattr(device, "type", None)
    if dev_type is not None:
        index = getattr(device, "index", None)
        return str(dev_type), (int(index) if index is not None else None)

    text = str(device).strip()
    if not text:
        raise ValueError("device is empty")
    if ":" not in text:
        return text, None

    head, _, tail = text.partition(":")
    if not tail.isdigit():
        raise ValueError(f"Invalid device string '{device}'")
    return head, int(tail)


class Accelerator(ABC):
    """A compute backend addressed by device strings of one type.

    One instance serves every device of its type; methods take the device string so a
    single instance can drive ``cuda:0`` and ``cuda:1``.
    """

    type: ClassVar[str]
    """Device-string prefix this backend serves (``"cuda"``, ``"xpu"``, ``"cpu"``)."""

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Whether this backend can be used in the current environment."""
        ...

    @abstractmethod
    def list_devices(self) -> List[str]:
        """Device strings for every visible device of this type."""
        ...

    @abstractmethod
    def device_count(self) -> int:
        """Number of visible devices of this type."""
        ...

    @abstractmethod
    def set_device(self, device: str) -> None:
        """Make ``device`` current for the calling process."""
        ...

    @abstractmethod
    def synchronize(self, device: Optional[str] = None) -> None:
        """Block until all work queued on ``device`` has completed."""
        ...

    @abstractmethod
    def empty_cache(self) -> None:
        """Release cached device memory back to the driver."""
        ...

    @abstractmethod
    def device_name(self, device: str) -> str:
        """Raw vendor name for ``device``, as reported by the driver."""
        ...

    @abstractmethod
    def capabilities(self, device: str) -> Capabilities:
        """Capability record for ``device``."""
        ...

    @abstractmethod
    def make_timer(self, device: str) -> Timer:
        """Timing strategy appropriate for ``device``."""
        ...

    def canonical_id(self, device: str) -> str:
        """Stable grouping identifier for ``device``."""
        return self.capabilities(device).canonical_id

    def lib_versions(self) -> Dict[str, str]:
        """Backend-specific library versions, merged into the trace environment."""
        return {}

    def supports_graphs(self, device: str) -> bool:
        """Whether captured graph replay is available on ``device``."""
        return self.capabilities(device).supports_graphs


_REGISTRY: Dict[str, Type[Accelerator]] = {}
_INSTANCES: Dict[str, Accelerator] = {}

_PREFERENCE: Tuple[str, ...] = ("cuda", "xpu", "cpu")
"""Backend search order when ``FIB_DEVICE_BACKEND`` is ``auto``."""


def register_accelerator(cls: Type[Accelerator]) -> Type[Accelerator]:
    """Register an accelerator class under its :attr:`Accelerator.type`."""
    _REGISTRY[cls.type] = cls
    return cls


def registered_types() -> List[str]:
    """Backend types known to the registry, in preference order."""
    known = [t for t in _PREFERENCE if t in _REGISTRY]
    known.extend(sorted(t for t in _REGISTRY if t not in _PREFERENCE))
    return known


def available_types() -> List[str]:
    """Backend types usable in this environment, in preference order."""
    return [t for t in registered_types() if _REGISTRY[t].is_available()]


def default_device_type() -> str:
    """Backend type to use when no device is specified.

    Honors ``FIB_DEVICE_BACKEND``. When it names a backend explicitly, that backend is
    used even if unavailable, so a misconfigured environment fails with a clear error
    instead of silently benchmarking on the CPU.
    """
    configured = get_fib_device_backend()
    if configured != "auto":
        if configured not in _REGISTRY:
            raise ValueError(
                f"FIB_DEVICE_BACKEND='{configured}' is not a known backend. "
                f"Known backends: {', '.join(registered_types())}"
            )
        return configured

    for dev_type in available_types():
        if dev_type != "cpu" and _REGISTRY[dev_type]().device_count() > 0:
            return dev_type
    return "cpu"


def get_accelerator(device: Any = None) -> Accelerator:
    """Resolve the accelerator serving ``device``.

    Parameters
    ----------
    device : Any, optional
        Device string or ``torch.device``. When omitted, the default backend is used.
    """
    dev_type = parse_device(device)[0] if device is not None else default_device_type()

    instance = _INSTANCES.get(dev_type)
    if instance is None:
        cls = _REGISTRY.get(dev_type)
        if cls is None:
            raise ValueError(
                f"No accelerator registered for device type '{dev_type}'. "
                f"Known backends: {', '.join(registered_types())}"
            )
        instance = cls()
        _INSTANCES[dev_type] = instance
    return instance


def list_devices() -> List[str]:
    """Device strings for the default backend."""
    return get_accelerator().list_devices()


def device_synchronize(device: str) -> None:
    """Block until all work queued on ``device`` has completed.

    Convenience wrapper over ``get_accelerator(device).synchronize(device)`` for the many
    call sites that only need a barrier.
    """
    get_accelerator(device).synchronize(device)


def reset_registry_cache() -> None:
    """Drop memoized accelerator instances. Intended for tests."""
    _INSTANCES.clear()
