"""Apply the channel-spreading row pad to vLLM's linear weights as they finish loading.

vLLM materialises every unquantized linear weight through
``UnquantizedLinearMethod.process_weights_after_loading``, and already uses that hook for a
layout transform of its own on Intel (``VLLM_XPU_FORCE_N_CONTIG_WEIGHT``). That makes it
the right seam for another one: it runs once per layer, after the values are final and on
the device that will read them, and before any forward pass.

The decision is the pitch test in :mod:`flashinfer_bench.integration.weight_layout` and
nothing else. No model, layer or shape is named here; a weight is padded because its row
pitch in bytes lands on the device's channel period, whatever produced it.

**Opt-in** via ``FIB_VLLM_PAD_WEIGHT_ROWS``, like every other change this package makes to
a serving stack. **Counted**: the exit report says how many weights were examined and how
many padded, and the first forward through each padded layer confirms the tensor the GEMM
receives still carries the padded pitch -- a transform that silently did not take produces
a clean null result otherwise. **Free per call**: nothing is patched on the forward path;
the one-time forward hook removes itself after it has looked.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

from flashinfer_bench.integration.patch_manager import PatchSpec, get_manager
from flashinfer_bench.integration.weight_layout import (
    PadProbe,
    channel_period_bytes,
    pad_rows_off_channel_period,
    row_pitch_bytes,
    streaming_pad_wins,
)

from .adapters import stats

logger = logging.getLogger(__name__)

ENV_VAR = "FIB_VLLM_PAD_WEIGHT_ROWS"
"""Set truthy to pad camping weights at load, keeping each pad only where a load-time
streaming A/B of that shape measures a win. ``force`` pads every camping weight without
measuring -- for establishing the pitch rule itself, not for serving. Off by default."""

FORCE = "force"

FAMILY = "weight_row_pad"
"""Name the counters report under. Each examined weight is one call; each padded, applied."""

_TARGET = PatchSpec(
    path="vllm.model_executor.layers.linear.UnquantizedLinearMethod.process_weights_after_loading",
    kind="method",
    name="unquantized_linear_process_weights_after_loading",
)


def _mode() -> Optional[str]:
    """``"probe"``, ``"force"`` or None (off)."""
    value = os.environ.get(ENV_VAR, "").lower()
    if value == FORCE:
        return FORCE
    return "probe" if value in ("1", "true", "yes", "on") else None


_DECISIONS: Dict[Tuple[Any, ...], Optional[PadProbe]] = {}
"""One measured decision per (shape, pitch, dtype, device); a model repeats its shapes."""


def _decide(data: Any, padded: Any) -> Optional[PadProbe]:
    key = (tuple(data.shape), row_pitch_bytes(data), str(data.dtype), str(data.device))
    if key not in _DECISIONS:
        _DECISIONS[key] = streaming_pad_wins(data, padded)
    return _DECISIONS[key]


def _confirm_at_first_forward(layer: Any, expect_pitch: int) -> None:
    """Record, once, whether the forward pass still sees the padded pitch.

    Between load and the first forward something could replace or re-materialise the
    parameter (a compile pass, a `.contiguous()` somewhere), and the load-time count would
    then over-report. The hook looks at ``layer.weight`` at the moment of the first call,
    which is the tensor ``apply`` hands the GEMM, and then removes itself.
    """
    register = getattr(layer, "register_forward_pre_hook", None)
    if register is None:
        return
    handle_box: Dict[str, Any] = {}

    def _hook(module, _inputs):
        weight = getattr(module, "weight", None)
        ok = weight is not None and row_pitch_bytes(weight) == expect_pitch
        stats.record(FAMILY, "forward-saw-padded-pitch" if ok else "forward-lost-padding")
        handle = handle_box.get("handle")
        if handle is not None:
            handle.remove()

    handle_box["handle"] = register(_hook)


def _in_device_memory(tensor: Any) -> bool:
    """Whether the tensor lives in a memory that has channels to spread over.

    Host memory is left alone: the pitch rule is about the accelerator's channel
    interleave, and a weight still on the CPU is not read by the accelerator's GEMM.
    """
    return tensor.device.type != "cpu"


def pad_layer_weight(layer: Any) -> bool:
    """Pad ``layer.weight`` in place if its pitch camps; True if it was padded."""
    weight = getattr(layer, "weight", None)
    if weight is None or not hasattr(weight, "data"):
        stats.record(FAMILY, "unsupported", "no-weight")
        return False
    data = weight.data
    if not _in_device_memory(data):
        stats.record(FAMILY, "unsupported", "cpu")
        return False
    period = channel_period_bytes(data.device)
    padded = pad_rows_off_channel_period(data, period)
    if padded is None:
        stats.record(FAMILY, "no-op", f"pitch-off-period-{period}")
        return False
    before = row_pitch_bytes(data)
    if _mode() != FORCE:
        probe = _decide(data, padded)
        if probe is None:
            stats.record(FAMILY, "no-op", "unmeasurable")
            return False
        if not probe.win:
            stats.record(
                FAMILY, "no-op", f"measured-{probe.speedup:.3f}x-spread-{probe.spread:.3f}"
            )
            return False
        detail = f"pitch-{before}-to-{row_pitch_bytes(padded)}-measured-{probe.speedup:.2f}x"
    else:
        detail = f"pitch-{before}-to-{row_pitch_bytes(padded)}-forced"
    weight.data = padded
    stats.record(FAMILY, "applied", detail)
    _confirm_at_first_forward(layer, row_pitch_bytes(padded))
    return True


def _make_wrapper(spec: PatchSpec, orig: Callable[..., Any]) -> Callable[..., Any]:
    def process_weights_after_loading(self_, layer):
        result = orig(self_, layer)
        try:
            pad_layer_weight(layer)
        except Exception as exc:  # never fail a load over an optimisation
            stats.record(FAMILY, "unsupported", f"error:{type(exc).__name__}")
            logger.warning("row pad skipped on %s: %s", type(layer).__name__, exc)
        return result

    return process_weights_after_loading


def install_weight_row_padding(force: bool = False) -> bool:
    """Hook vLLM's unquantized linear method. True if the target was patched.

    Idempotent, inert without ``FIB_VLLM_PAD_WEIGHT_ROWS`` (or ``force``), and a no-op when
    vLLM is absent.
    """
    if not force and _mode() is None:
        logger.debug("%s is not set; leaving vLLM's weights as loaded.", ENV_VAR)
        return False
    patched = get_manager().patch(_TARGET, _make_wrapper)
    if patched:
        logger.info("flashinfer-bench will pad camping linear weights at load")
    return patched


__all__ = ["ENV_VAR", "FAMILY", "FORCE", "install_weight_row_padding", "pad_layer_weight"]
