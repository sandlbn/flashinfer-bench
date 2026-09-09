"""Load-time layout transforms for vLLM's unquantized linear weights.

Two of them live here, because they act at the same seam on the same tensor and only one of
them can have a given weight: the channel-spreading **row pad**, and the **mm operand**
conversion that puts a projection on the GEMM kernel the part's peak was measured on. Both
are decided per weight by a load-time measurement, both are opt-in behind their own switch,
and both count what they did so a serving A/B can tell a null result from a silent no-op.

vLLM materialises every unquantized linear weight through
``UnquantizedLinearMethod.process_weights_after_loading``, and already uses that hook for a
layout transform of its own on Intel (``VLLM_XPU_FORCE_N_CONTIG_WEIGHT``). That makes it
the right seam for another one: it runs once per layer, after the values are final and on
the device that will read them, and before any forward pass.

No model, layer or shape is named anywhere in this file. A weight is nominated by an
arithmetic or structural property of the tensor itself, and every nomination is then kept
or dropped on a measurement taken at that weight's own shape, in the process that will
serve it. Every outcome -- examined, applied, declined, unsupported -- is counted, because
a transform that silently did not apply produces a clean null that reads like an honest
negative.

**The row pad** (``FIB_VLLM_PAD_WEIGHT_ROWS``) moves a weight whose row pitch lands on the
device's memory-channel period off that period. The period is the part's calibrated one; on
a part where none was resolved the hook examines nothing, counts every weight as
``unsupported period-unmeasured`` and says so once -- a period borrowed from another part
would pad the wrong weights with no sign that it had. It costs nothing per call: nothing is
patched on the forward path, and the one-time forward hook removes itself after confirming
that the tensor the GEMM receives still carries the padded pitch.

**The mm operand** (``FIB_VLLM_MM_ENTRY``) re-lays a projection so that the GEMM library
selects a different kernel for it, and takes ``aten.mm`` to it rather than the linear entry
point -- the two halves were measured together and neither pays alone. That second half is
on the forward path, so this one also patches ``UnquantizedLinearMethod.apply``: a converted
layer's call reaches ``mm`` and reports, once, what operand it handed it; every other layer
costs one dictionary lookup and reaches vLLM's own call unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Set, Tuple

from flashinfer_bench.integration.patch_manager import PatchSpec, get_manager
from flashinfer_bench.integration.weight_layout import (
    PadProbe,
    Probe,
    channel_period_bytes,
    mm_entry_wins,
    mm_operand,
    mm_operand_ready,
    pad_rows_off_channel_period,
    probe_verdict,
    row_pitch_bytes,
    streaming_pad_wins,
    to_mm_operand_layout,
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

_PERIOD_WARNED: Set[str] = set()
"""Devices already warned about having no period, so a model's worth of weights logs once."""


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
    if period is None:
        stats.record(FAMILY, "unsupported", "period-unmeasured")
        if str(data.device) not in _PERIOD_WARNED:
            _PERIOD_WARNED.add(str(data.device))
            logger.warning(
                "No memory channel period is known for %s; weights are left as loaded. "
                "Calibration resolved none (a shared device, or no slow pitch in its sweep); "
                "set FIB_CHANNEL_PERIOD_BYTES from a measured sweep to supply one.",
                data.device,
            )
        return False
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


PAD = "pad"
MM = "mm"

_LIVE: Set[str] = set()
"""Which transforms an ``install_*`` call has switched on. Both act at one seam, and
:class:`PatchManager` installs one wrapper per target, so the wrapper asks this rather than
each transform patching over the other and the second silently doing nothing."""


def reset_installed() -> None:
    """Forget which transforms are live. For tests that share a process."""
    _LIVE.clear()


def _run_load_transforms(layer: Any) -> None:
    """Every live transform, in the order a weight can take them.

    They are alternatives, not a pipeline: the pad leaves a weight with a row pitch no
    longer equal to its width, which is exactly what the mm conversion declines to re-lay.
    Running both switches on therefore gives the pad the weights it wants and records the
    rest as ``not-row-major``, rather than stacking two layouts neither was measured in.
    """
    for name, transform, family in (
        (PAD, pad_layer_weight, FAMILY),
        (MM, convert_layer_to_mm_entry, MM_FAMILY),
    ):
        if name not in _LIVE:
            continue
        try:
            transform(layer)
        except Exception as exc:  # never fail a load over an optimisation
            stats.record(family, "unsupported", f"error:{type(exc).__name__}")
            logger.warning("%s skipped on %s: %s", family, type(layer).__name__, exc)


def _make_wrapper(spec: PatchSpec, orig: Callable[..., Any]) -> Callable[..., Any]:
    def process_weights_after_loading(self_, layer):
        result = orig(self_, layer)
        _run_load_transforms(layer)
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
        _LIVE.add(PAD)
        logger.info("flashinfer-bench will pad camping linear weights at load")
    return patched


# --------------------------------------------------------------- the mm operand and entry
#
# The second transform. Where the row pad changes the *addresses* a GEMM reads and leaves
# the kernel alone, this one changes which kernel oneDNN selects -- and the entry point the
# call takes to reach it. Both halves are needed: measured on the same converted operand,
# the linear entry point handed the layout's whole gain back, so a delivery that re-lays the
# weight and leaves ``F.linear`` in place ships the copy and none of the win.
#
# It is keyed on neither model, module nor shape. Every unquantized linear weight is
# nominated by a structural test that costs nothing (2-D, on the device, no bias, still in
# the layout a loader produced), and each nomination is then decided by timing the two arms
# at that weight's own shape -- because there is no arithmetic that predicts which kernel a
# library selects, and the same lever is known to measure differently at other shapes.
#
# The decision is taken at every row count the deployment will present, not one, and they
# come from the running configuration rather than from this file: the scheduler's token
# budget, which bounds the M a prefill GEMM sees, and its limit on concurrent sequences,
# which bounds the M a decode GEMM sees. A weight keeps the conversion when some regime
# measured a win and none measured a loss. A weight has one layout for every regime it is
# read in, so a lever that gains in one and regresses in another is not one this can take
# per weight -- it declines, and says which way each regime went.

# One consequence of deciding by measurement is worth stating plainly, because it is what
# stands between this and a default: the decision is a measurement, so a shape whose two
# arms sit near each other can be kept on one load and declined on the next. The converted
# operand is not always bit-identical to the stored one, so two runs of the same deployment
# can then generate different tokens -- not because a kernel is wrong, but because the
# process decided differently. A deployment that needs reproducibility across restarts wants
# the decision pinned (a recorded verdict per shape, or ``force``), not re-measured.

MM_ENV_VAR = "FIB_VLLM_MM_ENTRY"
"""Set truthy to convert linear weights whose measured decision says to, and route those
layers' GEMM through ``aten.mm``. ``force`` converts every nominated weight without
measuring -- for establishing the rule, not for serving. Off by default."""

MM_PROBE_TOKENS_ENV = "FIB_VLLM_MM_PROBE_TOKENS"
"""Comma-separated row counts to decide at, when the row counts a deployment presents are
not the ones its configuration bounds -- a benchmark of sixteen short prompts never reaches
its own token budget. Unset, they are the scheduler's token budget and sequence limit."""

MM_FAMILY = "linear_mm_entry"
"""Name the counters report under: one call per weight examined, one applied per converted."""

_TARGET_APPLY = PatchSpec(
    path="vllm.model_executor.layers.linear.UnquantizedLinearMethod.apply",
    kind="method",
    name="unquantized_linear_apply",
)

_MM_STATE = "_fib_mm_operand"
"""Where a converted layer keeps ``(operand, data_ptr)``. The pointer is what the forward
path checks: something that re-materialises the parameter after load would otherwise leave
the GEMM reading a stale view."""

_MM_FORWARD = "_fib_mm_forward"
"""What the forward path has already reported about this layer, so it reports once."""

_MM_DECISIONS: Dict[Tuple[Any, ...], Tuple[Optional[Probe], ...]] = {}
"""One measured decision per (shape, strides, dtype, device, row counts). A model repeats
its projection shapes on every layer; the measurement is of the shape, not of the copy."""

_MM_WARNED: Set[str] = set()


def _mm_mode() -> Optional[str]:
    """``"probe"``, ``"force"`` or None (off)."""
    value = os.environ.get(MM_ENV_VAR, "").lower()
    if value == FORCE:
        return FORCE
    return "probe" if value in ("1", "true", "yes", "on") else None


def probe_rows() -> Optional[Tuple[int, ...]]:
    """The row counts the decision is taken at, or None.

    A projection's GEMM is handed a different number of rows in each regime a deployment
    runs -- a step's worth of prompt tokens while prefilling, one row per running sequence
    while decoding -- and the same lever does not give the same answer at both. The weight
    has one layout for all of them, so every regime it will be read in has to be measured,
    and the default pair is the scheduler's own bounds on them: its per-step token budget,
    and its limit on concurrent sequences.

    None when neither the override nor a configuration can supply them; the transform then
    has no situation to decide in and stands down.
    """
    raw = os.environ.get(MM_PROBE_TOKENS_ENV, "").strip()
    if raw:
        parts = tuple(int(v) for v in raw.replace(" ", "").split(",") if v)
        if not parts or min(parts) < 1:
            raise ValueError(
                f"{MM_PROBE_TOKENS_ENV} must be positive row counts, comma separated; got {raw!r}"
            )
        return parts
    try:
        from vllm.config import get_current_vllm_config_or_none

        scheduler = getattr(get_current_vllm_config_or_none(), "scheduler_config", None)
    except Exception:
        return None
    prefill = getattr(scheduler, "max_num_batched_tokens", None)
    decode = getattr(scheduler, "max_num_seqs", None)
    if not prefill or not decode:
        return None
    return int(prefill), int(decode)


def _mm_decide(data: Any, converted: Any, rows: Tuple[int, ...]) -> Tuple[Optional[Probe], ...]:
    key = (tuple(data.shape), tuple(data.stride()), str(data.dtype), str(data.device), rows)
    if key not in _MM_DECISIONS:
        _MM_DECISIONS[key] = tuple(mm_entry_wins(data, converted, m=m) for m in rows)
    return _MM_DECISIONS[key]


def _mm_detail(data: Any, rows: Tuple[int, ...], probes: Tuple[Probe, ...]) -> str:
    """The shape decided on and how each arm went, so a counter line names its own weight."""
    measured = "-".join(f"m{m}-{probe_verdict(p)}-{p.speedup:.3f}x" for m, p in zip(rows, probes))
    return f"{data.shape[0]}x{data.shape[1]}-{measured}"


def keep_conversion(probes: Tuple[Probe, ...]) -> bool:
    """Whether a weight keeps the conversion, given how it measured in every regime.

    Kept when some regime measured a WIN and none measured a LOSS. Both halves matter and
    neither is the prefill's privilege: a weight has one layout for every row count it will
    be read at, so a gain anywhere is worth taking and a measured regression anywhere is not
    worth paying -- and which regime the gain turns up in is the part's answer, not ours.
    NOISE is neither: a regime that could not tell blocks nothing and justifies nothing.
    """
    verdicts = [probe_verdict(p) for p in probes]
    return "WIN" in verdicts and "LOSS" not in verdicts


def convert_layer_to_mm_entry(layer: Any) -> bool:
    """Re-lay ``layer.weight`` for the mm entry point if measurement says to; True if done.

    On success the weight keeps its shape, dtype, device and values -- only its strides
    change -- and the layer carries the ``[k, n]`` operand its forward pass will hand
    ``aten.mm``. Everything that reads the weight by shape is unaffected; everything that
    calls ``F.linear`` on it still gets the right answer, just not the faster kernel.
    """
    weight = getattr(layer, "weight", None)
    if weight is None or not hasattr(weight, "data"):
        stats.record(MM_FAMILY, "unsupported", "no-weight")
        return False
    data = weight.data
    if not _in_device_memory(data):
        stats.record(MM_FAMILY, "unsupported", "cpu")
        return False
    if getattr(layer, "bias", None) is not None:
        # `mm` takes no bias, and folding one in is `addmm` -- a different entry point that
        # was not the one measured. A biased projection is left as it is rather than
        # delivered through a call nothing priced.
        stats.record(MM_FAMILY, "unsupported", "bias")
        return False
    converted = data if mm_operand_ready(data) else to_mm_operand_layout(data)
    if converted is None:
        stats.record(MM_FAMILY, "unsupported", "not-row-major")
        return False

    if _mm_mode() == FORCE:
        detail = f"{data.shape[0]}x{data.shape[1]}-forced"
    else:
        rows = probe_rows()
        if rows is None:
            stats.record(MM_FAMILY, "no-op", "batch-sizes-unknown")
            if str(data.device) not in _MM_WARNED:
                _MM_WARNED.add(str(data.device))
                logger.warning(
                    "No scheduler batch sizes are visible at the load seam and %s is unset, "
                    "so there are no row counts to decide at; weights are left as loaded.",
                    MM_PROBE_TOKENS_ENV,
                )
            return False
        probes = _mm_decide(data, converted, rows)
        if any(p is None for p in probes):
            stats.record(MM_FAMILY, "no-op", "unmeasurable")
            return False
        detail = _mm_detail(data, rows, probes)
        if not keep_conversion(probes):
            stats.record(MM_FAMILY, "no-op", detail)
            return False

    weight.data = converted
    layer.__dict__[_MM_STATE] = (mm_operand(converted), converted.data_ptr())
    layer.__dict__.pop(_MM_FORWARD, None)
    stats.record(MM_FAMILY, "applied", detail)
    return True


def _make_apply_wrapper(spec: PatchSpec, orig: Callable[..., Any]) -> Callable[..., Any]:
    """Route a converted layer's GEMM through ``aten.mm``; everything else through vLLM's.

    An unconverted layer costs one dictionary lookup and reaches vLLM's own call unchanged,
    which is what lets the decision stay per weight: a shape the measurement declined is not
    paying for a shape it accepted.
    """
    import torch

    def apply(self_, layer, x, bias=None):
        state = layer.__dict__.get(_MM_STATE)
        if state is None:
            return orig(self_, layer, x, bias)
        operand, ptr = state
        reason = None
        if bias is not None:
            reason = "bias"
        elif x.ndim != 2:
            reason = f"input-{x.ndim}d"
        elif layer.weight.data_ptr() != ptr:
            reason = "weight-replaced-after-load"
        if reason is not None:
            if layer.__dict__.get(_MM_FORWARD) != reason:
                layer.__dict__[_MM_FORWARD] = reason
                stats.record(MM_FAMILY, "forward-fell-back", reason)
            return orig(self_, layer, x, bias)
        if layer.__dict__.get(_MM_FORWARD) is not True:
            layer.__dict__[_MM_FORWARD] = True
            # What the GEMM is actually handed, read off the operand at the moment of the
            # call: its shape, and whether it is the contiguous [k, n] the selection needs.
            # A transposed view would arrive here looking identical by shape and select the
            # kernel the stack already had.
            stats.record(
                MM_FAMILY,
                "forward-mm-entry",
                f"{'contiguous' if operand.is_contiguous() else 'strided'}"
                f"-{operand.shape[0]}x{operand.shape[1]}",
            )
        return torch.ops.aten.mm.default(x, operand)

    return apply


def install_mm_entry(force: bool = False) -> bool:
    """Hook both seams the transform needs. True only if both were patched.

    The load seam decides and re-lays; the apply seam takes the entry point. Half of it is
    worse than none -- a converted weight left on ``F.linear`` pays the copy and measured no
    gain -- so if either target is missing the transform stays switched off and says so.

    Idempotent, inert without ``FIB_VLLM_MM_ENTRY`` (or ``force``), a no-op without vLLM.
    """
    if not force and _mm_mode() is None:
        logger.debug("%s is not set; leaving vLLM's linear call as it is.", MM_ENV_VAR)
        return False
    manager = get_manager()
    load = manager.patch(_TARGET, _make_wrapper)
    call = manager.patch(_TARGET_APPLY, _make_apply_wrapper)
    if not (load and call):
        logger.warning(
            "flashinfer-bench found only part of vLLM's unquantized linear path "
            "(load seam %s, call seam %s); the mm entry stays off.",
            load,
            call,
        )
        return False
    _LIVE.add(MM)
    logger.info("flashinfer-bench will decide the mm entry per linear weight at load")
    return True


__all__ = [
    "ENV_VAR",
    "FAMILY",
    "FORCE",
    "MM",
    "MM_ENV_VAR",
    "MM_FAMILY",
    "MM_PROBE_TOKENS_ENV",
    "PAD",
    "convert_layer_to_mm_entry",
    "install_mm_entry",
    "install_weight_row_padding",
    "keep_conversion",
    "pad_layer_weight",
    "probe_rows",
    "reset_installed",
]
