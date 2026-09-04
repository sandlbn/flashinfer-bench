"""Cross-validation of reference implementations on a new accelerator.

Correctness of a solution is measured against the definition's PyTorch reference running
*on the same device*. That check is only as trustworthy as the reference: if an operator
is subtly wrong, lower-precision, or silently emulated on a new backend, then a wrong
solution validates against a wrong reference and the error is invisible.

This module closes that hole. Before a definition is benchmarked on a new accelerator, its
reference is run on that accelerator and on the host with bit-identical inputs, and the
two outputs are compared. Definitions that disagree are quarantined rather than
benchmarked, so no unvalidated number is ever produced.

Inputs are generated once on the host and copied to the device, so the two runs see
exactly the same values -- see :func:`flashinfer_bench.bench.utils.gen_inputs`.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence

import torch

from flashinfer_bench.bench.config import BenchmarkConfig, ResolvedEvalConfig, device_eval_defaults
from flashinfer_bench.bench.utils import compute_error_stats, gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Definition, TraceSet, Workload
from flashinfer_bench.device import get_accelerator

logger = logging.getLogger(__name__)

BASELINE_DEVICE = "cpu"
"""Device used as the trusted comparison point.

The host is the reference of last resort: it is the one backend whose operator coverage
and numerics are not in question when a new accelerator is being brought up.
"""


class ReferenceCheckStatus(str, Enum):
    """Outcome of cross-validating one definition's reference on a device."""

    PASSED = "PASSED"
    """Device and host references agree within tolerance."""
    MISMATCH = "MISMATCH"
    """References disagree. The definition must not be benchmarked on this device."""
    UNSUPPORTED_DTYPE = "UNSUPPORTED_DTYPE"
    """The device does not support a dtype the definition requires."""
    TARGET_ERROR = "TARGET_ERROR"
    """The reference could not run on the device (missing operator, runtime failure)."""
    BASELINE_ERROR = "BASELINE_ERROR"
    """The reference could not run on the host, so there is nothing to compare against."""
    BUILD_ERROR = "BUILD_ERROR"
    """The reference implementation could not be built."""
    NO_WORKLOAD = "NO_WORKLOAD"
    """The dataset carries no workload for this definition, so nothing could be checked."""

    @property
    def is_clearance(self) -> bool:
        """Whether this outcome clears the definition for benchmarking on the device."""
        return self is ReferenceCheckStatus.PASSED


@dataclass
class ReferenceCheckResult:
    """Result of cross-validating one definition's reference on one device."""

    definition: str
    device: str
    hardware: str
    status: ReferenceCheckStatus
    max_absolute_error: float = 0.0
    max_relative_error: float = 0.0
    matched_ratio: float = 1.0
    workload_uuid: Optional[str] = None
    detail: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status.is_clearance


@dataclass
class ReferenceAttestation:
    """A dated record of which definitions are cleared to benchmark on which hardware.

    Written after a check run and consumed by CI. It is the artifact that makes "the
    reference was validated on this hardware" checkable rather than asserted.
    """

    hardware: str
    device: str
    backend: str
    timestamp: str
    torch_version: str
    results: List[ReferenceCheckResult] = field(default_factory=list)

    @property
    def cleared(self) -> List[str]:
        """Definitions cleared for benchmarking on this hardware."""
        return sorted(r.definition for r in self.results if r.ok)

    @property
    def quarantined(self) -> List[str]:
        """Definitions that must not be benchmarked on this hardware."""
        return sorted(r.definition for r in self.results if not r.ok)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["results"] = [
            {**asdict(r), "status": r.status.value} for r in self.results  # enum -> str
        ]
        data["cleared"] = self.cleared
        data["quarantined"] = self.quarantined
        return data


def required_dtypes(definition: Definition) -> FrozenSet[str]:
    """Every dtype a definition's inputs and outputs require."""
    dtypes = {spec.dtype for spec in definition.inputs.values()}
    dtypes |= {spec.dtype for spec in definition.outputs.values()}
    return frozenset(dtypes)


def unsupported_dtypes(definition: Definition, device: str) -> FrozenSet[str]:
    """Dtypes this definition needs that ``device`` cannot execute.

    A non-empty result means the definition should be skipped on this device rather than
    benchmarked: a dtype the hardware lacks is either a hard failure or, worse, a silent
    emulation that would be reported as if it were a native kernel.
    """
    accelerator = get_accelerator(device)
    return accelerator.capabilities(device).unsupported_dtypes(required_dtypes(definition))


def _pick_workload(trace_set: TraceSet, definition_name: str) -> Optional[Workload]:
    """Pick a representative workload for a definition.

    The smallest workload by total axis extent is used: a reference cross-check is about
    numerics, not scale, and a small shape keeps the host side of the comparison fast.
    """
    traces = trace_set.workloads.get(definition_name) or []
    workloads = [t.workload for t in traces if t.workload is not None]
    if not workloads:
        return None
    return min(workloads, key=lambda w: sum(v for v in w.axes.values() if isinstance(v, int)))


def _run_reference(
    runnable: Any, definition: Definition, inputs: List[Any], device: str
) -> List[torch.Tensor]:
    """Run a reference implementation and normalise its outputs to a tensor list."""
    # Imported here rather than at module scope: the evaluators package imports the
    # runners, which import the evaluators back, so pulling it in at import time makes
    # `flashinfer_bench.bench` circular.
    from flashinfer_bench.bench.evaluators.utils import normalize_result

    with torch.no_grad():
        result = runnable(*inputs)
    get_accelerator(device).synchronize(device)
    return normalize_result(definition, result, device)


def _to_device(values: Sequence[Any], device: str) -> List[Any]:
    """Move tensor values to ``device``, leaving scalars untouched."""
    return [v.to(device=device) if isinstance(v, torch.Tensor) else v for v in values]


def check_reference(
    definition: Definition,
    workload: Workload,
    device: str,
    cfg: ResolvedEvalConfig,
    trace_set_root: Optional[Path] = None,
) -> ReferenceCheckResult:
    """Cross-validate one definition's reference between ``device`` and the host.

    Parameters
    ----------
    definition : Definition
        The definition whose reference implementation is under test.
    workload : Workload
        Workload supplying concrete axis values and inputs.
    device : str
        The device being brought up (e.g. ``"xpu:0"``).
    cfg : ResolvedEvalConfig
        Supplies the comparison tolerances and the number of trials.
    trace_set_root : Optional[Path]
        Dataset root, needed to resolve safetensors inputs.

    Returns
    -------
    ReferenceCheckResult
        The outcome. Only :attr:`ReferenceCheckStatus.PASSED` clears the definition for
        benchmarking on this device.
    """
    accelerator = get_accelerator(device)
    hardware = accelerator.canonical_id(device)

    def result(status: ReferenceCheckStatus, **kwargs: Any) -> ReferenceCheckResult:
        return ReferenceCheckResult(
            definition=definition.name,
            device=device,
            hardware=hardware,
            status=status,
            workload_uuid=workload.uuid,
            **kwargs,
        )

    missing = unsupported_dtypes(definition, device)
    if missing:
        return result(
            ReferenceCheckStatus.UNSUPPORTED_DTYPE,
            detail=f"device does not support dtype(s): {', '.join(sorted(missing))}",
        )

    try:
        runnable = BuilderRegistry.get_instance().build_reference(definition)
    except Exception as e:
        return result(ReferenceCheckStatus.BUILD_ERROR, detail=f"{type(e).__name__}: {e}")

    safe_tensors = (
        load_safetensors(definition, workload, trace_set_root)
        if any(d.type == "safetensors" for d in workload.inputs.values())
        else {}
    )

    max_abs = 0.0
    max_rel = 0.0
    min_matched = 1.0

    for trial in range(cfg.num_trials):
        # Generated once on the host, then copied. Regenerating per device would cast
        # narrow dtypes on each device separately, and the two runs would no longer be
        # comparing the same values.
        host_inputs = gen_inputs(
            definition, workload, device=BASELINE_DEVICE, safe_tensors=safe_tensors, trial=trial
        )
        device_inputs = _to_device(host_inputs, device)

        try:
            baseline_out = _run_reference(runnable, definition, host_inputs, BASELINE_DEVICE)
        except Exception:
            return result(ReferenceCheckStatus.BASELINE_ERROR, detail=_short_traceback())

        try:
            target_out = _run_reference(runnable, definition, device_inputs, device)
        except Exception:
            return result(ReferenceCheckStatus.TARGET_ERROR, detail=_short_traceback())

        if len(baseline_out) != len(target_out):
            return result(
                ReferenceCheckStatus.MISMATCH,
                detail=(
                    f"output count differs: host produced {len(baseline_out)}, "
                    f"device produced {len(target_out)}"
                ),
            )

        for index, (target, baseline) in enumerate(zip(target_out, baseline_out)):
            if tuple(target.shape) != tuple(baseline.shape):
                return result(
                    ReferenceCheckStatus.MISMATCH,
                    detail=(
                        f"output {index} shape differs: host {tuple(baseline.shape)}, "
                        f"device {tuple(target.shape)}"
                    ),
                )

            abs_err, rel_err, exceeds, matched = compute_error_stats(target.cpu(), baseline, cfg)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            min_matched = min(min_matched, matched)

            if exceeds:
                return result(
                    ReferenceCheckStatus.MISMATCH,
                    max_absolute_error=max_abs,
                    max_relative_error=max_rel,
                    matched_ratio=min_matched,
                    detail=(
                        f"output {index} differs beyond tolerance on trial {trial} "
                        f"(atol={cfg.atol}, rtol={cfg.rtol})"
                    ),
                )

    return result(
        ReferenceCheckStatus.PASSED,
        max_absolute_error=max_abs,
        max_relative_error=max_rel,
        matched_ratio=min_matched,
    )


def check_references(
    trace_set: TraceSet,
    device: str,
    config: Optional[BenchmarkConfig] = None,
    definitions: Optional[Sequence[str]] = None,
) -> ReferenceAttestation:
    """Cross-validate every definition's reference on ``device``.

    Parameters
    ----------
    trace_set : TraceSet
        Dataset supplying definitions and their workloads.
    device : str
        The device being brought up.
    config : Optional[BenchmarkConfig]
        Supplies comparison tolerances. Defaults to the packaged benchmark config.
    definitions : Optional[Sequence[str]]
        Restrict the check to these definition names.

    Returns
    -------
    ReferenceAttestation
        A dated record of which definitions are cleared on this hardware.
    """
    config = config if config is not None else BenchmarkConfig.default()
    accelerator = get_accelerator(device)

    names = sorted(definitions) if definitions else sorted(trace_set.definitions)
    results: List[ReferenceCheckResult] = []

    for name in names:
        definition = trace_set.definitions.get(name)
        if definition is None:
            logger.warning("Definition '%s' is not in the dataset; skipping", name)
            continue

        workload = _pick_workload(trace_set, name)
        if workload is None:
            results.append(
                ReferenceCheckResult(
                    definition=name,
                    device=device,
                    hardware=accelerator.canonical_id(device),
                    status=ReferenceCheckStatus.NO_WORKLOAD,
                    detail="dataset carries no workload for this definition",
                )
            )
            continue

        cfg = config.resolve_eval_config(
            definition, accelerator.canonical_id(device), device_eval_defaults(device)
        )
        result = check_reference(definition, workload, device, cfg, trace_set.root)
        results.append(result)
        logger.info(
            "%-44s %-18s max_abs=%.3e max_rel=%.3e",
            name,
            result.status.value,
            result.max_absolute_error,
            result.max_relative_error,
        )

    return ReferenceAttestation(
        hardware=accelerator.canonical_id(device),
        device=device,
        backend=accelerator.type,
        timestamp=datetime.now(timezone.utc).isoformat(),
        torch_version=torch.__version__,
        results=results,
    )


def _short_traceback(limit: int = 6) -> str:
    """Last few frames of the current exception, for a one-line-ish report field."""
    return "".join(traceback.format_exc(limit=limit).splitlines(keepends=True)[-limit:]).strip()
