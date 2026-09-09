"""Derive, for every hot op a model ran, which delivery mechanism is worth using -- and log
why every mechanism that is not was rejected.

oneDNN carries no table of which kernel to use. It evaluates its implementations against
the problem descriptor and, under ``ONEDNN_VERBOSE=dispatch``, prints why each candidate was
rejected, naming the gate that fired and its source line. This stage does the same for the
ways a kernel change can be delivered: every candidate is evaluated against **every**
mechanism, every gate writes one line, and the worklist is whatever survived. Nothing is
preferred; everything is priced from this part's calibration, and the pricing is inspectable.

Inputs, each named by the command that produces it:

    discovered.json    scripts/harness_from_model.py     ops, shapes, calls, op_share, edges,
                                                        device_time_by_kernel
    resolution.json    scripts/pull_kernel_source.py --json   class per op, source, schema,
                                                        launched kernels, bundle
    measurements.json  this script's --measure step, keyed by candidate_id:
                       scripts/kernel_trials.py benchmark of each harness against itself
                       (t_host_us, spread_us) and a torch.profiler pass over the same harness
                       (t_dev_us, device-side, per shape); spill / geom from unitrace may be
                       added by hand under the same keys
    calibration        flashinfer_bench.device.calibration.get(): bandwidth_gbs,
                       timing_floor_us, launch_floor_us, matmul_peak_tflops, dispatch_us

    python scripts/bound_candidates.py --report <dir>/discovered.json \\
        --resolution <dir>/resolution.json --out-dir <dir>/bound \\
        [--measure --harness-dir <dir>] [--pattern <op>=<run_bytes>:<stride_bytes>] \\
        [--bytes-min <op>=<bytes>] [--gaps <find_kernel_gaps json>] [--cutoff <fraction>]

Outputs, in ``--out-dir``:

    bound.json     AUTHORITATIVE. One record per gate evaluation, ACCEPT and UNROUTABLE, each
                   carrying the evaluated arithmetic and, on REJECT, ``needs``; plus the
                   per-candidate metrics (t_dev, bytes_min, t_mem, t_cmp, bound, regime).
    bound.log      the same rows, one per line, in the dispatch-style form of
                   SKILLS-REWRITE-PLAN.md section 6.8 (see LINE FORMS below).
    worklist.json  the ACCEPT records, ordered by worth.

CONTRACT -- what a consumer of these files may rely on:

  1. Every candidate that carries device-time share has at least one row: an op with
     ``op_share[op].device_us > 0`` in discovered.json, an op whose Stage 2 resolution
     launched kernels that carry device time there (a decomposition reaching oneDNN reports
     zero in op_share), or a Triton kernel with device time.
  2. The worklist is exactly the ACCEPT rows, ordered by ``worth`` descending, ties broken by
     ``candidate_id`` then ``mechanism``. Nothing else orders it, and nothing enters it that
     is not an ACCEPT row.
  3. Every mechanism evaluated for a candidate ends in exactly one ACCEPT or one REJECT. A
     REJECT names the gate that stopped it and the arithmetic that failed with both sides
     evaluated; in bound.json it also carries ``needs`` -- the quantity, the comparison and
     the threshold that was not met -- so the next stage can tell what would have to move.
     No prediction is made about whether it will.
  4. A candidate with no ACCEPT row has one UNROUTABLE row, written after its last mechanism,
     and a REJECT row for every mechanism.
  5. An unmeasurable input is rendered as ``None`` and never as a number. A mechanism whose
     delivery cost is unmeasured is rejected at ``cost_calibrated`` with ``dispatch_us=None``;
     it is unavailable, never free.
  6. Every bound.log line has exactly 13 comma-separated fields; no field contains a comma or
     a newline; the arithmetic field of a gate line splits on single spaces into exactly
     three tokens ``<lhs>=<value> <cmp> <rhs>`` where ``<rhs>`` is ``<name>=<value>`` or a bare
     value. :func:`parse_log` is the reference reader; :func:`worklist_from_log` rebuilds
     worklist.json from bound.log alone.

LINE FORMS (section 6.8):

    fib_bound,v1,<run_id>,<candidate_id>,<op>,<shape>,<class>,<regime>,<mechanism>,<gate>,PASS|REJECT,<lhs>=<v> <cmp> <rhs>=<v>,<file>:<line>
    fib_bound,v1,<run_id>,<candidate_id>,<op>,<shape>,<class>,<regime>,<mechanism>,ACCEPT,ceiling_us=<v>,worth=<v>,mechanism_us=<v>
    fib_bound,v1,<run_id>,<candidate_id>,<op>,<shape>,<class>,<regime>,-,UNROUTABLE,<n> mechanisms evaluated,<n> rejected,-

MECHANISMS and the fact that admits each (section 6.8; facts about how the system is
built, never a performance claim), with the delivery cost each pays:

    provider_patch      class = provider kernel; source bundled            0
    triton_in_place     class = Triton with file:line inside the stack     0
    library_call        class = oneDNN (a primitive,exec line)             0
    layout_transform    regime = memory-bound, layout-limited              0
    fusion_callsite     a GEMM-class producer feeds this op                0
    fusion_apply        as above, and a definition exists                  calibration.dispatch_us
    apply_substitution  a kernel runs on the device for this op            calibration.dispatch_us
    source_rewrite      the composite moves more bytes than the maths      0
                        requires (find_kernel_gaps output)
    upstream_report     class = ATen inside PyTorch, no local source       not a worklist row

GATES, in the order evaluated; the first REJECT ends the chain for that (candidate,
mechanism) pair, as in a dispatch list:

    class_admits, edge_present (fusion), epilogue_expressible (fusion), source_present
    (patch/Triton), definition_exists (apply), cost_calibrated (apply), measurable,
    headroom, net_positive, worth_cutoff -- and worklist_eligible, which is the one gate
    outside the plan's table: it turns upstream_report away from the worklist while still
    logging that the class admitted it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

LOG_TAG = "fib_bound"
LOG_VERSION = "v1"
LOG_FIELDS = 13
_FILE = pathlib.Path(__file__).name

MECHANISMS = (
    "provider_patch",
    "triton_in_place",
    "library_call",
    "layout_transform",
    "fusion_callsite",
    "fusion_apply",
    "apply_substitution",
    "source_rewrite",
    "upstream_report",
)
FUSION = ("fusion_callsite", "fusion_apply")
APPLY = ("fusion_apply", "apply_substitution")
PATCH = ("provider_patch", "triton_in_place")

# The resolver's class strings, as tokens the log can carry (no spaces, no commas).
CLASS_TOKENS = {
    "provider kernel": "provider_kernel",
    "Triton": "triton",
    "oneDNN": "onednn",
    "ATen kernel inside PyTorch": "aten",
    "Python-registered custom op": "python_op",
    "decomposition": "decomposition",
    "not on this device": "not_on_device",
    "registered, source not on this box": "no_source",
    "unresolved": "unresolved",
}
# Classes under which a kernel actually executes on the device for the op -- the precondition
# for substituting one through apply(). A decomposition has no kernel of its own to replace.
KERNEL_CLASSES = ("provider_kernel", "triton", "onednn", "aten", "python_op", "no_source")

# Regime tokens: the section 2.1 row names, spelled without spaces or commas.
R_UNMEASURED = "unmeasured"  # no t_dev, or no spread: nothing to classify (section 3)
R_UNMEASURABLE = "unmeasurable"
R_EMULATED = "emulated"
R_LAUNCH = "launch-bound"
R_SPILL = "spill-limited"
R_AT_BOUND = "at-the-bound"
R_MEM_LAYOUT = "memory-bound-layout-limited"
R_MEM_INEFF = "memory-bound-inefficient"
R_CMP_INEFF = "compute-bound-inefficient"
R_UNCLASSIFIED = "unclassified"

# What a definition's op_type is called in the op vocabulary of the stacks discovery records.
# A translation between two projects' words for one operation, in the manner of
# fusion_candidates._VOCABULARY; the shapes and dtypes do the rest of the matching.
_OP_TYPE_WORDS = {
    "rmsnorm": ("rms_norm", "rmsnorm", "layernorm", "layer_norm"),
    "rope": ("rotary", "rope"),
    "activation": ("silu_and_mul", "gelu_and_mul", "act_and_mul", "silu", "gelu"),
    "gemm": ("linear", "matmul", "addmm", "mm", "bmm", "scaled_mm"),
    "sampling": ("sampl", "argmax", "topk", "top_k", "top_p", "multinomial"),
}
_POST_OP_SYNONYMS = {"silu": "swish", "sigmoid": "logistic"}


# --------------------------------------------------------------------------- data


@dataclass
class Candidate:
    candidate_id: str
    op: str
    shape: str
    dtype: str
    tensors: List[Tuple[Tuple[int, ...], str]]
    args: list
    calls: int
    device_us_op: Optional[float]
    share_pct: Optional[float]
    share_source: str = "op_share"
    cls: str = "unresolved"
    provider: str = "unresolved"
    where: List[str] = field(default_factory=list)
    bundle: Optional[str] = None
    schema: Optional[str] = None
    launched: List[str] = field(default_factory=list)
    t_dev: Optional[float] = None
    t_dev_source: str = "none"
    t_host: Optional[float] = None
    spread_us: Optional[float] = None
    spill: Optional[int] = None
    geom: Optional[str] = None
    native: Optional[bool] = None
    bytes_min: Optional[int] = None
    bytes_min_note: str = "none"
    flops: Optional[float] = None
    pattern: Optional[Tuple[int, int]] = None
    bw_pattern: Optional[float] = None
    t_mem: Optional[float] = None
    t_mem_pattern: Optional[float] = None
    t_cmp: Optional[float] = None
    floor_us: Optional[float] = None
    floor_source: str = "none"
    launch_us: Optional[float] = None
    bound: Optional[float] = None
    bound_pattern: Optional[float] = None
    regime: str = R_UNMEASURED
    regime_test: str = "-"
    regime_rows_skipped: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    harness: Optional[str] = None

    def metrics(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "op": self.op,
            "shape": self.shape,
            "dtype": self.dtype,
            "calls": self.calls,
            "share_pct": self.share_pct,
            "share_source": self.share_source,
            "class": self.cls,
            "provider": self.provider,
            "where": self.where,
            "bundle": self.bundle,
            "harness": self.harness,
            "t_dev_us": self.t_dev,
            "t_dev_source": self.t_dev_source,
            "t_host_us": self.t_host,
            "spread_us": self.spread_us,
            "spill": self.spill,
            "geom": self.geom,
            "native": self.native,
            "bytes_min": self.bytes_min,
            "bytes_min_note": self.bytes_min_note,
            "flops": self.flops,
            "pattern": None if self.pattern is None else f"{self.pattern[0]}:{self.pattern[1]}",
            "bw_pattern_gbs": self.bw_pattern,
            "t_mem_us": self.t_mem,
            "t_mem_pattern_us": self.t_mem_pattern,
            "t_cmp_us": self.t_cmp,
            "floor_us": self.floor_us,
            "floor_source": self.floor_source,
            "launch_floor_us": self.launch_us,
            "bound_us": self.bound,
            "bound_pattern_us": self.bound_pattern,
            "regime": self.regime,
            "regime_test": self.regime_test,
            "regime_rows_skipped": self.regime_rows_skipped,
            "notes": self.notes,
        }


@dataclass
class Context:
    """Everything a gate reads besides the candidate. Built once per run."""

    cal: Any  # Calibration-like: dispatch_us, timing_floor_us, bandwidth_gbs, launch_floor_us, matmul_peak_tflops
    total_us: float
    by_kernel: Dict[str, float]
    edges: List[Dict[str, Any]]
    resolution: Dict[str, Dict[str, Any]]
    # None where the source could not be read -- Xe-Fuse absent, dataset unreadable -- as
    # distinct from read and empty. A gate renders the former as None (unavailable) and
    # never as a count of zero (evaluated, and nothing there).
    presets: Optional[List[Tuple[str, str]]]
    post_op_algorithms: Optional[Set[str]]
    definitions_matching: Callable[[Candidate], Optional[List[str]]]
    gaps: Dict[str, Dict[str, float]]
    cutoff: float
    run_id: str
    edge_threshold: int = 1


@dataclass
class Gate:
    passed: bool
    lhs: str
    lhs_value: Any
    cmp: str
    rhs: Optional[str]
    rhs_value: Any
    where: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def arithmetic(self) -> str:
        rhs = f"{self.rhs}={_fmt(self.rhs_value)}" if self.rhs else _fmt(self.rhs_value)
        return f"{self.lhs}={_fmt(self.lhs_value)} {self.cmp} {rhs}"

    def needs(self) -> Dict[str, Any]:
        direction = {
            ">": "increase",
            ">=": "increase",
            "<": "decrease",
            "<=": "decrease",
        }.get(self.cmp, "establish")
        return {
            "quantity": self.lhs,
            "observed": self.lhs_value,
            "cmp": self.cmp,
            "threshold_name": self.rhs,
            "threshold": self.rhs_value,
            "direction": direction,
        }


def _here() -> str:
    return f"{_FILE}:{sys._getframe(1).f_lineno}"


def _fmt(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "None"
        return f"{v:.6g}"
    if isinstance(v, (list, tuple, set, frozenset)):
        return "|".join(_fmt(x) for x in v) or "none"
    return str(v).replace(",", ";").replace(" ", "_")


# --------------------------------------------------------------------------- candidates


def _tensor_args(args: Iterable[Any]) -> List[Tuple[Tuple[int, ...], str]]:
    """Every tensor in `args`, depth-first, as (shape, dtype) -- discovery's ``["T", shape,
    dtype]`` spelling, including tensors nested in a list argument."""
    out: List[Tuple[Tuple[int, ...], str]] = []
    for a in args:
        if isinstance(a, list) and len(a) == 3 and a[0] == "T" and isinstance(a[1], list):
            out.append((tuple(int(x) for x in a[1]), str(a[2])))
        elif isinstance(a, list):
            out.extend(_tensor_args(a))
    return out


_ITEMSIZE = {
    "float64": 8,
    "int64": 8,
    "float32": 4,
    "int32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "bool": 1,
}


def _nbytes(shape: Tuple[int, ...], dtype: str) -> Optional[int]:
    size = _ITEMSIZE.get(dtype)
    if size is None:
        return None
    return math.prod(shape) * size


def candidate_id(op: str, shape: str, dtype: str) -> str:
    return hashlib.sha1(f"{op}|{shape}|{dtype}".encode()).hexdigest()[:10]


def _shape_sig(tensors: Sequence[Tuple[Tuple[int, ...], str]]) -> str:
    return "/".join("x".join(str(d) for d in shape) or "scalar" for shape, _ in tensors) or "-"


def _dtype_of(tensors: Sequence[Tuple[Tuple[int, ...], str]]) -> str:
    for _, dt in tensors:
        if dt.startswith(("float", "bfloat")):
            return dt
    return tensors[0][1] if tensors else "-"


def _op_key(op_dotted: str) -> str:
    parts = op_dotted.split(".")
    return f"{parts[0]}::{parts[1]}" if len(parts) > 1 else op_dotted


def _mutated_positions(schema: Optional[str]) -> Set[int]:
    """Argument positions the schema marks as written (``Tensor(a!)``, ``Tensor($0! -> )``)."""
    if not schema or "(" not in schema:
        return set()
    body = schema[schema.index("(") + 1 :]
    depth, start, args = 0, 0, []
    for i, ch in enumerate(body):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            if depth == 0:
                args.append(body[start:i])
                break
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(body[start:i])
            start = i + 1
    return {i for i, a in enumerate(args) if "!" in a.split(" ")[0] or "!" in a.split(")")[0]}


def _returns_tensor(schema: Optional[str]) -> Optional[bool]:
    if not schema or "->" not in schema:
        return None
    return schema.rsplit("->", 1)[1].strip() != "()"


def _bytes_min(args: list, schema: Optional[str]) -> Tuple[Optional[int], str]:
    """Bytes the op is obliged to move, from the shapes and the schema.

    Every tensor argument is moved once (read, or written if it is an output the caller
    passed); an argument the schema marks mutated is written back as well. The output of a
    value-returning op is not in the recorded arguments, so for such an op this is a lower
    bound and the note says so. A lower bound on bytes is a lower bound on t_mem, which keeps
    ``bound`` a bound: headroom is overstated, never understated.
    """
    total = 0
    mutated = _mutated_positions(schema)
    for i, a in enumerate(args):
        for shape, dt in _tensor_args([a]):
            n = _nbytes(shape, dt)
            if n is None:
                return None, f"unknown itemsize for {dt}"
            total += n * (2 if i in mutated else 1)
    returns = _returns_tensor(schema)
    if returns is None:
        return total, "no schema: outputs not counted (lower bound)"
    if returns:
        return total, "value-returning op: output not counted (lower bound)"
    return total, "inputs once plus mutated arguments written back"


def _gemm_like(res: Optional[Dict[str, Any]]) -> bool:
    """Whether the resolver saw a matrix primitive run for this op."""
    if not res or res.get("provider") != "oneDNN":
        return False
    text = " ".join(str(w) for w in res.get("where") or res.get("primitives") or []).lower()
    return not text or any(k in text for k in ("matmul", "gemm", "inner_product"))


def _gemm_flops(tensors: Sequence[Tuple[Tuple[int, ...], str]]) -> Optional[float]:
    mats = [s for s, _ in tensors if len(s) >= 2]
    if len(mats) < 2:
        return None
    a, b = mats[0], mats[1]
    k = a[-1]
    if len(b) == 2 and b[1] == k:
        n = b[0]
    elif b[-2] == k:
        n = b[-1]
    else:
        return None
    m = math.prod(a[:-1])
    return 2.0 * m * n * k


def candidates_from_report(
    report: Dict[str, Any],
    resolution: Dict[str, Dict[str, Any]],
    measurements: Dict[str, Dict[str, Any]],
    bytes_overrides: Optional[Dict[str, int]] = None,
) -> List[Candidate]:
    """One candidate per (op, shape) that carries device time, plus each Triton kernel."""
    bytes_overrides = bytes_overrides or {}
    shares = report.get("op_share") or {}
    op_calls = report.get("op_calls") or {}
    by_kernel = report.get("device_time_by_kernel") or {}
    total_us = float(report.get("device_time_total_us") or 0.0)
    out: List[Candidate] = []
    for row in report.get("ops") or []:
        op = row["op"]
        res = resolution.get(_op_key(op))
        share = shares.get(op) or {}
        device_us, share_source = float(share.get("device_us") or 0.0), "op_share"
        if device_us <= 0 and res:
            # Discovery charges an op only the kernels it could tie to it; a decomposition
            # that reaches oneDNN reports zero there. Stage 2 ran the op and recorded which
            # kernels it launched, so charge it those -- the same prefix match the resolver
            # uses for its own share column.
            device_us = _launched_us(res.get("launched") or [], by_kernel)
            share_source = "resolution:launched"
        if device_us <= 0:
            continue
        tensors = _tensor_args(row.get("args") or [])
        shape, dtype = _shape_sig(tensors), _dtype_of(tensors)
        c = Candidate(
            candidate_id=candidate_id(op, shape, dtype),
            op=op,
            shape=shape,
            dtype=dtype,
            tensors=tensors,
            args=row.get("args") or [],
            calls=int(row.get("calls") or 0),
            device_us_op=device_us,
            share_pct=share.get("share_pct")
            if share_source == "op_share"
            else (100.0 * device_us / total_us if total_us else None),
        )
        c.share_source = share_source
        if res:
            c.provider = str(res.get("provider") or "unresolved")
            c.cls = CLASS_TOKENS.get(c.provider, "unresolved")
            c.where = [str(w) for w in res.get("where") or []]
            c.bundle = res.get("bundle")
            c.schema = res.get("schema")
            c.launched = [str(k) for k in res.get("launched") or []]
        total_calls = int(op_calls.get(op) or 0)
        if total_calls > 0:
            c.t_dev = c.device_us_op / total_calls
            c.t_dev_source = "profiler:op_average"
        key = op if op in bytes_overrides else c.candidate_id
        if key in bytes_overrides:
            c.bytes_min, c.bytes_min_note = int(bytes_overrides[key]), "operator override"
        else:
            c.bytes_min, c.bytes_min_note = _bytes_min(c.args, c.schema)
        if _gemm_like(res):
            c.flops = _gemm_flops(tensors)
        _apply_measurement(c, measurements.get(c.candidate_id))
        out.append(c)
    for t in report.get("triton") or []:
        name = str(t.get("kernel"))
        us = by_kernel.get(name)
        if not us or float(us) <= 0:
            continue
        c = Candidate(
            candidate_id=candidate_id(name, "-", "-"),
            op=name,
            shape="-",
            dtype="-",
            tensors=[],
            args=[],
            calls=int(t.get("calls") or 0),
            device_us_op=float(us),
            share_pct=None,
        )
        c.cls, c.provider = "triton", "Triton"
        c.where = [str(t.get("source"))] if t.get("source") else []
        c.launched = [name]
        if c.calls > 0:
            c.t_dev = c.device_us_op / c.calls
            c.t_dev_source = "profiler:kernel_average"
        c.bytes_min_note = "no argument record for a Triton kernel"
        _apply_measurement(c, measurements.get(c.candidate_id))
        out.append(c)
    return out


def _launched_us(launched: Sequence[str], by_kernel: Dict[str, float]) -> float:
    """Device time of the kernels an op launched, by the profiler's truncated-name prefix."""
    us = 0.0
    for k in launched:
        us += next((float(v) for name, v in by_kernel.items() if name.startswith(k[:60])), 0.0)
    return us


def _apply_measurement(c: Candidate, m: Optional[Dict[str, Any]]) -> None:
    if not m:
        return
    if m.get("t_dev_us") is not None:
        # The source names the instrument: "unitrace"/"profiler" are device-side durations;
        # "event" carries the timer's region overhead and is held to timing_floor_us.
        c.t_dev, c.t_dev_source = float(m["t_dev_us"]), str(m.get("t_dev_source") or "measurements")
    if m.get("t_host_us") is not None:
        c.t_host = float(m["t_host_us"])
    if m.get("spread_us") is not None:
        c.spread_us = float(m["spread_us"])
    if m.get("spill") is not None:
        c.spill = int(m["spill"])
    if m.get("geom") is not None:
        c.geom = str(m["geom"])
    if m.get("harness"):
        c.harness = str(m["harness"])


# --------------------------------------------------------------------------- regime (section 2.1)


def derive(c: Candidate, ctx: Context, bw_pattern: Callable[[int, int], Optional[float]]) -> None:
    """t_mem, t_mem_pattern, t_cmp and the bounds from the section 2.1 inputs; None where an
    input is unmeasured.

    Two floors enter a bound. The instrument floor is what the timer that produced ``t_dev``
    adds to it: ``timing_floor_us`` for an event-timed number, nothing for a device-side
    duration from the profiler or unitrace, which carries no region overhead. The launch
    floor applies to every ``t_dev`` regardless of instrument: no kernel runs shorter than one
    launch, so nothing inside a kernel can take it below ``launch_floor_us``.

    ``bound`` is the section 2.1 regime bound, ``max(t_mem, t_cmp, floors)``, at contiguous
    bandwidth; ``bound_pattern`` swaps in ``t_mem_pattern`` and is what a kernel obliged to
    keep its access pattern cannot go below (the ceiling of section 6.8).
    """
    cal = ctx.cal
    bw = getattr(cal, "bandwidth_gbs", None)
    if c.bytes_min is not None and bw:
        c.t_mem = c.bytes_min / (bw * 1e9) * 1e6
        c.bw_pattern = bw_pattern(*c.pattern) if c.pattern is not None else bw
        c.t_mem_pattern = None if c.bw_pattern is None else c.bytes_min / (c.bw_pattern * 1e9) * 1e6
    peak = (getattr(cal, "matmul_peak_tflops", None) or {}).get(c.dtype)
    if c.flops is not None and peak:
        c.t_cmp = c.flops / (peak * 1e12) * 1e6
    if c.t_dev_source.startswith("event"):
        c.floor_us, c.floor_source = getattr(cal, "timing_floor_us", None), "timing_floor_us"
    else:
        c.floor_us, c.floor_source = 0.0, "device-side duration: no region overhead"
    c.launch_us = getattr(cal, "launch_floor_us", None)
    floors = [v for v in (c.floor_us, c.launch_us) if v is not None]
    parts = [v for v in (c.t_mem, c.t_cmp, *floors) if v is not None]
    c.bound = max(parts) if parts else None
    parts = [v for v in (c.t_mem_pattern, c.t_cmp, *floors) if v is not None]
    c.bound_pattern = max(parts) if parts else None


def classify(c: Candidate, ctx: Context) -> None:
    """The section 2.1 row this candidate falls in, applied top to bottom as a validity order.

    Two rows the plan lists lower are evaluated first because their own text makes them
    preconditions: ``emulated`` (a latency on an emulated dtype is not the format's, so no
    later comparison is about the kernel) and ``spill-limited`` (spill traffic is not in
    bytes_min, so t_mem is wrong until it is gone). Rows whose inputs this run did not
    measure -- host/sync-bound, occupancy-limited -- are recorded as skipped, not guessed.
    """
    floor, launch, spread = c.floor_us, c.launch_us, c.spread_us
    c.regime_rows_skipped = ["host-bound:no profiler CPU time in this run"]
    if c.geom is None:
        c.regime_rows_skipped.append("occupancy-limited:no geom")
    if launch is None and c.t_host is None:
        c.regime_rows_skipped.append("launch-bound:no launch_floor_us and no t_host")
    if c.t_host is not None and c.t_dev is not None and c.t_host < c.t_dev:
        # Wall time per call cannot be below the device time per call of the same work.
        # The two instruments disagree; the ceiling still rests on t_dev, and the reader
        # is told. A unitrace device duration would arbitrate.
        c.notes.append(
            f"instruments disagree: t_host_us={_fmt(c.t_host)} < t_dev_us={_fmt(c.t_dev)} "
            f"({c.t_dev_source})"
        )
    if c.t_dev is None:
        c.regime, c.regime_test = R_UNMEASURED, "t_dev_us=None"
        return
    if floor is not None and c.t_dev < floor:
        c.regime, c.regime_test = (
            R_UNMEASURABLE,
            f"t_dev_us={_fmt(c.t_dev)} < floor_us={_fmt(floor)}",
        )
        return
    if c.native is False:
        c.regime, c.regime_test = R_EMULATED, "native=False"
        return
    if c.t_host is not None and (c.t_host - c.t_dev) > c.t_dev:
        c.regime = R_LAUNCH
        c.regime_test = f"t_host_us-t_dev_us={_fmt(c.t_host - c.t_dev)} > t_dev_us={_fmt(c.t_dev)}"
        return
    if launch is not None and c.t_dev <= launch:
        c.regime, c.regime_test = (
            R_LAUNCH,
            f"t_dev_us={_fmt(c.t_dev)} <= launch_floor_us={_fmt(launch)}",
        )
        return
    if c.spill is not None and c.spill > 0:
        c.regime, c.regime_test = R_SPILL, f"spill={c.spill} > 0"
        return
    if spread is None:
        c.regime, c.regime_test = R_UNMEASURED, "spread_us=None"
        return
    if c.bound is not None and abs(c.t_dev - c.bound) <= spread:
        c.regime = R_AT_BOUND
        c.regime_test = (
            f"|t_dev_us-bound_us|={_fmt(abs(c.t_dev - c.bound))} <= spread_us={_fmt(spread)}"
        )
        return
    t_mem, t_pat, t_cmp = c.t_mem, c.t_mem_pattern, c.t_cmp
    if t_mem is not None and t_pat is not None:
        if abs(c.t_dev - t_pat) <= spread and (t_pat - t_mem) > spread:
            c.regime = R_MEM_LAYOUT
            c.regime_test = (
                f"|t_dev_us-t_mem_pattern_us|={_fmt(abs(c.t_dev - t_pat))} <= spread_us={_fmt(spread)}"
                f" and t_mem_pattern_us-t_mem_us={_fmt(t_pat - t_mem)} > spread_us"
            )
            return
        if t_mem > (t_cmp or 0.0) and (c.t_dev - t_pat) > spread:
            c.regime = R_MEM_INEFF
            c.regime_test = (
                f"t_dev_us-t_mem_pattern_us={_fmt(c.t_dev - t_pat)} > spread_us={_fmt(spread)}"
            )
            return
    if t_cmp is not None and t_cmp >= (t_mem or 0.0) and (c.t_dev - t_cmp) > spread:
        c.regime = R_CMP_INEFF
        c.regime_test = f"t_dev_us-t_cmp_us={_fmt(c.t_dev - t_cmp)} > spread_us={_fmt(spread)}"
        return
    c.regime, c.regime_test = R_UNCLASSIFIED, "no row matched"


# --------------------------------------------------------------------------- gates (section 6.8)


def _gemm_producers(c: Candidate, ctx: Context) -> List[Dict[str, Any]]:
    return [
        e
        for e in ctx.edges
        if e.get("consumer") == c.op
        and _gemm_like(ctx.resolution.get(_op_key(str(e.get("producer")))))
    ]


def gate_class_admits(c: Candidate, mech: str, ctx: Context) -> Gate:
    where = _here()
    if mech == "provider_patch":
        return Gate(
            c.cls == "provider_kernel", "class", c.cls, "==", "admits", "provider_kernel", where
        )
    if mech == "triton_in_place":
        return Gate(c.cls == "triton", "class", c.cls, "==", "admits", "triton", where)
    if mech == "library_call":
        return Gate(c.cls == "onednn", "class", c.cls, "==", "admits", "onednn", where)
    if mech == "upstream_report":
        return Gate(c.cls == "aten", "class", c.cls, "==", "admits", "aten", where)
    if mech == "layout_transform":
        return Gate(
            c.regime == R_MEM_LAYOUT, "regime", c.regime, "==", "admits", R_MEM_LAYOUT, where
        )
    if mech in FUSION:
        n = len(_gemm_producers(c, ctx))
        return Gate(n > 0, "gemm_producers", n, ">", "required", 0, where)
    if mech == "apply_substitution":
        return Gate(
            c.cls in KERNEL_CLASSES, "class", c.cls, "in", "kernel_classes", KERNEL_CLASSES, where
        )
    if mech == "source_rewrite":
        gap = ctx.gaps.get(c.op) or ctx.gaps.get(_op_key(c.op)) or {}
        moved, required = gap.get("bytes_moved"), gap.get("bytes_required")
        ok = moved is not None and required is not None and moved > required
        return Gate(ok, "composite_bytes_moved", moved, ">", "bytes_required", required, where)
    raise ValueError(mech)


def gate_edge_present(c: Candidate, mech: str, ctx: Context) -> Gate:
    count = max((int(e.get("count") or 0) for e in _gemm_producers(c, ctx)), default=0)
    return Gate(
        count >= ctx.edge_threshold,
        "edge_count",
        count,
        ">=",
        "threshold",
        ctx.edge_threshold,
        _here(),
    )


def _post_op(c: Candidate, ctx: Context) -> Optional[str]:
    if ctx.post_op_algorithms is None:
        return None
    bare = c.op.split(".")[1] if "." in c.op else c.op
    bare = _POST_OP_SYNONYMS.get(bare, bare)
    for alg in sorted(ctx.post_op_algorithms):
        if alg == bare or alg.startswith(bare + "_"):
            return alg
    return None


def gate_epilogue_expressible(c: Candidate, mech: str, ctx: Context) -> Gate:
    where = _here()
    presets = (
        None
        if ctx.presets is None
        else [name for name, desc in ctx.presets if _preset_match(c.op, name, desc)]
    )
    post_op = _post_op(c, ctx)
    # With neither the preset list nor the post-op catalog readable nothing was evaluated:
    # the count is None, and the mechanism is unavailable rather than inexpressible.
    n: Optional[int] = None
    if presets is not None or ctx.post_op_algorithms is not None:
        n = len(presets or []) + (1 if post_op else 0)
    detail = {
        "presets": "unreadable" if presets is None else presets,
        "post_op": post_op,
        "post_op_algorithms": "unreadable"
        if ctx.post_op_algorithms is None
        else len(ctx.post_op_algorithms),
    }
    return Gate(bool(n), "expressions", n, ">", "required", 0, where, detail)


def gate_source_present(c: Candidate, mech: str, ctx: Context) -> Gate:
    where = _here()
    if mech == "provider_patch":
        n = 0
        if c.bundle and (pathlib.Path(c.bundle) / "PROVENANCE.md").is_file():
            n = sum(1 for p in (pathlib.Path(c.bundle) / "source").rglob("*") if p.is_file())
        return Gate(n > 0, "source_files", n, ">", "required", 0, where, {"bundle": c.bundle})
    files = [w.split(":")[0] for w in c.where if ":" in w and w.split(":")[-1].isdigit()]
    present = [f for f in files if pathlib.Path(f).is_file()]
    return Gate(
        len(present) > 0, "source_files", len(present), ">", "required", 0, where, {"files": files}
    )


def gate_definition_exists(c: Candidate, mech: str, ctx: Context) -> Gate:
    where = _here()
    found = ctx.definitions_matching(c)
    if found is None:
        # The dataset could not be read, so no definition was looked for. Rendering that as
        # zero would certify an absence nobody checked; None says the mechanism is
        # unavailable until the dataset is.
        return Gate(
            False,
            "definitions_matching",
            None,
            ">",
            "required",
            0,
            where,
            {"definitions": None, "dataset": "unreadable"},
        )
    names = list(found)
    return Gate(
        len(names) > 0,
        "definitions_matching",
        len(names),
        ">",
        "required",
        0,
        where,
        {"definitions": names},
    )


def gate_cost_calibrated(c: Candidate, mech: str, ctx: Context) -> Gate:
    cost = getattr(ctx.cal, "dispatch_us", None)
    return Gate(cost is not None, "dispatch_us", cost, "is_not", "unmeasured", None, _here())


def _kernel_matched(c: Candidate, ctx: Context) -> bool:
    if not c.launched:
        return False
    return any(name.startswith(k[:60]) for k in c.launched for name in ctx.by_kernel)


def gate_measurable(c: Candidate, mech: str, ctx: Context) -> Gate:
    where = _here()
    floor = c.floor_us
    detail = {"floor_source": c.floor_source}
    if c.t_dev is None or floor is None or not c.t_dev > floor:
        return Gate(False, "t_dev_us", c.t_dev, ">", "instrument_floor_us", floor, where, detail)
    matched = _kernel_matched(c, ctx)
    if not matched:
        return Gate(
            False,
            "kernel_matched",
            matched,
            "==",
            "required",
            True,
            where,
            {"launched": c.launched},
        )
    return Gate(True, "t_dev_us", c.t_dev, ">", "instrument_floor_us", floor, where, detail)


def mechanism_cost(mech: str, ctx: Context) -> Optional[float]:
    """What delivering through `mech` costs per call, from the calibration -- never a literal
    for a measured quantity. Patching source, transforming a layout at load time, changing a
    library call or fusing at the call site put nothing in the call path; only apply()'s
    dispatch does, and its cost is whatever this part measured (None if it did not)."""
    if mech in APPLY:
        return getattr(ctx.cal, "dispatch_us", None)
    return 0.0


def headroom(c: Candidate, mech: str, ctx: Context) -> Tuple[Optional[float], Dict[str, Any]]:
    """t_dev minus the bound the mechanism cannot go below, and how that bound was formed."""
    if c.t_dev is None:
        return None, {"reason": "t_dev_us=None"}
    floors = [v for v in (c.floor_us, c.launch_us) if v is not None]
    if mech in FUSION:
        # The consumer's whole kernel disappears into the epilogue, and so does its launch;
        # the launch is priced only when this part measured it.
        value = c.t_dev + (c.launch_us or 0.0)
        return value, {
            "bound_of": "t_dev_us+launch_floor_us",
            "launch_priced": c.launch_us is not None,
        }
    if mech == "layout_transform":
        if c.bound is None:
            return None, {"reason": "bound_us=None"}
        return c.t_dev - c.bound, {"bound_of": "max(t_mem_us;t_cmp_us;floors)"}
    if mech == "source_rewrite":
        gap = ctx.gaps.get(c.op) or ctx.gaps.get(_op_key(c.op)) or {}
        bw = getattr(ctx.cal, "bandwidth_gbs", None)
        required = gap.get("bytes_required")
        t_req = required / (bw * 1e9) * 1e6 if (required is not None and bw) else None
        parts = [v for v in (t_req, c.t_cmp, *floors) if v is not None]
        if not parts:
            return None, {"reason": "no bound"}
        return c.t_dev - max(parts), {"bound_of": "max(bytes_required/bw;t_cmp_us;floors)"}
    if c.bound_pattern is None:
        return None, {"reason": "bound_pattern_us=None"}
    return c.t_dev - c.bound_pattern, {"bound_of": "max(t_mem_pattern_us;t_cmp_us;floors)"}


def gate_headroom(c: Candidate, mech: str, ctx: Context) -> Gate:
    value, detail = headroom(c, mech, ctx)
    return Gate(
        value is not None and value > 0, "headroom_us", value, ">", None, 0, _here(), detail
    )


def gate_net_positive(c: Candidate, mech: str, ctx: Context) -> Gate:
    where = _here()
    h, _ = headroom(c, mech, ctx)
    cost = mechanism_cost(mech, ctx)
    ceiling = None if (h is None or cost is None) else h - cost
    ok = ceiling is not None and c.spread_us is not None and ceiling > c.spread_us
    return Gate(
        ok, "ceiling_us", ceiling, ">", "spread_us", c.spread_us, where, {"mechanism_us": cost}
    )


def worth_of(c: Candidate, ceiling: Optional[float], ctx: Context) -> Optional[float]:
    if ceiling is None or not ctx.total_us:
        return None
    return ceiling * c.calls / ctx.total_us


def gate_worth_cutoff(c: Candidate, mech: str, ctx: Context) -> Gate:
    h, _ = headroom(c, mech, ctx)
    cost = mechanism_cost(mech, ctx)
    ceiling = None if (h is None or cost is None) else h - cost
    worth = worth_of(c, ceiling, ctx)
    ok = worth is not None and worth >= ctx.cutoff
    return Gate(
        ok,
        "worth",
        worth,
        ">=",
        "cutoff",
        ctx.cutoff,
        _here(),
        {"ceiling_us": ceiling, "mechanism_us": cost},
    )


def gate_worklist_eligible(c: Candidate, mech: str, ctx: Context) -> Gate:
    return Gate(False, "worklist_row", False, "==", "required", True, _here())


def chain_for(mech: str) -> List[Tuple[str, Callable[[Candidate, str, Context], Gate]]]:
    if mech == "upstream_report":
        return [("class_admits", gate_class_admits), ("worklist_eligible", gate_worklist_eligible)]
    chain: List[Tuple[str, Callable[[Candidate, str, Context], Gate]]] = [
        ("class_admits", gate_class_admits)
    ]
    if mech in FUSION:
        chain += [
            ("edge_present", gate_edge_present),
            ("epilogue_expressible", gate_epilogue_expressible),
        ]
    if mech in PATCH:
        chain.append(("source_present", gate_source_present))
    if mech in APPLY:
        chain += [
            ("definition_exists", gate_definition_exists),
            ("cost_calibrated", gate_cost_calibrated),
        ]
    chain += [
        ("measurable", gate_measurable),
        ("headroom", gate_headroom),
        ("net_positive", gate_net_positive),
        ("worth_cutoff", gate_worth_cutoff),
    ]
    return chain


# --------------------------------------------------------------------------- evaluation


def _base(c: Candidate, ctx: Context, mech: str) -> Dict[str, Any]:
    return {
        "run_id": ctx.run_id,
        "candidate_id": c.candidate_id,
        "op": c.op,
        "shape": c.shape,
        "class": c.cls,
        "regime": c.regime,
        "mechanism": mech,
    }


def evaluate(c: Candidate, ctx: Context) -> List[Dict[str, Any]]:
    """Every mechanism through its gate chain; one record per gate, then ACCEPT or the
    candidate's UNROUTABLE."""
    rows: List[Dict[str, Any]] = []
    accepted = 0
    for mech in MECHANISMS:
        rejected = False
        for gate_name, fn in chain_for(mech):
            g = fn(c, mech, ctx)
            row = {
                **_base(c, ctx, mech),
                "gate": gate_name,
                "status": "PASS" if g.passed else "REJECT",
                "lhs": g.lhs,
                "lhs_value": g.lhs_value,
                "cmp": g.cmp,
                "rhs": g.rhs,
                "rhs_value": g.rhs_value,
                "arithmetic": g.arithmetic(),
                "where": g.where,
                "detail": g.detail,
            }
            if not g.passed:
                row["needs"] = g.needs()
            rows.append(row)
            if not g.passed:
                rejected = True
                break
        if not rejected:
            h, _ = headroom(c, mech, ctx)
            cost = mechanism_cost(mech, ctx)
            ceiling = h - cost  # both non-None: net_positive passed
            rows.append(
                {
                    **_base(c, ctx, mech),
                    "gate": None,
                    "status": "ACCEPT",
                    "ceiling_us": ceiling,
                    "worth": worth_of(c, ceiling, ctx),
                    "mechanism_us": cost,
                    "calls": c.calls,
                }
            )
            accepted += 1
    if not accepted:
        rows.append(
            {
                **_base(c, ctx, "-"),
                "gate": None,
                "status": "UNROUTABLE",
                "mechanisms_evaluated": len(MECHANISMS),
                "mechanisms_rejected": len(MECHANISMS),
            }
        )
    return rows


def worklist(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Exactly the ACCEPT rows, by worth descending, then candidate_id, then mechanism."""
    accepted = [r for r in rows if r["status"] == "ACCEPT"]
    return sorted(
        accepted,
        key=lambda r: (
            -(r["worth"] if r["worth"] is not None else -math.inf),
            r["candidate_id"],
            r["mechanism"],
        ),
    )


# --------------------------------------------------------------------------- log


def format_line(row: Dict[str, Any]) -> str:
    head = [
        LOG_TAG,
        LOG_VERSION,
        row["run_id"],
        row["candidate_id"],
        row["op"],
        row["shape"],
        row["class"],
        row["regime"],
        row["mechanism"],
    ]
    if row["status"] == "ACCEPT":
        tail = [
            "ACCEPT",
            f"ceiling_us={_fmt(row['ceiling_us'])}",
            f"worth={_fmt(row['worth'])}",
            f"mechanism_us={_fmt(row['mechanism_us'])}",
        ]
    elif row["status"] == "UNROUTABLE":
        tail = [
            "UNROUTABLE",
            f"{row['mechanisms_evaluated']} mechanisms evaluated",
            f"{row['mechanisms_rejected']} rejected",
            "-",
        ]
    else:
        tail = [row["gate"], row["status"], row["arithmetic"], row["where"]]
    fields = [str(f) for f in head + tail]
    for f in fields:
        if "," in f or "\n" in f:
            raise ValueError(f"log field would break the line grammar: {f!r}")
    if len(fields) != LOG_FIELDS:
        raise ValueError(f"{len(fields)} fields, expected {LOG_FIELDS}")
    return ",".join(fields)


def _parse_value(text: str) -> Any:
    if text == "None":
        return None
    if text in ("True", "False"):
        return text == "True"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _kv(token: str) -> Tuple[Optional[str], Any]:
    if "=" in token:
        name, value = token.split("=", 1)
        return name, _parse_value(value)
    return None, _parse_value(token)


def parse_log(text: str) -> List[Dict[str, Any]]:
    """The reference reader: bound.log back into records, with no guessing."""
    records: List[Dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        f = line.split(",")
        if len(f) != LOG_FIELDS or f[0] != LOG_TAG or f[1] != LOG_VERSION:
            raise ValueError(
                f"line {lineno}: not a {LOG_TAG} {LOG_VERSION} line with {LOG_FIELDS} fields"
            )
        rec: Dict[str, Any] = {
            "run_id": f[2],
            "candidate_id": f[3],
            "op": f[4],
            "shape": f[5],
            "class": f[6],
            "regime": f[7],
            "mechanism": f[8],
        }
        if f[9] == "ACCEPT":
            rec["status"] = "ACCEPT"
            for token in f[10:13]:
                name, value = _kv(token)
                rec[name] = value
        elif f[9] == "UNROUTABLE":
            rec["status"] = "UNROUTABLE"
            rec["mechanisms_evaluated"] = int(f[10].split(" ")[0])
            rec["mechanisms_rejected"] = int(f[11].split(" ")[0])
        else:
            tokens = f[11].split(" ")
            if len(tokens) != 3 or f[10] not in ("PASS", "REJECT"):
                raise ValueError(f"line {lineno}: gate line grammar")
            rec["gate"], rec["status"] = f[9], f[10]
            rec["lhs"], rec["lhs_value"] = _kv(tokens[0])
            rec["cmp"] = tokens[1]
            rec["rhs"], rec["rhs_value"] = _kv(tokens[2])
            rec["arithmetic"], rec["where"] = f[11], f[12]
        records.append(rec)
    return records


def worklist_from_log(text: str) -> List[Dict[str, Any]]:
    return worklist(parse_log(text))


# --------------------------------------------------------------------------- consumers
#
# The stages after this one -- the trial loop and the serving A/B -- exercise one
# (candidate, mechanism) pair at a time. A pair this stage rejected has been priced out: a
# measurement of it says nothing about the pipeline, yet its number reads exactly like one
# that does. So a consuming stage asks here before it measures, and refuses -- in the key
# contract, with the gate and the arithmetic that decided -- when the pair is not an ACCEPT
# row, or when the routing it is reading was computed from a discovery that has since
# changed. bound.json is the artifact consulted (it is authoritative and carries the
# provenance); bound.log is its mirror for grep.

ROUTING_OK = "OK"
ROUTING_REJECTED = "REJECTED"
ROUTING_UNEVALUATED = "UNEVALUATED"
ROUTING_STALE = "STALE"
ROUTING_UNCHECKED = "UNCHECKED"

VERDICT_ROUTING_REJECTED = "ROUTING_REJECTED"
VERDICT_STALE_INPUT = "STALE_INPUT"

_HARNESS_TENSORS = re.compile(r"torch\.\w+\(\[([^\]]*)\], dtype=torch\.(\w+)")


class RoutingRefused(Exception):
    """What a consuming stage reports, in the key contract, instead of measuring.

    ``fields`` are the contract keys in print order (``VERDICT`` last) and ``why`` is the
    prose beside them: what was checked, what it found, and what the operator would have
    to change. Neither contains a measurement.
    """

    def __init__(self, fields: Dict[str, Any], why: str):
        super().__init__(why)
        self.fields, self.why = fields, why


def report_digest(path: pathlib.Path) -> str:
    """The identity of a discovery report: a digest of its bytes, also the default run_id.

    Every consumer of the routing recomputes this and compares it with what bound.json
    recorded, so re-running discovery voids the routing computed from the previous run.
    """
    return hashlib.sha1(pathlib.Path(path).read_bytes()).hexdigest()[:12]


def _stale(why: str, **fields: Any) -> RoutingRefused:
    return RoutingRefused({"routing": ROUTING_STALE, **fields, "verdict": VERDICT_STALE_INPUT}, why)


def load_bound(path: Any) -> Dict[str, Any]:
    """bound.json from a path to it or to the directory holding it; refuses anything else."""
    p = pathlib.Path(path)
    if p.is_dir():
        p = p / "bound.json"
    if not p.is_file():
        raise _stale(
            f"no routing at {p}. Run scripts/bound_candidates.py --report <discovered.json> "
            f"--resolution <resolution.json> --out-dir {p.parent} first; a stage does not "
            "measure a pair nothing has priced.",
            bound=str(p),
        )
    data = json.loads(p.read_text())
    if data.get("format") != f"{LOG_TAG}/{LOG_VERSION}" or "rows" not in data:
        raise _stale(
            f"{p} is not a {LOG_TAG}/{LOG_VERSION} bound.json; regenerate it with "
            "scripts/bound_candidates.py.",
            bound=str(p),
        )
    data["_path"] = str(p)
    return data


def check_bound_provenance(bound: Dict[str, Any], report: Any = None) -> pathlib.Path:
    """Refuse routing whose discovery report has moved, changed or gone since it was made.

    The routing's worth and ceilings were computed from one discovery report; if that file
    now has different bytes, discovery was re-run (another model, another stack revision,
    another setting) and every row is about a run that no longer exists.
    """
    path = pathlib.Path(report or bound.get("report") or "")
    where = bound.get("_path", "bound.json")
    expected = bound.get("report_sha1")
    if not expected:
        raise _stale(
            f"{where} records no report_sha1, so it cannot be tied to the discovery it "
            "was computed from; regenerate it with scripts/bound_candidates.py.",
            bound=where,
            run_id=bound.get("run_id"),
        )
    if not str(path) or not path.is_file():
        raise _stale(
            f"the routing in {where} was computed from {path or '(unrecorded)'}, which is "
            "not there. If the report moved, name it with --report; otherwise re-run "
            "discovery and scripts/bound_candidates.py.",
            bound=where,
            report=str(path) or None,
            run_id=bound.get("run_id"),
        )
    actual = report_digest(path)
    if actual != expected:
        raise _stale(
            f"{path} has changed since the routing in {where} was computed from it "
            f"(sha1 {actual} now, {expected} then): discovery was re-run, so every row is "
            "about a run that no longer exists. Re-run scripts/bound_candidates.py on the "
            "current report before measuring anything against it.",
            bound=where,
            report=str(path),
            report_sha1=actual,
            expected_sha1=expected,
            run_id=bound.get("run_id"),
        )
    return path


def harness_identity(path: Any) -> Tuple[str, str, str, Optional[int]]:
    """(op, shape signature, dtype, calls) as a generated harness records them.

    These are the fields a candidate is identified by, read the way :func:`harness_for`
    reads them, so the two directions of the harness <-> candidate match agree.
    """
    text = pathlib.Path(path).read_text(errors="ignore")
    m = re.search(r'^OP = "(.+)"$', text, re.M)
    if not m:
        raise RoutingRefused(
            {
                "routing": ROUTING_UNEVALUATED,
                "harness": str(path),
                "verdict": VERDICT_ROUTING_REJECTED,
            },
            f"{path} records no OP, so it cannot be tied to a candidate; only a harness "
            "emitted by scripts/harness_from_model.py (or one that keeps its OP and "
            "get_inputs shapes) can be routed.",
        )
    tensors = [
        (tuple(int(x) for x in dims.split(",") if x.strip()), dtype)
        for dims, dtype in _HARNESS_TENSORS.findall(text)
    ]
    calls = re.search(r"^CALLS = (\d+)$", text, re.M)
    return (
        m.group(1),
        _shape_sig(tensors),
        _dtype_of(tensors),
        int(calls.group(1)) if calls else None,
    )


def candidate_for_harness(bound: Dict[str, Any], harness: Any) -> Dict[str, Any]:
    """The candidate record this harness is the harness of, or a refusal.

    A harness is tied to a candidate by (op, shape), the inverse of :func:`harness_for`.
    No match means the harness came from another discovery run, or its op carried no
    device time in this one -- and an op with no share was never a worklist row.
    """
    op, shape, _, calls = harness_identity(harness)
    matches = [c for c in bound.get("candidates") or [] if c["op"] == op and c["shape"] == shape]
    if not matches:
        raise RoutingRefused(
            {
                "routing": ROUTING_UNEVALUATED,
                "harness": str(harness),
                "op": op,
                "shape": shape,
                "run_id": bound.get("run_id"),
                "verdict": VERDICT_ROUTING_REJECTED,
            },
            f"run {bound.get('run_id')} has no candidate for {op} at {shape}: the harness "
            "is from a different discovery run, or the op carried no device time in this "
            "one. Neither is a worklist row; re-run discovery and bounding together.",
        )
    c = matches[0]
    if calls is not None and c.get("calls") is not None and int(c["calls"]) != calls:
        raise _stale(
            f"{harness} records {calls} calls of {op} and the routing priced {c['calls']}: "
            "the harness and the routing come from different discovery runs, and worth was "
            "computed from the other one. Regenerate both from one run.",
            harness=str(harness),
            candidate=c["candidate_id"],
            op=op,
            shape=shape,
            run_id=bound.get("run_id"),
        )
    return c


def routing_of(bound: Dict[str, Any], candidate_id: str, mechanism: str) -> Dict[str, Any]:
    """The row that ended the pair's chain: ACCEPT, the REJECT that stopped it, or {}."""
    rows = [
        r
        for r in bound.get("rows") or []
        if r["candidate_id"] == candidate_id and r["mechanism"] == mechanism
    ]
    return rows[-1] if rows else {}


def _needs_text(needs: Dict[str, Any]) -> str:
    threshold = needs.get("threshold_name")
    rhs = f"{threshold}={needs.get('threshold')}" if threshold else str(needs.get("threshold"))
    return (
        f"{needs.get('quantity')} must {needs.get('direction')} from "
        f"{needs.get('observed')} to {needs.get('cmp')} {rhs}"
    )


def require_routed(bound: Dict[str, Any], candidate_id: str, mechanism: str) -> Dict[str, Any]:
    """The contract fields of the ACCEPT row for the pair, or a refusal naming the gate."""
    if mechanism not in MECHANISMS:
        raise RoutingRefused(
            {
                "routing": ROUTING_UNEVALUATED,
                "candidate": candidate_id,
                "mechanism": mechanism,
                "verdict": VERDICT_ROUTING_REJECTED,
            },
            f"{mechanism!r} is not a mechanism the routing prices; one of: "
            + ", ".join(MECHANISMS),
        )
    row = routing_of(bound, candidate_id, mechanism)
    base = {
        "candidate": candidate_id,
        "op": row.get("op"),
        "shape": row.get("shape"),
        "mechanism": mechanism,
    }
    if not row:
        raise RoutingRefused(
            {
                "routing": ROUTING_UNEVALUATED,
                **base,
                "run_id": bound.get("run_id"),
                "verdict": VERDICT_ROUTING_REJECTED,
            },
            f"run {bound.get('run_id')} evaluated no {mechanism} for candidate "
            f"{candidate_id}: it is not a candidate of this run. A pair the routing never "
            "priced is not a worklist row.",
        )
    if row["status"] == "ACCEPT":
        return {
            "routing": ROUTING_OK,
            **base,
            "ceiling_us": row.get("ceiling_us"),
            "worth": row.get("worth"),
            "run_id": bound.get("run_id"),
        }
    needs = row.get("needs") or {}
    raise RoutingRefused(
        {
            "routing": ROUTING_REJECTED,
            **base,
            "gate": row.get("gate"),
            "arithmetic": row.get("arithmetic"),
            "needs": _needs_text(needs) if needs else None,
            "where": row.get("where"),
            "run_id": bound.get("run_id"),
            "verdict": VERDICT_ROUTING_REJECTED,
        },
        f"{mechanism} for {row.get('op')} at {row.get('shape')} was rejected at gate "
        f"{row.get('gate')}: {row.get('arithmetic')} ({row.get('where')}). Measuring it now "
        "measures what the routing already priced out, and the number would be read as "
        "evidence about the pipeline. To exercise this pair, change what the gate reads -- "
        + (_needs_text(needs) if needs else "see the gate")
        + " -- and re-run scripts/bound_candidates.py; do not measure past the gate.",
    )


def routed_from_harness(
    bound_path: Any, mechanism: str, harness: Any, report: Any = None
) -> Dict[str, Any]:
    """Everything a trial stage checks before it times a harness, in one call.

    Loads the routing, verifies it is still about the discovery it names, ties the harness
    to its candidate, and requires the (candidate, mechanism) pair to be an ACCEPT row.
    Raises :class:`RoutingRefused` at the first failure.
    """
    bound = load_bound(bound_path)
    check_bound_provenance(bound, report)
    candidate = candidate_for_harness(bound, harness)
    return require_routed(bound, candidate["candidate_id"], mechanism)


# --------------------------------------------------------------------------- inputs


def _load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(
        name, pathlib.Path(__file__).with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_fusion = None


def _preset_match(consumer: str, name: str, desc: str) -> bool:
    global _fusion
    if _fusion is None:
        _fusion = _load_sibling("fusion_candidates")
    return _fusion.match(consumer, name, desc)


def read_presets(xe_fuse: pathlib.Path) -> Optional[List[Tuple[str, str]]]:
    """Xe-Fuse's presets, or None when the generator is not there to list them."""
    global _fusion
    if _fusion is None:
        _fusion = _load_sibling("fusion_candidates")
    return _fusion.presets(xe_fuse)


def read_post_op_algorithms(header: pathlib.Path) -> Optional[Set[str]]:
    """oneDNN's eltwise post-op algorithms, read from its own header, or None if unreadable."""
    if not header.is_file():
        return None
    names = set(re.findall(r"dnnl_eltwise_([a-z0-9_]+)", header.read_text(errors="ignore")))
    return {n.replace("_use_dst_for_bwd", "") for n in names} or None


def _required_file(path: pathlib.Path, what: str, producer: str) -> pathlib.Path:
    """A named input that is not there is a failed precondition, never an empty input.

    Silently reading a missing file as ``{}`` lets the run continue with every op
    unresolved or every gap unknown, and the log it writes then reads as a finding --
    "no mechanism admits this op" -- about an input nobody supplied.
    """
    if not path.is_file():
        raise SystemExit(f"{what} not found at {path}; it is written by {producer}.")
    return path


def load_resolution(path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    _required_file(path, "resolution", "scripts/pull_kernel_source.py --json")
    data = json.loads(path.read_text())
    rows = data.get("ops") if isinstance(data, dict) else data
    resolution = {str(r["op"]): r for r in rows or []}
    if not resolution:
        raise SystemExit(
            f"{path} resolves no ops, so no class admits any mechanism and every row would "
            "be a class_admits REJECT of class=unresolved -- a statement about the missing "
            "input, not about the model. Re-run scripts/pull_kernel_source.py --json."
        )
    return resolution


def load_gaps(path: Optional[pathlib.Path]) -> Dict[str, Dict[str, float]]:
    if path is None:
        return {}
    _required_file(path, "gap analysis", "scripts/find_kernel_gaps.py")
    data = json.loads(path.read_text())
    return {str(k): v for k, v in (data.items() if isinstance(data, dict) else [])}


def check_report(report: Dict[str, Any], path: Any) -> None:
    """Refuse a discovery report that cannot carry share: no rows can be priced from it.

    Discovery writes ``device_time_error`` and a zero total when its device-time pass
    failed; bounding such a report yields an empty worklist that reads as "nothing worth
    doing" when the truth is "nothing was measured".
    """
    error = report.get("device_time_error")
    if error:
        raise SystemExit(
            f"{path}: discovery recorded no device time ({error}); no share, no bound. "
            "Re-run scripts/harness_from_model.py until its device-time pass succeeds."
        )
    if float(report.get("device_time_total_us") or 0.0) <= 0:
        raise SystemExit(
            f"{path}: device_time_total_us is not positive, so no op carries share and no "
            "row could be priced. Re-run scripts/harness_from_model.py."
        )
    if not report.get("ops"):
        raise SystemExit(f"{path}: no ops recorded; re-run scripts/harness_from_model.py.")


def default_cutoff(report: Dict[str, Any]) -> float:
    """The noise floor of Stage 1's share estimate: the smallest device time the profiler
    resolved for any kernel in this run, as a fraction of the total."""
    total = float(report.get("device_time_total_us") or 0.0)
    times = [float(v) for v in (report.get("device_time_by_kernel") or {}).values() if float(v) > 0]
    if not total or not times:
        return 0.0
    return min(times) / total


def calibration_from_json(path: pathlib.Path):
    from types import SimpleNamespace

    return SimpleNamespace(**json.loads(path.read_text()))


def definition_matcher(dataset: str) -> Callable[[Candidate], List[str]]:
    """Definitions whose tensor interface the candidate's arguments bind to, by op vocabulary,
    dtype, rank, constant axes and consistent variable axes."""
    from flashinfer_bench.data import TraceSet

    definitions = list(TraceSet.from_path(dataset).definitions.values())

    def matching(c: Candidate) -> List[str]:
        bare = (c.op.split(".")[1] if "." in c.op else c.op).lower()
        return [d.name for d in _same_operation(bare, definitions) if _binds(d, c.tensors)]

    return matching


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _same_operation(bare: str, definitions: Sequence[Any]) -> List[Any]:
    """Definitions that name the same operation as the op.

    A definition's name carries the operation it computes (``silu_and_mul_d3072``,
    ``fused_add_rmsnorm_h1024``), so when the op's own name appears in any definition name,
    only those definitions are the same operation -- ``gelu_and_mul`` and ``fused_add_rmsnorm``
    bind the same shapes as ``silu_and_mul`` and ``rms_norm`` and compute something else. When
    no definition name carries the op's name (``linear`` against ``gemm_*``), the op_type
    vocabulary decides, which is as far as names can take it; the shapes do the rest.
    """
    key = _norm(bare)
    by_name = [d for d in definitions if key and key in _norm(d.name)]
    if by_name:
        return by_name
    out = []
    for d in definitions:
        words = _OP_TYPE_WORDS.get(d.op_type)
        if words is not None and any(w in bare for w in words):
            out.append(d)
    return out


def _binds(definition, tensors: Sequence[Tuple[Tuple[int, ...], str]]) -> bool:
    specs = [s for s in definition.inputs.values() if s.shape is not None]
    if not specs or len(specs) > len(tensors):
        return False
    axes = definition.axes
    for combo in itertools.permutations(tensors, len(specs)):
        bound: Dict[str, int] = {}
        ok = True
        for spec, (shape, dtype) in zip(specs, combo):
            if str(getattr(spec.dtype, "value", spec.dtype)) != dtype or len(spec.shape) != len(
                shape
            ):
                ok = False
                break
            for axis, dim in zip(spec.shape, shape):
                ax = axes.get(axis)
                if ax is None:
                    ok = False
                elif getattr(ax, "type", "") == "const":
                    ok = ax.value == dim
                elif bound.setdefault(axis, dim) != dim:
                    ok = False
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return True
    return False


# --------------------------------------------------------------------------- measure


_HARNESS_SHAPES = re.compile(r"torch\.\w+\(\[([^\]]*)\]")


def harness_for(c: Candidate, harness_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """The harness discovery emitted for exactly this (op, shape), matched on its recorded OP
    and the shapes get_inputs builds."""
    want = [shape for shape, _ in c.tensors]
    for f in sorted(harness_dir.glob("*.py")):
        text = f.read_text(errors="ignore")
        m = re.search(r'^OP = "(.+)"$', text, re.M)
        if not m or m.group(1) != c.op:
            continue
        shapes = [
            tuple(int(x) for x in s.split(",") if x.strip()) for s in _HARNESS_SHAPES.findall(text)
        ]
        if shapes == want:
            return f
    return None


_KEY = re.compile(r"^([A-Z_]+): (.*)$", re.M)

PROFILE_WARMUP_S = 0.5
"""Sustained load before the profiled region of a harness. The part gates its clock when
the queue drains and needs load, not a call count, to reach its working clock; a device
duration profiled before that is the idle clock's, not the kernel's."""

# Run inside the harness's own interpreter: device-side duration per call of the op the
# harness calls, through the same profiler instrument Stage 1 charged shares with, so the
# per-shape t_dev and the op-level share are the same kind of number.
_PROFILE_SNIPPET = r"""
import importlib.util, json, sys, time
import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

path, calls, warm_s = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
spec = importlib.util.spec_from_file_location("harness_under_profile", path)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
model, inputs = m.Model(*m.get_init_inputs()), m.get_inputs()
dev = next((t.device for t in inputs if isinstance(t, torch.Tensor)), None)
backend = None
if dev is not None and dev.type == "xpu":
    backend = torch.xpu
elif dev is not None and dev.type == "cuda":
    backend = torch.cuda
act = [ProfilerActivity.CPU]
if backend is torch.xpu:
    act.append(ProfilerActivity.XPU)
elif backend is torch.cuda:
    act.append(ProfilerActivity.CUDA)


def sync():
    if backend is not None:
        backend.synchronize()


with torch.no_grad():
    model(*inputs)
    sync()
    deadline = time.perf_counter() + warm_s
    while time.perf_counter() < deadline:
        for _ in range(20):
            model(*inputs)
        sync()
    with profile(activities=act) as prof:
        for _ in range(calls):
            model(*inputs)
        sync()
kernels = {}
for e in prof.key_averages():
    us = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
    if us > 0 and e.device_type != DeviceType.CPU:
        kernels[e.key] = kernels.get(e.key, 0.0) + us
print("FIB_PROFILE " + json.dumps({"calls": calls, "kernels": kernels}))
"""


def profile_harness(
    harness: pathlib.Path, interp: str, calls: int, warm_s: float = PROFILE_WARMUP_S
) -> Dict[str, Any]:
    """Device time per call of the harness's op, and the kernels it launched, or an error."""
    try:
        r = subprocess.run(
            [interp, "-c", _PROFILE_SNIPPET, str(harness.resolve()), str(calls), str(warm_s)],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"profile_error": str(exc)[-2000:]}
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("FIB_PROFILE ")), None)
    if line is None:
        return {"profile_error": (r.stdout + r.stderr)[-2000:]}
    data = json.loads(line[len("FIB_PROFILE ") :])
    total = sum(float(v) for v in data["kernels"].values())
    if not data["kernels"] or total <= 0:
        return {"profile_error": "no device kernels recorded", "kernels": data["kernels"]}
    return {
        "t_dev_us": total / data["calls"],
        "t_dev_source": "profiler:harness",
        "kernels": data["kernels"],
        "profiled_calls": data["calls"],
    }


def parse_benchmark_keys(stdout: str) -> Dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in _KEY.finditer(stdout)}


def measure_harnesses(
    cands: Sequence[Candidate],
    harness_dir: pathlib.Path,
    out: pathlib.Path,
    python: Optional[str],
    rounds: int,
    calls: int,
    remeasure: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Each candidate's harness against itself through ``kernel_trials.py benchmark`` -- the
    one sanctioned benchmark -- for the harness's own wall time and paired spread. Results
    are kept in `out` so a re-run does not touch the device again."""
    kt = pathlib.Path(__file__).with_name("kernel_trials.py")
    existing: Dict[str, Dict[str, Any]] = json.loads(out.read_text()) if out.is_file() else {}
    work = out.parent / "measure-work"
    work.mkdir(parents=True, exist_ok=True)
    for c in cands:
        if c.candidate_id in existing and not remeasure:
            continue
        h = harness_for(c, harness_dir)
        if h is None:
            continue
        interp = python
        if interp is None:
            m = re.search(r'^RECORDED_WITH = "(.+)"$', h.read_text(errors="ignore"), re.M)
            interp = m.group(1) if m and pathlib.Path(m.group(1)).is_file() else sys.executable
        series = f"bound-{c.candidate_id}"
        record: Dict[str, Any] = {
            "harness": str(h),
            "interpreter": interp,
            "source": "kernel_trials benchmark A/A",
        }
        try:
            subprocess.run(
                [interp, str(kt), "init", series, str(h.resolve())],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=600,
                check=True,
            )
            r = subprocess.run(
                [
                    interp,
                    str(kt),
                    "benchmark",
                    series,
                    str(h.resolve()),
                    "--rounds",
                    str(rounds),
                    "--calls",
                    str(calls),
                ],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=900,
            )
            keys = parse_benchmark_keys(r.stdout)
            record["keys"] = keys
            if "CANDIDATE_US" in keys:
                t_host = float(keys["CANDIDATE_US"])
                pct = keys.get("SPREAD_PCT") or keys.get("CANDIDATE_SPREAD_PCT")
                record["t_host_us"] = t_host
                record["spread_us"] = None if pct is None else float(pct) / 100.0 * t_host
            else:
                record["error"] = (r.stdout + r.stderr)[-2000:]
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            record["error"] = str(exc)[-2000:]
        record.update(profile_harness(h, interp, calls * rounds))
        existing[c.candidate_id] = record
        out.write_text(json.dumps(existing, indent=2) + "\n")
    return existing


# --------------------------------------------------------------------------- run


def run(
    report: Dict[str, Any],
    resolution: Dict[str, Dict[str, Any]],
    measurements: Dict[str, Dict[str, Any]],
    cal: Any,
    run_id: str,
    presets: Optional[Sequence[Tuple[str, str]]] = None,
    post_op_algorithms: Optional[Set[str]] = None,
    definitions_matching: Optional[Callable[[Candidate], Optional[List[str]]]] = None,
    gaps: Optional[Dict[str, Dict[str, float]]] = None,
    cutoff: Optional[float] = None,
    patterns: Optional[Dict[str, Tuple[int, int]]] = None,
    bw_pattern: Optional[Callable[[int, int], Optional[float]]] = None,
    bytes_overrides: Optional[Dict[str, int]] = None,
    native: Optional[Callable[[str], Optional[bool]]] = None,
) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
    edges = list(report.get("edges") or [])
    ctx = Context(
        cal=cal,
        total_us=float(report.get("device_time_total_us") or 0.0),
        by_kernel=dict(report.get("device_time_by_kernel") or {}),
        edges=edges,
        resolution=resolution,
        presets=None if presets is None else list(presets),
        post_op_algorithms=post_op_algorithms,
        definitions_matching=definitions_matching or (lambda c: []),
        gaps=gaps or {},
        cutoff=default_cutoff(report) if cutoff is None else cutoff,
        run_id=run_id,
        edge_threshold=min((int(e.get("count") or 0) for e in edges), default=1) or 1,
    )
    cands = candidates_from_report(report, resolution, measurements, bytes_overrides)
    rows: List[Dict[str, Any]] = []
    for c in cands:
        if patterns and c.op in patterns:
            c.pattern = patterns[c.op]
        if native is not None and c.dtype != "-":
            c.native = native(c.dtype)
        derive(c, ctx, bw_pattern or (lambda run, stride: None))
        classify(c, ctx)
        rows.extend(evaluate(c, ctx))
    return cands, rows


def write_outputs(
    out_dir: pathlib.Path,
    run_id: str,
    cands: Sequence[Candidate],
    rows: Sequence[Dict[str, Any]],
    cal: Any,
    cutoff: float,
    report_path: str,
    report_sha1: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """bound.log, bound.json and worklist.json.

    ``report_sha1`` is the digest of the discovery report the rows were computed from and
    ``model`` the model that report was recorded on; the stages that consume this routing
    check both (:func:`check_bound_provenance`), so a re-run of discovery, or a run on
    another model, voids the routing instead of being measured against it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [format_line(r) for r in rows]
    (out_dir / "bound.log").write_text("\n".join(lines) + ("\n" if lines else ""))
    cal_dict = {
        k: getattr(cal, k, None)
        for k in (
            "hardware_id",
            "timer",
            "dispatch_us",
            "timing_floor_us",
            "bandwidth_gbs",
            "launch_floor_us",
            "matmul_peak_tflops",
        )
    }
    (out_dir / "bound.json").write_text(
        json.dumps(
            {
                "format": f"{LOG_TAG}/{LOG_VERSION}",
                "authoritative": "bound.json",
                "run_id": run_id,
                "report": report_path,
                "report_sha1": report_sha1,
                "model": model,
                "calibration": cal_dict,
                "cutoff": cutoff,
                "candidates": [c.metrics() for c in cands],
                "rows": list(rows),
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    (out_dir / "worklist.json").write_text(json.dumps(worklist(rows), indent=2, default=str) + "\n")


def print_summary(
    cands: Sequence[Candidate], rows: Sequence[Dict[str, Any]], out_dir: pathlib.Path
) -> None:
    wl = worklist(rows)
    print(
        f"\n{len(cands)} candidates, {len(rows)} gate rows -> {out_dir / 'bound.log'} (bound.json authoritative)\n"
    )
    print(f"{'worth':>10}  {'ceiling_us':>10}  {'mech_us':>8}  {'mechanism':18} {'op':40} shape")
    print("-" * 110)
    for r in wl:
        print(
            f"{_fmt(r['worth']):>10}  {_fmt(r['ceiling_us']):>10}  {_fmt(r['mechanism_us']):>8}  {r['mechanism']:18} {r['op'][:40]:40} {r['shape'][:30]}"
        )
    if not wl:
        print("(empty: no (candidate, mechanism) pair passed every gate; grep REJECT in bound.log)")
    unroutable = [r for r in rows if r["status"] == "UNROUTABLE"]
    print(
        f"\n{len(wl)} worklist row(s); {len(unroutable)} candidate(s) unroutable through every mechanism."
    )


def _parse_kv_option(items: Sequence[str], parse: Callable[[str], Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        key, _, value = item.partition("=")
        if not key or not value:
            raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {item!r}")
        out[key] = parse(value)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--report", required=True, help="discovered.json from scripts/harness_from_model.py"
    )
    ap.add_argument(
        "--resolution",
        required=True,
        help="resolution JSON from scripts/pull_kernel_source.py --json",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--measurements",
        help="per-candidate measurements JSON (default <out-dir>/measurements.json)",
    )
    ap.add_argument(
        "--measure",
        action="store_true",
        help="benchmark each harness against itself first (device)",
    )
    ap.add_argument(
        "--remeasure", action="store_true", help="with --measure: redo candidates already measured"
    )
    ap.add_argument("--harness-dir", help="directory of harnesses written by harness_from_model.py")
    ap.add_argument(
        "--python", help="interpreter for --measure (default: the harness's RECORDED_WITH)"
    )
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--calibration-json",
        help="read the calibration record from this file instead of calibration.get()",
    )
    ap.add_argument("--dataset", default="tmp/flashinfer-trace")
    ap.add_argument("--xe-fuse", default="tmp/Xe-Fuse")
    ap.add_argument("--onednn-include", default="tmp/oneDNN/include/oneapi/dnnl/dnnl_types.h")
    ap.add_argument("--gaps", help="find_kernel_gaps JSON: {op: {bytes_moved, bytes_required}}")
    ap.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="worth cutoff (fraction of device time); default: the run's share noise floor",
    )
    ap.add_argument(
        "--pattern",
        action="append",
        default=[],
        metavar="OP=RUN_BYTES:STRIDE_BYTES",
        help="the access pattern a kernel is obliged to use; measured through calibration",
    )
    ap.add_argument(
        "--bytes-min",
        action="append",
        default=[],
        metavar="OP_OR_ID=BYTES",
        help="override the bytes an op is obliged to move",
    )
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    report_path = _required_file(
        pathlib.Path(args.report), "discovery report", "scripts/harness_from_model.py"
    )
    report = json.loads(report_path.read_text())
    check_report(report, report_path)
    report_sha1 = report_digest(report_path)
    run_id = args.run_id or report_sha1
    resolution = load_resolution(pathlib.Path(args.resolution))
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.measurements:
        meas_path = _required_file(
            pathlib.Path(args.measurements), "measurements", "this script's --measure step"
        )
    else:
        meas_path = out_dir / "measurements.json"

    if args.calibration_json:
        cal = calibration_from_json(pathlib.Path(args.calibration_json))
        device = args.device
    else:
        from flashinfer_bench.device import calibration, default_device_type, get_accelerator

        accel = get_accelerator(args.device or default_device_type())
        device = args.device or accel.list_devices()[0]
        cal = calibration.get(device)
        if cal is None:
            raise SystemExit(
                "calibration unavailable on this device; run scripts/calibrate_part.py or pass --calibration-json"
            )

    bytes_overrides = _parse_kv_option(args.bytes_min, int)
    patterns = _parse_kv_option(args.pattern, lambda v: tuple(int(x) for x in v.split(":")))

    prelim = candidates_from_report(report, resolution, {}, bytes_overrides)
    measurements: Dict[str, Dict[str, Any]] = (
        json.loads(meas_path.read_text()) if meas_path.is_file() else {}
    )
    if args.measure:
        if not args.harness_dir:
            ap.error("--measure needs --harness-dir")
        measurements = measure_harnesses(
            prelim,
            pathlib.Path(args.harness_dir),
            meas_path,
            args.python,
            args.rounds,
            args.calls,
            args.remeasure,
        )

    def bw_pattern(run_bytes: int, stride_bytes: int) -> Optional[float]:
        if args.calibration_json or device is None:
            return None
        from flashinfer_bench.device import calibration

        return calibration.strided_read_bandwidth_gbs(device, run_bytes, stride_bytes)

    native: Optional[Callable[[str], Optional[bool]]] = None
    if device is not None and not args.calibration_json:
        from flashinfer_bench.device import get_accelerator

        caps = get_accelerator(device).capabilities(device)
        native = caps.is_native_dtype

    # An unreadable dataset is not an empty one: the gate then reads
    # definitions_matching=None and the apply mechanisms are unavailable, not "without a
    # definition". The reason is printed once here and the gate's detail says "unreadable".
    matcher: Callable[[Candidate], Optional[List[str]]] = lambda c: None  # noqa: E731
    if not pathlib.Path(args.dataset).is_dir():
        print(
            f"WARNING: no dataset at {args.dataset}; definition_exists reads None for every "
            "candidate and the apply mechanisms are unavailable.",
            file=sys.stderr,
        )
    else:
        try:
            matcher = definition_matcher(args.dataset)
        except Exception as exc:
            print(
                f"WARNING: dataset at {args.dataset} not readable ({exc}); definition_exists "
                "reads None for every candidate and the apply mechanisms are unavailable.",
                file=sys.stderr,
            )

    cands, rows = run(
        report,
        resolution,
        measurements,
        cal,
        run_id,
        presets=read_presets(pathlib.Path(args.xe_fuse)),
        post_op_algorithms=read_post_op_algorithms(pathlib.Path(args.onednn_include)),
        definitions_matching=matcher,
        gaps=load_gaps(pathlib.Path(args.gaps) if args.gaps else None),
        cutoff=args.cutoff,
        patterns=patterns,
        bw_pattern=bw_pattern,
        bytes_overrides=bytes_overrides,
        native=native,
    )
    cutoff = default_cutoff(report) if args.cutoff is None else args.cutoff
    write_outputs(
        out_dir,
        run_id,
        cands,
        rows,
        cal,
        cutoff,
        str(report_path),
        report_sha1=report_sha1,
        model=report.get("model"),
    )
    print_summary(cands, rows, out_dir)


if __name__ == "__main__":
    main()
