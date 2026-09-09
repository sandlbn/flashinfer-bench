"""What each adapter actually did, per call, reported once at exit.

An A/B that patches a method proves nothing on its own: `apply` can miss for reasons that
are invisible from outside -- no definition of that shape in the dataset, no solution that
clears the correctness gate, a runtime key that matches no recorded workload. Each of those
falls back silently and correctly, and the run then measures vLLM against vLLM.

That happened three times in one session on this machine. A vLLM A/B was reported as
parity-within-noise when in fact the tolerance gate had rejected every bf16 solution and
nothing was ever substituted; a later run substituted exactly one kernel of the three that
were patched, because the dataset had no definition at the served model's hidden size.
Both looked like a result and were an absence of one.

So each adapter records the outcome of every call, and the counts print to stderr at exit
-- stderr because this runs inside vLLM's EngineCore subprocess, where nothing has
configured a handler for this package's logger and `logger.info` is silently dropped.
"""

from __future__ import annotations

import atexit
import collections
import logging
import sys
from typing import Dict

logger = logging.getLogger(__name__)

_SEEN: "collections.Counter[tuple]" = collections.Counter()


def record(adapter: str, outcome: str, detail: str = "") -> None:
    """Note one call: ``outcome`` is "applied", "no-solution", "deferred" or "unsupported"."""
    _SEEN[(adapter, outcome, detail)] += 1


def bucket(n: int) -> str:
    """Power-of-two bucket, so a histogram stays readable across three orders."""
    if n <= 1:
        return "1"
    hi = 1 << (n - 1).bit_length()
    return f"{hi // 2 + 1}-{hi}"


def dispatch_stats() -> Dict[str, int]:
    """Every recorded outcome, as ``{"rmsnorm applied h2048": n, ...}``."""
    return {
        " ".join(p for p in (adapter, outcome, detail) if p): n
        for (adapter, outcome, detail), n in sorted(_SEEN.items())
    }


def reset() -> None:
    _SEEN.clear()


def format_report() -> str:
    if not _SEEN:
        return "[flashinfer-bench] no adapter calls recorded"
    lines = ["[flashinfer-bench] adapter dispatch:"]
    by_adapter: "collections.Counter[str]" = collections.Counter()
    applied: "collections.Counter[str]" = collections.Counter()
    for (adapter, outcome, _), n in _SEEN.items():
        by_adapter[adapter] += n
        if outcome == "applied":
            applied[adapter] += n
    for adapter in sorted(by_adapter):
        total, hit = by_adapter[adapter], applied[adapter]
        lines.append(f"  {adapter}: {total} call(s), {hit} applied ({100.0 * hit / total:.1f}%)")
    lines.append(f"  detail: {dispatch_stats()}")
    return "\n".join(lines)


def _report() -> None:
    if not _SEEN:
        return
    message = format_report()
    logger.info("%s", message)
    print(message, file=sys.stderr, flush=True)


atexit.register(_report)
