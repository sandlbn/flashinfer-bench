"""Derive a streaming-weight baseline from a generated harness, keeping its identity.

A generated harness calls its op on one weight in a tight loop, so after the first call
the weight sits in the last-level cache and the loop measures either the cache or, below
that, the host cost of issuing the call. A decode step touches every weight once, so in
serving the operand arrives from device memory. The derived harness is the generated one
with its `forward` routed through a pool of copies larger than the cache, chosen by
``FIB_WEIGHT_POOL_MB``; with the variable unset it is the generated harness unchanged.

The derivation is textual and keeps `OP`, `CALLS` and the `get_inputs()` tensor lines
verbatim, which is what ties a harness to its routed candidate, so the derived file can
open a series with `--bound`.

    python tools/kernel-harness/trials/stream_harness.py <generated.py> [-o out.py]
"""

from __future__ import annotations

import argparse
import pathlib
import re

_POOL_CODE = '''

# --- streaming pool (added by tools/kernel-harness/trials/stream_harness.py) ---------------
import math as _math
import os as _os

_POOL_ENV = "FIB_WEIGHT_POOL_MB"

_PAD_ENV = "FIB_WEIGHT_PAD"


def _prepare_weight(w):
    """The delivered pitch fix, when asked for: rows moved off the memory-channel period."""
    if _os.environ.get(_PAD_ENV, "") not in ("1", "true", "yes", "on"):
        return w
    from flashinfer_bench.integration import weight_layout as _wl

    padded = _wl.pad_rows_off_channel_period(w)
    return w if padded is None else padded


def _pool_copies(t):
    pool_mb = int(_os.environ.get(_POOL_ENV, "0"))
    if pool_mb <= 0:
        return 1
    return max(1, _math.ceil(pool_mb * 2**20 / (t.numel() * t.element_size())))


class _Streaming(Model):
    """The generated Model, with its largest tensor argument rotated over a pool of copies.

    The largest argument is the weight in every GEMM-shaped op; rotating it -- and only
    it -- keeps the activations resident exactly as a decode step would.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._slot, self._ptr, self._pool, self._calls = None, None, [], 0

    def forward(self, *args):
        slot = self._slot
        if slot is None or args[slot].data_ptr() != self._ptr:
            tensors = [(i, t) for i, t in enumerate(args) if hasattr(t, "data_ptr")]
            if not tensors:
                return super().forward(*args)
            slot, w = max(tensors, key=lambda it: it[1].numel() * it[1].element_size())
            self._slot, self._ptr = slot, w.data_ptr()
            self._pool = [_prepare_weight(w)] + [_prepare_weight(w.clone()) for _ in range(_pool_copies(w) - 1)]
            self._calls = 0
        pool = self._pool
        args = list(args)
        if len(pool) == 1:
            args[slot] = pool[0]
            return super().forward(*args)
        args[slot] = pool[self._calls % len(pool)]
        self._calls += 1
        return super().forward(*args)


Model = _Streaming
'''


def derive(src: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
    text = src.read_text()
    for needed in (r'^OP = "', r"^class Model\(", r"^def get_inputs\("):
        if not re.search(needed, text, re.M):
            raise SystemExit(f"{src} is not a generated harness (missing {needed!r})")
    header = (
        f'"""Streaming-weight derivation of {src.as_posix()}; see stream_harness.py."""\n'
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(header + text + _POOL_CODE)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("harness")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()
    src = pathlib.Path(args.harness)
    dst = pathlib.Path(args.output) if args.output else src.parent.parent / "trials" / "generated" / f"stream_{src.name}"
    print(derive(src, dst))


if __name__ == "__main__":
    main()
