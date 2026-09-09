"""Derive a producer->consumer pair baseline from two generated harnesses.

A `fusion_callsite` trial replaces two calls the stack makes -- a GEMM and the elementwise
op that consumes its output -- with one. Its baseline is therefore the pair, called exactly
as production calls them: the producer's op on the producer's inputs, its output handed to
the consumer's op in the argument slot the model uses. Neither op is reimplemented.

The derived file is the *consumer's* generated harness with the pair wired in after it, so
its `OP`, `CALLS` and `get_inputs()` tensor lines stay verbatim: that is what ties a harness
to its routed candidate, and the consumer is the candidate a fusion row is about. The
producer is imported from the path recorded in the file; its weight can be rotated over a
pool larger than the cache with ``FIB_WEIGHT_POOL_MB``, as a decode step streams it.

    python tools/kernel-harness/trials/pair_harness.py <consumer.py> <producer.py> \
        --replaces <index of the consumer argument the producer's output becomes> [-o out.py]

The derived harness's `get_inputs()` is the producer's inputs followed by the consumer's
inputs minus the replaced one; `forward` returns what the consumer returns.
"""

from __future__ import annotations

import argparse
import pathlib
import re

_PAIR_CODE = '''

# --- producer -> consumer pair (added by tools/kernel-harness/trials/pair_harness.py) ------
import importlib.util as _ilu
import math as _math
import os as _os

PRODUCER = {producer!r}
REPLACES = {replaces}
SCALARS = {scalars}  # non-tensor literals the consumer's forward passes, in order
_POOL_ENV = "FIB_WEIGHT_POOL_MB"

_PAD_ENV = "FIB_WEIGHT_PAD"


def _prepare_weight(w):
    """The delivered pitch fix, when asked for: rows moved off the memory-channel period."""
    if _os.environ.get(_PAD_ENV, "") not in ("1", "true", "yes", "on"):
        return w
    from flashinfer_bench.integration import weight_layout as _wl

    padded = _wl.pad_rows_off_channel_period(w)
    return w if padded is None else padded

_ConsumerModel = Model  # captured before the rebinding at the end of this file

_spec = _ilu.spec_from_file_location("pair_producer", PRODUCER)
_producer = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_producer)



def _pool_copies(t):
    pool_mb = int(_os.environ.get(_POOL_ENV, "0"))
    if pool_mb <= 0:
        return 1
    return max(1, _math.ceil(pool_mb * 2**20 / (t.numel() * t.element_size())))


class _Pair(nn.Module):
    """Producer op, then consumer op on its output: the two calls a fusion replaces."""

    def __init__(self):
        super().__init__()
        self.producer = _producer.Model(*_producer.get_init_inputs())
        self.consumer = _ConsumerModel()
        self.n_producer = None
        self._slot, self._ptr, self._pool, self._calls = None, None, [], 0

    def forward(self, *args):
        n = self.n_producer
        if n is None:
            n = self.n_producer = len(_producer.get_inputs())
        prod_args = list(args[:n])
        slot = self._slot
        if slot is None or prod_args[slot].data_ptr() != self._ptr:
            tensors = [(i, t) for i, t in enumerate(prod_args) if hasattr(t, "data_ptr")]
            slot, w = max(tensors, key=lambda it: it[1].numel() * it[1].element_size())
            self._slot, self._ptr = slot, w.data_ptr()
            self._pool = [_prepare_weight(w)] + [_prepare_weight(w.clone()) for _ in range(_pool_copies(w) - 1)]
            self._calls = 0
        if len(self._pool) > 1:
            prod_args[slot] = self._pool[self._calls % len(self._pool)]
            self._calls += 1
        else:
            prod_args[slot] = self._pool[0]
        produced = self.producer(*prod_args)
        cons_args = list(args[n:])
        cons_args.insert(REPLACES, produced)
        return self.consumer(*cons_args)


_consumer_get_inputs = get_inputs


def get_inputs():
    cons = _consumer_get_inputs()
    return list(_producer.get_inputs()) + [t for i, t in enumerate(cons) if i != REPLACES]


Model = _Pair
'''


def derive(consumer: pathlib.Path, producer: pathlib.Path, replaces: int, dst: pathlib.Path) -> pathlib.Path:
    text = consumer.read_text()
    for needed in (r'^OP = "', r"^class Model\(", r"^def get_inputs\("):
        if not re.search(needed, text, re.M):
            raise SystemExit(f"{consumer} is not a generated harness (missing {needed!r})")
    if not re.search(r'^OP = "', producer.read_text(), re.M):
        raise SystemExit(f"{producer} is not a generated harness")
    call = re.search(r"return _op\(\)\((.*)\)", text)
    scalars = []
    if call:
        for tok in call.group(1).split(","):
            tok = tok.strip().split(" or ")[0].strip()
            try:
                scalars.append(float(tok) if "." in tok or "e" in tok else int(tok))
            except ValueError:
                pass
    header = (
        f'"""Pair derivation: {producer.as_posix()} -> {consumer.as_posix()}; '
        'see pair_harness.py."""\n'
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(header + text + _PAIR_CODE.format(producer=producer.as_posix(), replaces=replaces, scalars=scalars))
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("consumer")
    ap.add_argument("producer")
    ap.add_argument("--replaces", type=int, required=True)
    ap.add_argument("-o", "--output")
    args = ap.parse_args()
    consumer, producer = pathlib.Path(args.consumer), pathlib.Path(args.producer)
    dst = (
        pathlib.Path(args.output)
        if args.output
        else consumer.parent.parent / "trials" / "generated" / f"pair_{producer.stem}__{consumer.stem}.py"
    )
    print(derive(consumer, producer, args.replaces, dst))


if __name__ == "__main__":
    main()
