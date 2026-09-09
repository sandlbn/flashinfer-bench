"""Fused decode GEMV, `swiglu` epilogue, variant `swz`; see fused_gemv.py."""

import sys

sys.path.insert(0, "tools/kernel-harness/trials")
from fused_gemv import make  # noqa: E402

Model, get_inputs, get_init_inputs = make("swiglu", dict(sg=16, vec=8, rows=16, unroll=1, swz=1))
