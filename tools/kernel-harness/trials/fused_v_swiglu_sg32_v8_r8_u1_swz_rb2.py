"""Fused decode GEMV, `swiglu` epilogue, variant `swz_rb2`; see fused_gemv.py."""

import sys

sys.path.insert(0, "tools/kernel-harness/trials")
from fused_gemv import make  # noqa: E402

Model, get_inputs, get_init_inputs = make("swiglu", dict(sg=32, vec=8, rows=8, unroll=1, swz=1, rb=2))
