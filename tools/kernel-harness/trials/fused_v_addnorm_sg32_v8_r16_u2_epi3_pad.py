"""Fused decode GEMV, `addnorm` epilogue, variant `epi3_pad`; see fused_gemv.py."""

import sys

sys.path.insert(0, "tools/kernel-harness/trials")
from fused_gemv import make  # noqa: E402

Model, get_inputs, get_init_inputs = make("addnorm", dict(sg=32, vec=8, rows=16, unroll=2, epi=3, pad=1))
