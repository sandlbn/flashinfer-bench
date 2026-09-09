"""Fused decode GEMV, `addnorm` epilogue -- control: the product alone, then the production consumer kernel; see fused_gemv.py."""

import sys

sys.path.insert(0, "tools/kernel-harness/trials")
from fused_gemv import make  # noqa: E402

Model, get_inputs, get_init_inputs = make("addnorm", dict(sg=32, vec=8, rows=8, unroll=2), control=True)
