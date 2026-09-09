"""Reproduce the oneDNN selection for aten::linear at the shape the model called it at.

    ONEDNN_VERBOSE=all python repro.py 2>&1 | grep '^onednn_verbose'
"""

import torch

DEVICE = "xpu:0"
SPEC = [["T", [4, 2048], "bfloat16"], ["T", [2560, 2048], "bfloat16"], ["T", [2560], "bfloat16"]]


def mk(a):
    if isinstance(a, list) and a and a[0] == "T":
        _, shape, dtype = a
        f = torch.randn if dtype.startswith(("float", "bfloat")) else torch.ones
        return f(shape, dtype=getattr(torch, dtype), device=DEVICE)
    if isinstance(a, list):
        return [mk(i) for i in a]
    return a


# A custom op only exists once the package registering it is imported; ATen ops need nothing.
fn = getattr(getattr(torch.ops, "aten"), "linear")
fn(*[mk(a) for a in SPEC])
getattr(torch, DEVICE.split(":")[0]).synchronize()
