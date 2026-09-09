"""linear_call variant 'matmul' at the routed shape; see tools/kernel-harness/trials/linear_call.py."""

import sys

sys.path.insert(0, "tools/kernel-harness/trials")
from linear_call import make  # noqa: E402

OP = "aten.linear.default"
CALLS = 28

Model, get_inputs, get_init_inputs = make("matmul")
