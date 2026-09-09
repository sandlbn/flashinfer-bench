"""oneDNN matmul called directly on the framework queue, weight layout `ba`; see onednn_call.py."""

import sys

sys.path.insert(0, "tools/kernel-harness/trials")
from onednn_call import make  # noqa: E402

Model, get_inputs, get_init_inputs = make("ba")
