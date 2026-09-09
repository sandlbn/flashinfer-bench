"""oneDNN matmul called direct on PyTorch's queue, weight descriptor 'ba'.

See tools/kernel-harness/trials/onednn_call.py: engine, stream and primitive are kept alive
across calls, and for 'ba' == any the weight is offered as format_tag::any and reordered
once into whatever layout the library picks.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from onednn_call import make  # noqa: E402

OP = "aten.linear.default"
CALLS = 28

Model, get_inputs, get_init_inputs = make("ba")
