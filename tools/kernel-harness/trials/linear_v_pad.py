"""`aten.linear` through the `pad` call shape; see linear_call.py."""

import sys

sys.path.insert(0, "tools/kernel-harness/trials")
from linear_call import make  # noqa: E402

Model, get_inputs, get_init_inputs = make("pad")
