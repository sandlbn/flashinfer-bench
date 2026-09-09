"""Control: the production call, unaltered, through the linear_call variant wrapper.

Same entry point (`aten.linear.default`) and the same weight the model stores; the only
thing this adds over the routed baseline is the wrapper itself and the wrapper's operand
pool. Whatever it costs is charged here and not to any lever.
"""

import sys

sys.path.insert(0, "tools/kernel-harness/trials")
from linear_call import make  # noqa: E402

OP = "aten.linear.default"
CALLS = 28

Model, get_inputs, get_init_inputs = make("ba")
