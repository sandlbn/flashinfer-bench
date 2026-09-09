"""Control for a fusion series: oneDNN called directly (weight `ba`), then the production consumer kernel; see onednn_call.py."""

import sys

sys.path.insert(0, "tools/kernel-harness/trials")
from onednn_call import make_pair_control  # noqa: E402

Model, get_inputs, get_init_inputs = make_pair_control("ba")
