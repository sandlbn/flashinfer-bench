# Added by kernel_trials finalize: this trial imported helpers from the series
# directory it was written in. Resolved relative to this file, never absolute.
import pathlib as _p, sys as _s
for _r in ['../trials/e95e9118e7-library_call', '../trials']:
    _s.path.insert(0, str((_p.Path(__file__).resolve().parent / _r).resolve()))
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from onednn_call import make

Model, get_inputs, get_init_inputs = make("any")
