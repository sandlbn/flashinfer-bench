import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from linear_call import make

Model, get_inputs, get_init_inputs = make("splitn")
