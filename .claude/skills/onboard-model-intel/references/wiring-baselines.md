# Wiring an upstream kernel in as a baseline

`flashinfer_bench/integration/xpu_kernels.py` holds a `REGISTRY` of `BaselineKernel`
records. A definition matches a record only when **all three** hold exactly:

```python
definition.op_type == kernel.op_type
tuple(definition.inputs)  == kernel.inputs      # names, in declaration order
tuple(definition.outputs) == kernel.outputs
```

Names, not shapes or dtypes. A near-miss produces **no baseline** — logged under "No
upstream kernel matches" and skipped — so a typo looks exactly like "not registered yet".

**1. Read the real signature.**

```python
import torch, vllm_xpu_kernels._C  # noqa: F401
print(torch.ops._C.rms_norm.default._schema)
# _C::rms_norm(Tensor! out, Tensor input, Tensor weight, float epsilon) -> ()

import sgl_kernel, inspect
print(inspect.signature(sgl_kernel.rmsnorm))
```

`Tensor! out` first means destination-passing and in-place.

**2. Check semantics, not arity.** A kernel belongs in the registry only when it computes
the same function as the definition's reference (`gemma_rms_norm` scales by `(1 + weight)`
and does not match a plain RMSNorm definition).

**3. Write the wrapper** — plain Python source as a string, defining `run(...)` with the
definition's inputs in order. Allocate outputs yourself for destination-passing ops, and
**clone in-place inputs**: the benchmark reuses tensors across trials.

`run(...)` takes the definition's **inputs** and nothing else. Scalars the kernel needs but
the definition does not declare — epsilon, `is_neox` — are `constants`: the template is
`str.format`-ed with them before it is compiled, so they become module-level literals.
Naming one as a `run` argument makes the entry match nothing.

```python
_MY_KERNEL_VLLM = """import torch
import vllm_xpu_kernels._C  # noqa: F401  (registers torch.ops._C)

EPS = {eps!r}


def run(hidden_states, weight):          # exactly the definition's inputs, in order
    out = torch.empty_like(hidden_states)
    torch.ops._C.my_kernel(out, hidden_states, weight, EPS)
    return out
"""
```

Declare `constants=("eps",)` on the registry entry. `definition_eps()` resolves it from the
definition's reference source, warning if the definition states none.

**4. Add the registry entry**, with `inputs`/`outputs` copied from the **definition**, not
from the upstream signature.

**5. Verify it matched.**

```python
from flashinfer_bench.data import TraceSet
from flashinfer_bench.integration import available_providers, find_baselines

ts = TraceSet.from_path("tmp/flashinfer-trace")
d = ts.definitions["<name>"]
print("providers:", available_providers())
print("signature:", d.op_type, tuple(d.inputs), "->", tuple(d.outputs))
print("matched:", [k.solution_name for k in find_baselines(d)])
```

Empty `matched`: op_type spelled differently, an input the definition declares that the
kernel does not take (`eps` as an input vs a constant), or outputs in the other order. Fix
the entry to match reality — never rename a dataset definition to make a baseline match.
