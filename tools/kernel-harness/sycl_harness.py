"""Turn inline SYCL (or oneDNN) source into the harness contract the trial loop runs.

The loop in `scripts/kernel_trials.py` is language-agnostic: it imports a file exposing
`Model`, `get_inputs` and `get_init_inputs`, and times it. What a SYCL trial needs is a way
to go from source text to a callable without hand-rolling a build, and the repo already has
that -- the same builder the benchmark uses, so a trial is compiled exactly as a Solution
would be and inherits its calling convention.

A trial file is then just source plus a few lines:

    from sycl_harness import build, inputs_for

    SOURCE = r'''...'''

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.run = build("rmsnorm_h2560", SOURCE)
        def forward(self, x, w):
            out = torch.empty_like(x)
            self.run(x, w, out)
            return out

    get_inputs = inputs_for("rmsnorm_h2560", batch_size=2048)

oneDNN needs no different path -- it is SYCL with `dependencies=["onednn"]`, which `build`
takes as an argument.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any, Callable, List

DATASET = pathlib.Path("tmp/flashinfer-trace")


def _definition(name: str):
    from flashinfer_bench.data import TraceSet

    trace_set = TraceSet.from_path(str(DATASET))
    if name not in trace_set.definitions:
        raise SystemExit(f"{name!r} is not in {DATASET}")
    return trace_set.definitions[name]


_EXPORT = re.compile(r"TVM_FFI_DLL_EXPORT_TYPED_FUNC\s*\(\s*([A-Za-z_]\w*)")


def _entry_point_of(source: str, path: str) -> str:
    """Read the exported symbol out of the source rather than being told it.

    The name is already stated once, in the export macro. Taking it as an argument makes it
    a second place to be wrong, and being wrong there fails at build time with a message
    about a missing symbol rather than about the mismatch.
    """
    names = _EXPORT.findall(source)
    if not names:
        raise SystemExit(
            "No TVM_FFI_DLL_EXPORT_TYPED_FUNC(...) found; the kernel exports no entry point."
        )
    if len(names) > 1:
        raise SystemExit(f"Source exports {names}; keep one entry point per trial.")
    return f"{path}::{names[0]}"


def build(
    definition_name: str,
    source: str,
    entry_point: str | None = None,
    dependencies: List[str] | None = None,
    language: str = "sycl",
    source_path: str = "kernel.cpp",
) -> Callable[..., Any]:
    """Compile `source` against a dataset definition and return the callable.

    Built through the same builder the benchmark uses, so a trial that wins here is a
    Solution already -- no reimplementation between tuning and deployment, and no chance of
    the two disagreeing about the calling convention.
    """
    from flashinfer_bench.compile import BuilderRegistry
    from flashinfer_bench.data import Solution

    definition = _definition(definition_name)
    entry_point = entry_point or _entry_point_of(source, source_path)
    path = entry_point.split("::")[0]
    solution = Solution.model_validate(
        {
            "name": f"{definition_name}__trial",
            "definition": definition_name,
            "author": "kernel-trials",
            "spec": {
                "language": language,
                "entry_point": entry_point,
                "target_hardware": ["xpu"],
                "dependencies": dependencies or [],
                # Destination-passing: the definition's outputs arrive as trailing arguments,
                # which is what every in-tree Intel kernel already expects.
                "destination_passing_style": True,
            },
            "sources": [{"path": path, "content": source}],
        }
    )
    runnable = BuilderRegistry.get_instance().build(definition, solution)
    return runnable.call_destination_passing


def inputs_for(definition_name: str, **axes: int) -> Callable[[], List[Any]]:
    """Build realistic inputs for a definition, from its own axes and dtypes.

    Shapes come from the definition rather than from a guess, so a trial cannot quietly
    optimize a problem the dataset does not contain. `axes` supplies the variable ones.
    """
    import torch

    definition = _definition(definition_name)

    def make() -> List[Any]:
        device = "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
        resolved = {
            name: spec.value
            for name, spec in definition.axes.items()
            if getattr(spec, "type", None) == "const"
        }
        resolved.update(axes)
        missing = [
            name
            for name, spec in definition.axes.items()
            if getattr(spec, "type", None) != "const" and name not in resolved
        ]
        if missing:
            raise SystemExit(f"{definition_name}: supply the variable axes {missing}")

        out = []
        for spec in definition.inputs.values():
            shape = [resolved[a] if isinstance(a, str) else a for a in spec.shape]
            dtype = getattr(torch, spec.dtype)
            out.append(
                torch.randn(*shape, dtype=dtype, device=device)
                if dtype.is_floating_point
                else torch.ones(*shape, dtype=dtype, device=device)
            )
        return out

    return make
