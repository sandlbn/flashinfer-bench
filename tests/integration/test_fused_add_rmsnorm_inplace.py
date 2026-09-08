"""The fused-add RMSNorm kernel must give the same answer when its buffers alias.

The vLLM adapter calls it in place -- `output is hidden_states` and
`residual_out is residual` -- because vLLM's own kernel is in place and allocating a fresh
pair instead cost two [tokens, hidden] allocations on every call, which showed up as a
throughput regression with no device-time cause.

That is only safe because of how the kernel is written: the group reduction is a barrier
between the pass that reads and the pass that writes, and within the writing pass each
work-item writes exactly the index it just read. Neither is guaranteed by the interface,
so it is pinned here -- a future kernel that reorders those passes has to be caught.
"""

import pytest
import torch

xpu = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()), reason="needs an Intel GPU"
)

DEFINITION = "fused_add_rmsnorm_residual_h1024"
HIDDEN = 1024


@pytest.fixture(scope="module")
def runtime():
    from pathlib import Path

    from flashinfer_bench.apply import ApplyConfig, enable_apply
    from flashinfer_bench.apply.runtime import ApplyRuntime

    dataset = Path(__file__).resolve().parents[2] / "tmp" / "flashinfer-trace"
    if not (dataset / "definitions" / "rmsnorm" / f"{DEFINITION}.json").exists():
        pytest.skip(f"{DEFINITION} is not in the local dataset clone")
    ApplyRuntime._stack.clear()
    rt = enable_apply(
        str(dataset), ApplyConfig(max_atol=0.02, max_rtol=0.02, on_miss_policy="use_def_best")
    )
    if rt._table.def_best.get(DEFINITION) is None:
        pytest.skip("no benchmarked solution for this definition on this machine")
    yield rt
    rt.stop()


@xpu
@pytest.mark.parametrize("tokens", [1, 7, 32, 129, 512, 4096])
def test_aliased_buffers_match_separate_ones(runtime, tokens):
    from flashinfer_bench.apply import apply

    dev = "xpu:0"
    x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device=dev)
    r = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device=dev)
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device=dev)

    out, res = torch.empty_like(x), torch.empty_like(r)
    apply(
        DEFINITION,
        kwargs={"hidden_states": x, "residual": r, "weight": w, "output": out, "residual_out": res},
        fallback=lambda **_k: pytest.fail("no solution dispatched"),
    )

    xa, ra = x.clone(), r.clone()
    apply(
        DEFINITION,
        kwargs={"hidden_states": xa, "residual": ra, "weight": w, "output": xa, "residual_out": ra},
        fallback=lambda **_k: pytest.fail("no solution dispatched"),
    )

    assert torch.equal(out, xa), "aliasing the output changed the normalized result"
    assert torch.equal(res, ra), "aliasing the residual changed the summed residual"


@xpu
def test_the_summed_residual_is_actually_returned(runtime):
    """The whole point of the two-output form: the caller must not have to recompute it."""
    from flashinfer_bench.apply import apply

    dev = "xpu:0"
    x = torch.randn(64, HIDDEN, dtype=torch.bfloat16, device=dev)
    r = torch.randn(64, HIDDEN, dtype=torch.bfloat16, device=dev)
    w = torch.ones(HIDDEN, dtype=torch.bfloat16, device=dev)
    out, res = torch.empty_like(x), torch.empty_like(r)
    apply(
        DEFINITION,
        kwargs={"hidden_states": x, "residual": r, "weight": w, "output": out, "residual_out": res},
        fallback=lambda **_k: pytest.fail("no solution dispatched"),
    )
    torch.testing.assert_close(res.float(), (x.float() + r.float()), atol=0.05, rtol=0.02)
