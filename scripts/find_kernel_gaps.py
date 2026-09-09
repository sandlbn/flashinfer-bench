"""Find hot operations that have no kernel, and say what to do about each.

`/profile-intel` ranks families that already have definitions. This answers the prior
question: which of the operations actually burning device time are *not represented at all*,
and of those, which are cheap rewrites rather than new kernels.

The distinction matters because the biggest wins found this way were not new kernels. On
a hybrid SSM model the top op was an `aten::sum` over a rank-6 tensor, the single largest
consumer of device time; it is a batched GEMM written as broadcast-multiply-then-sum, and
rewriting it as `bmm` was orders of magnitude faster with no kernel written at all.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Eager-mode shapes that are almost always a cheaper operation in disguise.
REWRITES = [
    (
        # `cumsum` also ends in "sum" and is a scan, not a contraction -- excluding it
        # matters because it is a genuinely different operation with a different fix.
        lambda op, rank, shapes: re.sub(r"^aten::", "", op) == "sum" and rank >= 5,
        "contraction materialised as broadcast-multiply-then-sum",
        "Express it as `torch.bmm`/`einsum`. G[b,i,j,h]=sum_s A[b,i,h,s]*B[b,j,h,s] is a "
        "batched GEMM; the rank-5+ intermediate exists only because it is not written as "
        "one, and the rewrite is typically orders of magnitude faster.",
    ),
    (
        lambda op, rank, shapes: op.endswith("mul") and rank >= 5,
        "broadcast multiply feeding a reduction",
        "Look at what consumes it -- if a sum follows, the pair is one contraction.",
    ),
    (
        lambda op, rank, shapes: op.endswith(("cat", "stack")) and rank >= 3,
        "materialising concatenation",
        "Often removable by a layout change or by fusing the producer.",
    ),
    (
        lambda op, rank, shapes: op.endswith(("copy_", "contiguous", "clone")),
        "layout materialisation",
        "A transpose or permute forced a copy. Change the layout at load time instead.",
    ),
]


def known_definitions(local: Path) -> dict:
    """definition name -> op_type, for deciding whether an op is represented at all."""
    out = {}
    for f in (local / "definitions").rglob("*.json"):
        try:
            out[json.loads(f.read_text())["name"]] = f.parent.name
        except Exception:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="xpu:0")
    ap.add_argument("--local", type=Path, default=Path("tmp/flashinfer-trace"))
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--min-share", type=float, default=1.0, help="Percent of device time.")
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Execute modeling code from the model repository. Needed for any architecture "
             "transformers does not ship; off by default because it runs third-party Python.",
    )
    a = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = {"trust_remote_code": True} if a.trust_remote_code else {}
    if a.trust_remote_code:
        # Same shims the extractor applies: published modeling code often targets
        # an older transformers, and the failures look like our bug.
        import importlib.util as _ilu, pathlib as _pl
        _spec = _ilu.spec_from_file_location(
            "_fib_extract", _pl.Path(__file__).with_name("extract_model_kernels_xpu.py"))
        _ex = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_ex)
        _ex._compat_remote_code()
    tok = AutoTokenizer.from_pretrained(a.model, **hf)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, **hf).to(a.device)
    text = a.prompt
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            [{"role": "user", "content": a.prompt}], add_generation_prompt=True, tokenize=False
        )
    ids = tok(text, return_tensors="pt").to(a.device)

    with torch.no_grad():
        model.generate(**ids, max_new_tokens=4, do_sample=False)
    torch.xpu.synchronize()
    with (
        torch.no_grad(),
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
        ) as prof,
    ):
        model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=False)
        torch.xpu.synchronize()

    rows = []
    for e in prof.key_averages(group_by_input_shape=True):
        us = getattr(e, "self_device_time_total", 0) or getattr(e, "self_xpu_time_total", 0)
        # aten rows carry both the shapes and the device time of the kernels they launched.
        if us <= 0 or (getattr(e, "self_cpu_time_total", 0) or 0) <= 0:
            continue
        shapes = [
            s for s in (getattr(e, "input_shapes", None) or []) if isinstance(s, (list, tuple))
        ]
        rank = max((len(s) for s in shapes), default=0)
        rows.append((e.key, us, e.count, rank, shapes))
    total = sum(r[1] for r in rows) or 1.0
    rows.sort(key=lambda r: -r[1])

    defs = known_definitions(a.local)
    print(f"\n  Hot operations in {a.model} -- device total {total / 1000:.0f} ms\n")
    print(f"  {'op':22} {'share':>7} {'ms':>9} {'calls':>7}  assessment")
    print("  " + "-" * 96)

    actions = []
    for key, us, count, rank, shapes in rows:
        share = 100.0 * us / total
        if share < a.min_share:
            continue
        verdict, advice = "represented — see /profile-intel routing", None
        for pred, name, how in REWRITES:
            if pred(key, rank, shapes):
                verdict, advice = f"REWRITE: {name}", how
                break
        else:
            base = re.sub(r"^aten::", "", key)
            if not any(base in d for d in defs):
                verdict = "no definition covers this op"
        print(f"  {key[:22]:22} {share:6.1f}% {us / 1000:8.2f} {count:7}  {verdict}")
        if advice:
            actions.append((share, key, str(shapes[0])[:34] if shapes else "", advice))

    if actions:
        print("\n  Rewrites worth measuring, largest first:")
        for share, key, shape, advice in sorted(actions, reverse=True):
            print(f"\n  [{share:.1f}%] {key}  {shape}")
            for line in advice.split(". "):
                if line.strip():
                    print(f"      {line.strip().rstrip('.')}.")
        print("\n  Verify equivalence and measure the ceiling BEFORE writing anything:")
        print("      assert torch.allclose(naive(), rewritten(), atol=1e-3)")
        print("      then time both -- if the gap is small, there is nothing here.")
    else:
        print("\n  No rewritable materialisations above the threshold.")


if __name__ == "__main__":
    main()
