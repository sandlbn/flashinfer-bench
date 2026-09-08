"""Profile a model on an Intel GPU and route each hot kernel family to the skill that fixes it.

Optimizing without this is guessing: a 3x win on 2% of device time is worth less than a
1.15x win on 50%. This ranks what the model actually spends time in, then says which skill
owns each family.

Uses `torch.profiler` with `ProfilerActivity.XPU`, which reports per-kernel device time.
unitrace is not used here -- on Level Zero drivers tested it reports only a Device Timing
Summary with no per-kernel rows. unitrace remains the tool for register spill and GRF mode
(its Kernel Properties section), which torch.profiler does not expose.
"""

from __future__ import annotations

import argparse
import collections
import re

# kernel-name pattern -> (family, which skill owns it, what to do)
ROUTES = [
    (
        # Must precede the gemm route: these names usually contain "gemm" too, and a
        # quantized GEMM is a different problem. Xe2's kernel.db is overwhelmingly tuned
        # for low-bit weights against f16 compute (s4/nf4/s8 x f16) and has zero native
        # bf16 rows, so a quantized path here is running on Intel's *tuned* strategies
        # while a bf16 one inherits PVC's. Standalone dequant/repack kernels are the usual
        # cost -- they are a separate launch and a full pass over the weights.
        r"quant|dequant|awq|gptq|woq|mxfp|nvfp|fp8|fp4|int4|s4|u4|scaled_mm",
        "quantized gemm",
        "/optimize-intel-kernels",
        "check xe-matrix.md first: fp8 and block-scaled mxfp4/mxfp8 are Crescent Island only "
        "and run emulated here; s4/nf4/s8 x f16 is native. Fold dequant into the GEMM rather "
        "than beating the matmul",
    ),
    (
        r"gemm|matmul|linear|xetla|dnnl",
        "gemm",
        "/optimize-onednn",
        "oneDNN is already the path; fix how it is called (layout, caching, post-ops, no host block)",
    ),
    (
        r"attention|sdpa|fmha|flash|paged",
        "attention",
        "/onboard-model-intel Phase 5",
        "sgl-kernel-xpu is the only Intel attention source; wire it as a baseline first",
    ),
    (
        # A norm shows up as its reduction kernel (ReduceKernel<...ReduceOp<float>>),
        # never by the word "norm" -- match the reduction or it lands in "other".
        r"norm|rms|reduce",
        "norm",
        "/optimize-intel-kernels",
        "memory-bound: vectorize loads, multi-row work-groups; in-tree SYCL beats vLLM here",
    ),
    (
        r"silu|gelu|activation|elementwise|vectorized|unrolled",
        "elementwise",
        "/optimize-intel-kernels",
        "fuse into the producing GEMM's epilogue (oneDNN post-ops) before writing a kernel",
    ),
    (
        r"rope|rotary",
        "rope",
        "/optimize-intel-kernels",
        "vllm-xpu-kernels ships rotary_embedding; benchmark against it before writing one",
    ),
    (
        r"softmax|sampling|topk|top_p",
        "sampling",
        "/onboard-model-intel Phase 5",
        "sgl-kernel-xpu has the sampling family; verify it is built before registering",
    ),
    (
        r"copy|cat|reshape|permute|contiguous|transpose",
        "data movement",
        "(no kernel to write)",
        "look for an avoidable materialisation -- a layout or fusion change removes it",
    ),
]


def achievable_speedups(local: str, hardware_id: str) -> dict:
    """Best measured speedup per op_type on THIS hardware, from the trace dataset.

    Ranking by share alone picks the wrong target. What matters is how much of a family's
    time is actually recoverable, and that depends on how far the current kernel is from the
    best one anybody has produced on this part.

    Traces are grouped by `hardware_id` and never compared across devices -- an Intel
    speedup and an NVIDIA speedup are not the same quantity, and mixing them here produced
    a confident 0.42x for paged attention that was pure cross-hardware noise.

    Two evidence sources, in order:
      1. best solution vs the vendor baseline, where a baseline exists;
      2. best solution vs the definition's reference, where it does not -- on Intel the
         reference for a GEMM is `torch.matmul`, which *is* oneDNN, so that ratio is
         genuinely "vs the production path".
    """
    import json
    import statistics
    from collections import defaultdict
    from pathlib import Path

    root = Path(local)
    op_of = {}
    for f in (root / "definitions").rglob("*.json"):
        try:
            op_of[json.loads(f.read_text())["name"]] = f.parent.name
        except Exception:
            continue

    best = defaultdict(dict)
    sizes: dict = {}
    for f in (root / "traces").rglob("*.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            ev = r.get("evaluation") or {}
            if ev.get("status") != "PASSED":
                continue
            lat = (ev.get("performance") or {}).get("latency_ms")
            hw = (ev.get("environment") or {}).get("hardware_id")
            op = op_of.get(r.get("definition", ""))
            if not lat or not op or hw != hardware_id:
                continue
            sol = r.get("solution", "")
            kind = (
                "baseline"
                if ("vllm" in sol or "sgl_" in sol or "flashinfer_wrapper" in sol)
                else "ours"
            )
            key = (op, r["definition"], r["workload"]["uuid"])
            ax = (r.get("workload") or {}).get("axes") or {}
            sizes[key] = max((v for v in ax.values() if isinstance(v, int)), default=0)
            prev = best[key].get(kind)
            best[key][kind] = lat if prev is None else min(prev, lat)
            ref = (ev.get("performance") or {}).get("reference_latency_ms")
            if ref:
                best[key]["reference"] = ref

    # Weight by workload size. A ratio measured at batch=1 is not evidence: repeated runs
    # of the same rmsnorm workload varied 74% at batch=1 and 0% at batch=8192, and the
    # median over all workloads read 1.09x where the measurable ones read 1.01x. Small
    # kernels at decode sizes are launch-bound, so the numbers there are scheduler noise.
    # Take the largest workload per definition, which is the one that is actually resolvable.
    per_def = defaultdict(dict)
    for (op, d, w), v in best.items():
        anchor = v.get("baseline") or v.get("reference")
        if not (anchor and v.get("ours")):
            continue
        size = sizes.get((op, d, w), 0)
        prev = per_def[(op, d)].get("size", -1)
        if size >= prev:
            per_def[(op, d)] = {"size": size, "ratio": anchor / v["ours"]}

    ratios = defaultdict(list)
    for (op, _d), v in per_def.items():
        ratios[op].append(v["ratio"])
    return {op: (statistics.median(r), len(r)) for op, r in ratios.items() if r}


def high_rank_contractions(prof, total_us: float, min_ndim: int = 5):
    """Hot `aten` ops whose operands are rank >= min_ndim.

    A state-space scan (Mamba2/SSD, GDN) is written in `transformers` as a broadcast
    multiply followed by a sum over 6-D tensors, materialising an intermediate that a fused
    chunked scan never creates -- on Zamba2-1.2B one such `aten::sum` over
    [1,1,256,256,64,128] was 27.8% of device time in 38 calls.

    This cannot be spotted from kernel names: that scan and an ordinary RMSNorm both appear
    as `ReduceKernel<1, ReduceOp<float>>`. Only the operand rank separates them, and rank is
    only visible on the aten row.
    """
    out = []
    for e in prof.key_averages(group_by_input_shape=True):
        us = getattr(e, "self_device_time_total", 0) or getattr(e, "self_xpu_time_total", 0)
        if us <= 0 or (getattr(e, "self_cpu_time_total", 0) or 0) <= 0:
            continue
        shapes = getattr(e, "input_shapes", None) or []
        rank = max((len(sh) for sh in shapes if isinstance(sh, (list, tuple))), default=0)
        if rank >= min_ndim:
            elems = 1
            for sh in shapes:
                if isinstance(sh, (list, tuple)) and len(sh) == rank:
                    for d in sh:
                        elems *= max(int(d), 1)
                    break
            out.append((e.key, us, e.count, rank, elems, shapes))
    return sorted(out, key=lambda r: -r[1])


def classify(name: str) -> tuple[str, str, str]:
    low = name.lower()
    for pattern, family, skill, advice in ROUTES:
        if re.search(pattern, low):
            return family, skill, advice
    return "other", "(investigate)", "not matched by any route; inspect the kernel name"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HuggingFace repo id.")
    ap.add_argument("--device", default="xpu:0")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Execute modeling code from the model repository. Needed for any architecture "
             "transformers does not ship; off by default because it runs third-party Python.",
    )
    ap.add_argument(
        "--local",
        default="tmp/flashinfer-trace",
        help="Trace dataset, for measured achievable speedups.",
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
    ids = tok(a.prompt, return_tensors="pt").to(a.device)

    with torch.no_grad():  # warm up, so compile/alloc is not profiled
        model.generate(**ids, max_new_tokens=4, do_sample=False)
    torch.xpu.synchronize()

    act = [ProfilerActivity.CPU, ProfilerActivity.XPU]
    with torch.no_grad(), profile(activities=act, record_shapes=True) as prof:
        model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=False)
        torch.xpu.synchronize()

    # Count device kernels only. `key_averages()` also lists the aten op that launched each
    # kernel, carrying the same device time -- `aten::mm` and `gemm_kernel` are one GEMM, not
    # two. Including both double-counts every hot path and invents a large "other" bucket.
    # A launched kernel is the row with device time and no self CPU time of its own.
    per_family: dict[str, list] = collections.defaultdict(list)
    total = 0.0
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", 0) or getattr(e, "self_xpu_time_total", 0)
        if us <= 0:
            continue
        if (getattr(e, "self_cpu_time_total", 0) or 0) > 0:
            continue  # an aten op, not the kernel it launched
        total += us
        family, skill, advice = classify(e.key)
        per_family[family].append((e.key, us, e.count, skill, advice))

    from flashinfer_bench.device import get_accelerator

    hw = get_accelerator(a.device).capabilities(a.device).canonical_id
    evidence = achievable_speedups(a.local, hw)
    FAMILY_TO_OP = {
        "gemm": "gemm",
        "norm": "rmsnorm",
        "elementwise": "activation",
        "attention": "gqa_paged",
        "rope": "rope",
        "sampling": "sampling",
    }

    print(f"\n  Device time by family -- {a.model} on {a.device} ({hw})")
    print(f"  {'family':14} {'share':>7} {'ms':>9} {'evidence':>13} {'recoverable':>12}  route")
    print("  " + "-" * 92)

    scored = []
    for family, rows in per_family.items():
        share = 100.0 * sum(r[1] for r in rows) / total
        sp, n = evidence.get(FAMILY_TO_OP.get(family, ""), (None, 0))
        # Amdahl: eliminating a family entirely recovers its share; a speedup of s recovers
        # share*(1-1/s). No evidence means unknown, not zero -- it is a measurement to make,
        # and it is reported as such rather than silently ranked last.
        recoverable = share * (1 - 1 / sp) if sp and sp > 1 else (0.0 if sp else None)
        scored.append((family, rows, share, sp, n, recoverable))

    for family, rows, share, sp, n, rec in sorted(
        scored, key=lambda x: -(x[5] if x[5] is not None else -1)
    ):
        ms = sum(r[1] for r in rows) / 1000.0
        ev = f"{sp:.2f}x (n={n})" if sp else "none yet"
        rc = "unknown" if rec is None else f"{rec:5.1f}%"
        print(f"  {family:14} {share:6.1f}% {ms:8.2f} {ev:>13} {rc:>12}  {rows[0][3]}")

    contractions = high_rank_contractions(prof, total)
    if contractions:
        share = 100.0 * sum(c[1] for c in contractions) / total
        print(f"\n  !! Materialised high-rank contractions: {share:.1f}% of device time")
        print("     A state-space / SSD scan written as broadcast-multiply + sum. The intermediate")
        print("     is created only because the scan is not fused -- see /optimize-ssm-scan.")
        for k, us, n, rank, elems, shapes in contractions[:4]:
            gb = elems * 4 / 2**30
            print(
                f"     {k[:22]:22} {us / 1000:8.2f} ms {100 * us / total:5.1f}% x{n:<4} "
                f"rank={rank} ~{gb:.2f} GB/call  {str(shapes[0])[:30]}"
            )

    print("\n  Rank by recoverable time, not by share. Highest first:")
    ranked = sorted(scored, key=lambda x: -(x[5] if x[5] is not None else -1))
    for family, rows, share, sp, n, rec in ranked:
        if share < 1.0:
            continue
        head = f"recoverable {rec:.1f}%" if rec is not None else "recoverable unknown -- measure it"
        print(f"\n  [{share:.1f}% of device time | {head}] {family} -> {rows[0][3]}")
        print(f"      {rows[0][4]}")
        for name, us, count, _, _ in sorted(rows, key=lambda r: -r[1])[:3]:
            print(f"      - {name[:62]:62} {us / 1000.0:7.2f} ms  x{count}")

    print("\n  Register spill is not visible here. For a kernel you wrote, check it with:")
    print("      unitrace -d -v -o prof python <script>.py")
    print("  The script must END with torch.xpu.current_stream().synchronize() --")
    print("  torch.xpu.synchronize() maps to zeDeviceSynchronize, which unitrace does not")
    print("  hook, and the per-kernel records are discarded before the exit flush.")


if __name__ == "__main__":
    main()
