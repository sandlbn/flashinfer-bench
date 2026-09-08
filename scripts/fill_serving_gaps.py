"""Turn a serving run's `no-solution` reports into the definitions needed to close them.

`measure_serving_win.py` reports, per kernel family, the shapes vLLM asked for and got
nothing for -- `silu_and_mul no-solution d8192`. Those are not missing *kernels*: the
provider kernels and the in-tree SYCL/Triton templates are shape-parametric and already work
at any width. What is missing is the **definition**, and without one `add-baselines` has
nothing to match, so no kernel is ever sourced and none is ever optimized.

The extraction path cannot supply them. It hooks `nn.Module`s, and neither family is one:
SwiGLU is written inline inside an MLP's `forward`, and a fused add+norm spans two
statements, so a hook on the norm sees only its half. The serving dispatch counters are the
only place these shapes are observed, which is why this reads from there.

This clones the nearest sibling definition of the same family and reparametrizes it, then
emits workloads over the same batch sweep. Sourcing and optimizing follow as usual --
`add-baselines`, then `flashinfer-bench run`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# "silu_and_mul no-solution d8192" / "fused_add_rmsnorm no-solution h2560"
_GAP = re.compile(r"([a-z0-9_]+) no-solution ([a-z])(\d+)")

# The suffix letter a family names its width with, and the axis that width sets. Anything
# not listed is not safely reparametrizable by substitution and is reported, not guessed.
_FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}
"""Dtypes a definition may be reparametrized between. Quantized storage dtypes are not
here: changing those changes the operation, not its precision."""

_FAMILIES: Dict[str, Tuple[str, str]] = {
    "silu_and_mul": ("d", "d"),
    "gelu_and_mul": ("d", "d"),
    "rmsnorm": ("h", "hidden_size"),
    "fused_add_rmsnorm": ("h", "hidden_size"),
    "fused_add_rmsnorm_residual": ("h", "hidden_size"),
}


def gaps_from_details(details: List[str]) -> List[Tuple[str, str, int]]:
    """(family, letter, width) for every no-solution report, deduplicated."""
    found = []
    for text in details:
        for family, letter, width in _GAP.findall(text):
            entry = (family, letter, int(width))
            if entry not in found:
                found.append(entry)
    return found


def _details_from_json(path: Path) -> List[str]:
    data = json.loads(path.read_text())
    out: List[str] = []
    for run in data.get("runs", {}).get("ours", []) or [data.get("ours", {})]:
        out.extend(run.get("detail", []) or [])
    return out


def _sibling(defs_dir: Path, family: str, letter: str) -> Optional[Path]:
    """The existing definition of this family to clone. Widest wins.

    Widest, because a reference's constants are rewritten by substitution: cloning the
    widest sibling means the values being replaced are the least likely to collide with an
    unrelated literal in the source.
    """
    candidates = []
    for path in defs_dir.rglob(f"{family}_{letter}*.json"):
        m = re.fullmatch(rf"{re.escape(family)}_{letter}(\d+)", path.stem)
        if m:
            candidates.append((int(m.group(1)), path))
    return max(candidates)[1] if candidates else None


def _reparametrize(defn: Dict[str, Any], name: str, scale: float) -> Dict[str, Any]:
    """Scale every constant axis, then fix the reference's asserted values to match.

    Constants are rewritten only inside `assert <axis> == <int>` statements. Those are
    emitted in a fixed form by every definition in these families, and confining the
    substitution to them keeps it from touching an epsilon or an unrelated literal that
    happens to share the value.
    """
    out = json.loads(json.dumps(defn))
    out["name"] = name
    old_new: Dict[str, Tuple[int, int]] = {}
    for axis, spec in out["axes"].items():
        if spec.get("type") == "const" and isinstance(spec.get("value"), int):
            old = spec["value"]
            new = int(round(old * scale))
            spec["value"] = new
            old_new[axis] = (old, new)

    def fix(match: re.Match) -> str:
        axis, value = match.group(1), int(match.group(2))
        if axis in old_new and old_new[axis][0] == value:
            return f"assert {axis} == {old_new[axis][1]}"
        return match.group(0)

    out["reference"] = re.sub(r"assert (\w+) == (\d+)", fix, out["reference"])

    # Every constant the reference asserts must now be this definition's value. Checking
    # only that the old value is gone would pass a reference whose assertion never matched
    # the axis in the first place -- a definition and a reference describing different
    # shapes, which validates and benchmarks cleanly while computing the wrong thing.
    for axis, (_, new_value) in old_new.items():
        found = re.search(rf"assert {axis} == (\d+)", out["reference"])
        if found and int(found.group(1)) != new_value:
            raise ValueError(
                f"{name}: reference asserts {axis} == {found.group(1)}, but the axis is "
                f"{new_value}; it cannot be reparametrized by substitution"
            )

    # The description names the width in prose; leaving the sibling's there would label
    # the file with a size it does not have.
    desc = out.get("description", "")
    for old, new in old_new.values():
        desc = re.sub(rf"\b{old}\b", str(new), desc)
    out["description"] = desc

    # `model:` tags describe where the sibling was observed, which says nothing about this
    # width. status:verified is likewise not inherited -- nothing has validated this yet.
    out["tags"] = [
        t for t in out.get("tags", []) if not t.startswith("model:") and not t.startswith("status:")
    ]
    out["tags"].append("status:unverified")
    return out


_EPS = re.compile(r"^(\s*EPS\s*=\s*)([0-9.eE+-]+)\s*$", re.MULTILINE)


def _reference_eps(reference: str) -> Optional[str]:
    m = _EPS.search(reference)
    return m.group(2) if m else None


def _set_eps(reference: str, eps: str) -> str:
    return _EPS.sub(lambda m: f"{m.group(1)}{eps}", reference)


def _eps_slug(eps: str) -> str:
    return f"{float(eps):.0e}".replace("-0", "-").replace("e-", "e")


def _existing_at_other_eps(defs_dir: Path, name: str, eps: str) -> Optional[str]:
    """The name of a same-width definition whose reference uses a different epsilon."""
    for path in defs_dir.rglob(f"{name}.json"):
        other = _reference_eps(json.loads(path.read_text())["reference"])
        if other is not None and float(other) != float(eps):
            return path.stem
    return None


def _workloads(defn: Dict[str, Any], sibling_workloads: Path) -> List[Dict[str, Any]]:
    """The sibling's batch sweep, carrying only the variable axes.

    Constant axes are the definition's, not the workload's. Some existing workloads repeat
    them and the validator flags that as `extra axes`, so copying the sibling verbatim would
    propagate a defect into every file generated from it.
    """
    var_axes = {a for a, spec in defn["axes"].items() if spec.get("type") != "const"}
    rows = []
    for line in sibling_workloads.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["definition"] = defn["name"]
        axes = {a: v for a, v in row["workload"]["axes"].items() if a in var_axes}
        row["workload"]["axes"] = axes
        batch = axes.get("batch_size")
        row["workload"]["uuid"] = f"{defn['name']}-b{batch}"
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("tmp/flashinfer-trace"))
    ap.add_argument(
        "--from-json",
        type=Path,
        action="append",
        default=[],
        help="A measure_serving_win.py --json result to read gaps from.",
    )
    ap.add_argument(
        "--gaps",
        action="append",
        default=[],
        help="Explicit gap, e.g. 'silu_and_mul:d8192'. Repeatable.",
    )
    ap.add_argument(
        "--dtype",
        default=None,
        help="Tensor dtype this model runs in (float16, bfloat16, ...). Like epsilon, it is "
        "never safely inherited from a sibling: a definition whose dtype differs from the "
        "model's is rejected at dispatch and every call silently falls back, reported as "
        "no-solution. Read it from the model config's torch_dtype.",
    )
    ap.add_argument(
        "--eps",
        default=None,
        help="Epsilon for norm families. Models differ (1e-5 and 1e-6 both occur), so it "
        "is never safely inherited from a sibling: pass the value observed on the model "
        "this width came from.",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Write the files. Without it, report what would be written.",
    )
    args = ap.parse_args()

    details = []
    for path in args.from_json:
        details.extend(_details_from_json(path))
    for spec in args.gaps:
        family, _, shape = spec.partition(":")
        details.append(f"{family} no-solution {shape}")

    wanted = gaps_from_details(details)
    if not wanted:
        raise SystemExit("No no-solution gaps found. Nothing to do.")

    defs_dir = args.dataset / "definitions"
    made, skipped = [], []
    for family, letter, width in wanted:
        name = f"{family}_{letter}{width}"
        if family not in _FAMILIES:
            skipped.append(f"{name}: family not reparametrizable by width; extract it instead")
            continue
        # The existence check happens after the epsilon suffix is resolved, below: with
        # --eps the final name may gain a suffix, so a check on the bare name here would
        # either skip a definition that still needs writing or -- as it did -- let an
        # existing one through to be overwritten.
        sibling = _sibling(defs_dir, family, letter)
        if sibling is None:
            skipped.append(f"{name}: no sibling of this family to clone")
            continue

        base = json.loads(sibling.read_text())
        base_width = int(re.fullmatch(rf"{re.escape(family)}_{letter}(\d+)", sibling.stem).group(1))
        try:
            defn = _reparametrize(base, name, width / base_width)
        except ValueError as exc:
            skipped.append(str(exc))
            continue

        inherited = _reference_eps(defn["reference"])
        if inherited is not None:
            if args.eps:
                defn["reference"] = _set_eps(defn["reference"], args.eps)
                # `{family}_h{H}` does not encode epsilon, so two models that share a hidden
                # size but not their eps collide on one name -- and the one that loses is
                # served a reference computing a slightly different function. Suffix the
                # name when this width already exists at another eps, following the
                # dataset's own precedent for a distinguishing attribute.
                clash = _existing_at_other_eps(defs_dir, name, args.eps)
                if clash:
                    name = f"{name}_eps{_eps_slug(args.eps)}"
                    defn["name"] = name
                    print(
                        f"  note  {clash} exists with a different epsilon; naming this "
                        f"one {name} so they do not collide"
                    )
            else:
                skipped.append(
                    f"{name}: reference has EPS={inherited} inherited from {sibling.stem}, "
                    "which is a different model. Pass --eps with the value this model uses "
                    "(read it off the module, or from config.json's rms_norm_eps)."
                )
                continue

        # Never overwrite. An existing definition may be status:verified and carry real
        # collected workloads; replacing it with a reparametrized clone of a sibling would
        # silently downgrade both, and the only signal would be a line saying "wrote".
        if list(defs_dir.rglob(f"{name}.json")):
            skipped.append(f"{name}: already exists; refusing to overwrite it")
            continue

        # Same trap as epsilon: a width alone does not identify the operation. A
        # bfloat16 definition presented with float16 activations is refused by the dtype
        # guard in apply(), so it never substitutes -- and the counters say "no-solution",
        # which reads as "not extracted yet" rather than "extracted in the wrong dtype".
        sibling_dtypes = {
            spec.get("dtype")
            for spec in list(base["inputs"].values()) + list(base["outputs"].values())
            if spec.get("dtype")
        }
        floats = {d for d in sibling_dtypes if d in _FLOAT_DTYPES}
        if floats:
            if not args.dtype:
                skipped.append(
                    f"{name}: sibling {sibling.stem} is {'/'.join(sorted(floats))}; pass "
                    "--dtype with the dtype this model runs in (config.json torch_dtype), "
                    "or the definition will never match at dispatch."
                )
                continue
            if args.dtype not in floats:
                name = f"{name}_{args.dtype}"
                defn["name"] = name
                print(
                    f"  note  naming this one {name} so it does not collide with the "
                    f"{'/'.join(sorted(floats))} definition at the same width"
                )
            for spec in list(defn["inputs"].values()) + list(defn["outputs"].values()):
                if spec.get("dtype") in _FLOAT_DTYPES:
                    spec["dtype"] = args.dtype

        out_def = sibling.with_name(f"{name}.json")
        wl_src = args.dataset / "workloads" / defn["op_type"] / f"{sibling.stem}.jsonl"
        out_wl = wl_src.with_name(f"{name}.jsonl")
        rows = _workloads(defn, wl_src) if wl_src.exists() else []

        if args.write:
            out_def.write_text(json.dumps(defn, indent=2) + "\n")
            if rows:
                out_wl.write_text("".join(json.dumps(r) + "\n" for r in rows))
        made.append((name, sibling.stem, len(rows), out_def))

    for name, src, n_wl, path in made:
        verb = "wrote" if args.write else "would write"
        print(f"  {verb} {name}  (from {src}, {n_wl} workloads) -> {path}")
    for line in skipped:
        print(f"  skip  {line}")

    if made and args.write:
        # `validate-references` and `add-baselines` take one comma-separated string;
        # `run` takes space-separated names. Passing commas to `run` makes it look for a
        # single definition with commas in its name, which it reports as "not found" and
        # then exits 0 -- so the benchmark silently measures nothing.
        csv = ",".join(n for n, *_ in made)
        ssv = " ".join(n for n, *_ in made)
        print("\nNow source and optimize them:")
        print(
            f"  flashinfer-bench validate-references --local {args.dataset} "
            f"--device xpu:0 --definitions {csv}"
        )
        print(
            f"  flashinfer-bench add-baselines --local {args.dataset} --in-tree --definitions {csv}"
        )
        print(
            f"  flashinfer-bench add-baselines --local {args.dataset} "
            f"--providers vllm-xpu,sgl-kernel-xpu --definitions {csv}"
        )
        print(f"  flashinfer-bench run --local {args.dataset} --definitions {ssv} --save-results")
    elif made:
        print("\nRe-run with --write to create them.")


if __name__ == "__main__":
    main()
