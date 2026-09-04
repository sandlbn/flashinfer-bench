"""Extract kernel Definitions and Workloads from a HuggingFace model running on Intel GPU.

The onboarding skills harvest definitions from an SGLang run through FlashInfer's tracing
hooks, which is CUDA-only. This does the equivalent for Intel directly from a HuggingFace
model: run a short generation, observe the modules that actually execute, and emit the
Definitions and Workloads for the shapes that were really used.

Definitions produced here are ordinary, hardware-agnostic FlashInfer-Trace definitions --
the same artifacts the CUDA path produces. What makes this useful on Intel is only that it
needs no CUDA-only tooling to obtain them.

Usage
-----
    python scripts/extract_model_kernels_xpu.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --output ./qwen-intel-trace \\
        --prompt "Write a short poem." --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("extract-model-kernels")

RMSNORM_REFERENCE = (
    "import torch\n\n\n"
    "def run(x, weight, eps):\n"
    "    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight\n"
)

LINEAR_REFERENCE = (
    "import torch\n\n\n"
    "def run(x, weight):\n"
    "    return torch.nn.functional.linear(x, weight)\n"
)

_DTYPE_NAMES = {
    "torch.float32": "float32",
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
}


def _dtype_name(dtype: Any) -> str:
    name = _DTYPE_NAMES.get(str(dtype))
    if name is None:
        raise ValueError(f"Unsupported dtype for extraction: {dtype}")
    return name


class ShapeRecorder:
    """Records the shapes each module was actually called with.

    Counting calls matters as much as collecting shapes: a kernel invoked once per layer
    per token is worth optimizing, and one invoked twice in total is not.
    """

    def __init__(self) -> None:
        self.rmsnorm: Counter = Counter()
        self.linear: Counter = Counter()
        self._handles: List[Any] = []

    def attach(self, model: Any) -> None:
        import torch

        for module in model.modules():
            cls_name = type(module).__name__
            if "RMSNorm" in cls_name and hasattr(module, "weight"):
                self._handles.append(module.register_forward_hook(self._make_rmsnorm_hook(module)))
            elif isinstance(module, torch.nn.Linear) and module.bias is None:
                self._handles.append(module.register_forward_hook(self._make_linear_hook(module)))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_rmsnorm_hook(self, module: Any):
        def hook(_mod: Any, inputs: Tuple[Any, ...], _output: Any) -> None:
            x = inputs[0]
            if x.dim() < 2:
                return
            tokens = int(x.numel() // x.shape[-1])
            self.rmsnorm[(tokens, int(x.shape[-1]), _dtype_name(x.dtype))] += 1

        return hook

    def _make_linear_hook(self, module: Any):
        def hook(_mod: Any, inputs: Tuple[Any, ...], _output: Any) -> None:
            x = inputs[0]
            if x.dim() < 2:
                return
            tokens = int(x.numel() // x.shape[-1])
            self.linear[
                (tokens, int(module.in_features), int(module.out_features), _dtype_name(x.dtype))
            ] += 1

        return hook


def _rmsnorm_definition(hidden: int, dtype: str, model: str) -> Dict[str, Any]:
    return {
        "name": f"rmsnorm_h{hidden}_{dtype}",
        "description": f"RMSNorm over the last dimension, hidden {hidden}. Observed in {model}.",
        "op_type": "rmsnorm",
        "tags": ["stage:extracted", f"model:{model}"],
        "axes": {"tokens": {"type": "var"}, "hidden": {"type": "const", "value": hidden}},
        "inputs": {
            "x": {"shape": ["tokens", "hidden"], "dtype": dtype},
            "weight": {"shape": ["hidden"], "dtype": dtype},
            "eps": {"shape": None, "dtype": "float32"},
        },
        "outputs": {"out": {"shape": ["tokens", "hidden"], "dtype": dtype}},
        "reference": RMSNORM_REFERENCE,
    }


def _linear_definition(in_f: int, out_f: int, dtype: str, model: str) -> Dict[str, Any]:
    return {
        "name": f"linear_k{in_f}_n{out_f}_{dtype}",
        "description": (
            f"Bias-free linear projection, {in_f} -> {out_f}. Observed in {model}. "
            "GEMM dominates decoder time, so this is the highest-value optimization target."
        ),
        "op_type": "gemm",
        "tags": ["stage:extracted", f"model:{model}"],
        "axes": {
            "tokens": {"type": "var"},
            "in_features": {"type": "const", "value": in_f},
            "out_features": {"type": "const", "value": out_f},
        },
        "inputs": {
            "x": {"shape": ["tokens", "in_features"], "dtype": dtype},
            "weight": {"shape": ["out_features", "in_features"], "dtype": dtype},
        },
        "outputs": {"out": {"shape": ["tokens", "out_features"], "dtype": dtype}},
        "reference": LINEAR_REFERENCE,
    }


def _write(root: Path, definition: Dict[str, Any], workloads: List[Dict[str, Any]]) -> None:
    op_type = definition["op_type"]
    (root / "definitions" / op_type).mkdir(parents=True, exist_ok=True)
    (root / "workloads" / op_type).mkdir(parents=True, exist_ok=True)
    (root / "definitions" / op_type / f"{definition['name']}.json").write_text(
        json.dumps(definition, indent=2) + "\n"
    )
    (root / "workloads" / op_type / f"{definition['name']}.jsonl").write_text(
        "\n".join(json.dumps(w) for w in workloads) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Write a short poem about silicon.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default=None, help="Default: first available accelerator.")
    parser.add_argument(
        "--top-linear",
        type=int,
        default=3,
        help="Emit definitions for the N most-called linear shapes.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from flashinfer_bench.device import list_devices

    device = args.device or (list_devices() or ["cpu"])[0]
    logger.info(f"Loading {args.model} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model = model.to(device).eval()

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(text, return_tensors="pt").to(device)

    recorder = ShapeRecorder()
    recorder.attach(model)
    with torch.no_grad():
        model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False)
    recorder.detach()

    logger.info(
        f"Observed {sum(recorder.rmsnorm.values())} RMSNorm and "
        f"{sum(recorder.linear.values())} linear calls"
    )

    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    written = 0

    # RMSNorm: one definition per hidden size, one workload per observed token count.
    by_hidden: Dict[Tuple[int, str], List[int]] = {}
    for (tokens, hidden, dtype), _count in recorder.rmsnorm.items():
        by_hidden.setdefault((hidden, dtype), []).append(tokens)

    for (hidden, dtype), token_counts in sorted(by_hidden.items()):
        definition = _rmsnorm_definition(hidden, dtype, args.model)
        workloads = [
            {
                "definition": definition["name"],
                "workload": {
                    "axes": {"tokens": t, "hidden": hidden},
                    "inputs": {
                        "x": {"type": "random"},
                        "weight": {"type": "random"},
                        "eps": {"type": "scalar", "value": 1e-6},
                    },
                    "uuid": f"{definition['name']}-t{t}",
                },
            }
            for t in sorted(set(token_counts))
        ]
        _write(root, definition, workloads)
        logger.info(f"  {definition['name']}: {len(workloads)} workload(s)")
        written += 1

    # Linear: the most-called shapes carry the decoder's GEMM time.
    hottest = recorder.linear.most_common(args.top_linear)
    by_shape: Dict[Tuple[int, int, str], List[int]] = {}
    for (tokens, in_f, out_f, dtype), _count in hottest:
        by_shape.setdefault((in_f, out_f, dtype), []).append(tokens)

    for (in_f, out_f, dtype), token_counts in sorted(by_shape.items()):
        definition = _linear_definition(in_f, out_f, dtype, args.model)
        workloads = [
            {
                "definition": definition["name"],
                "workload": {
                    "axes": {"tokens": t, "in_features": in_f, "out_features": out_f},
                    "inputs": {"x": {"type": "random"}, "weight": {"type": "random"}},
                    "uuid": f"{definition['name']}-t{t}",
                },
            }
            for t in sorted(set(token_counts))
        ]
        _write(root, definition, workloads)
        logger.info(f"  {definition['name']}: {len(workloads)} workload(s)")
        written += 1

    logger.info(f"Wrote {written} definition(s) to {root}")
    logger.info(
        "Next: flashinfer-bench validate-references --local "
        f"{root} --device {device}, then add-baselines and run."
    )


if __name__ == "__main__":
    main()
