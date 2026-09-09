"""End-to-end Intel kernel optimization: profile a model, extract, optimize, verify.

Runs the whole loop for one model on one Intel GPU:

    1. profile   run the model under unitrace and rank kernels by device time
    2. extract   emit Definitions + Workloads for the shapes it actually ran
    3. baseline  add the upstream vLLM / SGLang kernels as competitors
    4. verify    cross-validate each reference on the device before trusting it
    5. optimize  search a tuning space, benchmarking every candidate, keep the best
    6. report    what won, by how much, against whom

Step 5 is a *search* over the parameters that matter on Intel -- work-group size and
sub-group width -- rather than an LLM writing new kernel bodies. It needs no API key and
runs unattended. The agentic variant (``examples/kernel_generator``) plugs into the same
place once credentials exist; what it changes is how candidates are produced, not how they
are judged.

Every candidate goes through the ordinary benchmark, so correctness gates performance:
a candidate that is fast and wrong never records a latency.

Usage
-----
    python scripts/optimize_model_kernels_xpu.py \\
        --model <org>/<model> --output ./run --definition <definition>
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("optimize-intel")

# Work-group sizes worth trying on Intel. Must be a multiple of the sub-group width and
# within the device's max_work_group_size; both are checked against the capability record.
WORK_GROUP_SIZES = (64, 128, 256, 512, 1024)

# Intel GPUs support 16 and 32 wide sub-groups. Pinning it matters: left unspecified the
# compiler picks, and the choice interacts with how a row-wise reduction schedules.
SUB_GROUP_SIZES = (16, 32)


@dataclass
class Candidate:
    """One point in the tuning space."""

    work_group: int
    sub_group: int

    @property
    def name(self) -> str:
        return f"wg{self.work_group}_sg{self.sub_group}"


def _rmsnorm_source(candidate: Candidate) -> str:
    """A SYCL RMSNorm specialised to one tuning configuration.

    The kernel body is fixed; only the launch geometry varies. That keeps the search
    honest -- differences in the result come from the configuration, not from a rewritten
    algorithm.
    """
    return f"""#include <sycl/sycl.hpp>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace fib_tuned {{

namespace {{
constexpr size_t kWorkGroupSize = {candidate.work_group};
constexpr int kSubGroupSize = {candidate.sub_group};

template <typename T>
void Launch(sycl::queue* q, const T* x, const T* w, T* out, int64_t rows, int64_t hidden,
            float eps) {{
  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  q->parallel_for(
      sycl::nd_range<1>(static_cast<size_t>(rows) * kWorkGroupSize, kWorkGroupSize),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSubGroupSize)]] {{
        const int64_t row = static_cast<int64_t>(item.get_group(0));
        const size_t lid = item.get_local_id(0);
        const size_t lsize = item.get_local_range(0);
        const T* row_in = x + row * hidden;
        T* row_out = out + row * hidden;

        float partial = 0.0f;
        for (int64_t i = static_cast<int64_t>(lid); i < hidden;
             i += static_cast<int64_t>(lsize)) {{
          const float v = static_cast<float>(row_in[i]);
          partial += v * v;
        }}
        const float sum_sq =
            sycl::reduce_over_group(item.get_group(), partial, sycl::plus<float>());
        const float scale = sycl::rsqrt(sum_sq * inv_hidden + eps);
        for (int64_t i = static_cast<int64_t>(lid); i < hidden;
             i += static_cast<int64_t>(lsize)) {{
          row_out[i] = static_cast<T>(static_cast<float>(row_in[i]) * scale *
                                      static_cast<float>(w[i]));
        }}
      }});
}}
}}  // namespace

void RMSNormTuned(tvm::ffi::TensorView x, tvm::ffi::TensorView weight, double eps,
                  tvm::ffi::TensorView out) {{
  TVM_FFI_ICHECK_EQ(x.ndim(), 2) << "x must be [rows, hidden]";
  const int64_t rows = x.size(0);
  const int64_t hidden = x.size(1);
  if (rows == 0 || hidden == 0) return;

  DLDevice dev = x.device();
  sycl::queue* q = static_cast<sycl::queue*>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";

  const DLDataType dtype = x.dtype();
  const float eps_f = static_cast<float>(eps);
  if (dtype.code == kDLFloat && dtype.bits == 16) {{
    Launch<sycl::half>(q, static_cast<const sycl::half*>(x.data_ptr()),
                       static_cast<const sycl::half*>(weight.data_ptr()),
                       static_cast<sycl::half*>(out.data_ptr()), rows, hidden, eps_f);
  }} else if (dtype.code == kDLFloat && dtype.bits == 32) {{
    Launch<float>(q, static_cast<const float*>(x.data_ptr()),
                  static_cast<const float*>(weight.data_ptr()),
                  static_cast<float*>(out.data_ptr()), rows, hidden, eps_f);
  }} else {{
    TVM_FFI_LOG_AND_THROW(RuntimeError) << "rmsnorm_tuned: unsupported dtype";
  }}
}}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm_tuned, RMSNormTuned);

}}  // namespace fib_tuned
"""


def find_unitrace() -> Optional[str]:
    """Locate the unitrace binary, including a local pti-gpu build."""
    found = shutil.which("unitrace")
    if found:
        return found
    local = Path("tmp/pti-gpu/tools/unitrace/build/unitrace")
    return str(local.resolve()) if local.exists() else None


def profile_model(
    model: str, prompt: str, max_new_tokens: int, top: int
) -> List[Tuple[str, float]]:
    """Rank kernels by share of device time, using unitrace.

    Returns ``[(kernel_name, percent)]``. Empty when unitrace is unavailable -- profiling
    informs which kernel to work on, so its absence degrades the run rather than ending it.
    """
    unitrace = find_unitrace()
    if unitrace is None:
        logger.warning(
            "unitrace not found; skipping profile (target must be given with --definition)"
        )
        return []

    script = Path("_profile_target.py")
    script.write_text(
        "import torch\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        f"name = {model!r}\n"
        "tok = AutoTokenizer.from_pretrained(name)\n"
        "m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to('xpu:0').eval()\n"
        f"text = tok.apply_chat_template([{{'role':'user','content':{prompt!r}}}],"
        " add_generation_prompt=True, tokenize=False)\n"
        "enc = tok(text, return_tensors='pt').to('xpu:0')\n"
        "with torch.no_grad():\n"
        f"    m.generate(**enc, max_new_tokens={max_new_tokens}, do_sample=False)\n"
        "    torch.xpu.current_stream().synchronize()\n"  # hooked by unitrace;
        # torch.xpu.synchronize() maps to zeDeviceSynchronize, which it does not hook,
        # and the per-kernel records are discarded before the exit flush.
    )
    try:
        result = subprocess.run(
            [unitrace, "--device-timing", sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except Exception as e:
        logger.warning(f"unitrace run failed ({e}); skipping profile")
        return []
    finally:
        script.unlink(missing_ok=True)

    ranked: List[Tuple[str, float]] = []
    for line in (result.stderr or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not parts[0].startswith('"'):
            continue
        try:
            ranked.append((parts[0].strip('"'), float(parts[3])))
        except ValueError:
            continue
    ranked.sort(key=lambda kv: -kv[1])
    return ranked[:top]


def _tuned_solution(definition_name: str, candidate: Candidate) -> Dict[str, Any]:
    return {
        "name": f"{definition_name}__tuned_{candidate.name}",
        "definition": definition_name,
        "author": "autotune",
        "spec": {
            "language": "sycl",
            "target_hardware": ["xpu"],
            "entry_point": "rmsnorm_tuned.cpp::rmsnorm_tuned",
            "destination_passing_style": True,
        },
        "sources": [{"path": "rmsnorm_tuned.cpp", "content": _rmsnorm_source(candidate)}],
        "description": (
            f"Autotuned SYCL RMSNorm: work-group {candidate.work_group}, "
            f"sub-group {candidate.sub_group}."
        ),
    }


def viable_candidates(device: str) -> List[Candidate]:
    """Tuning points this device can actually launch.

    Filtered against the driver's reported limits rather than tried and failed: a
    work-group larger than ``max_work_group_size``, or a sub-group width the device does
    not support, fails at launch and would pollute the search with noise.
    """
    from flashinfer_bench.device import get_accelerator

    caps = get_accelerator(device).capabilities(device)
    max_wg = int(caps.extra.get("max_work_group_size", 1024) or 1024)
    supported_sg = {int(s) for s in (caps.extra.get("sub_group_sizes") or SUB_GROUP_SIZES)}

    candidates = [
        Candidate(wg, sg)
        for wg in WORK_GROUP_SIZES
        for sg in SUB_GROUP_SIZES
        if wg <= max_wg and sg in supported_sg and wg % sg == 0
    ]
    logger.info(
        f"Tuning space: {len(candidates)} candidate(s) "
        f"(max_work_group_size={max_wg}, sub_group_sizes={sorted(supported_sg)})"
    )
    return candidates


def _import_from_dataset(dataset: Path, root: Path, name: str) -> None:
    """Copy one definition and its workloads out of the dataset into the working dir.

    Blobs come too when the workloads reference them; a workload whose inputs are
    ``random`` needs none.
    """
    found = next((p for p in (dataset / "definitions").rglob(f"{name}.json")), None)
    if found is None:
        raise SystemExit(f"{name!r} is not in {dataset}")
    op_type = found.parent.name
    out_def = root / "definitions" / op_type
    out_def.mkdir(parents=True, exist_ok=True)
    shutil.copy2(found, out_def / found.name)

    workloads = dataset / "workloads" / op_type / f"{name}.jsonl"
    if workloads.exists():
        out_wl = root / "workloads" / op_type
        out_wl.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workloads, out_wl / workloads.name)
    blobs = dataset / "blob" / "workloads" / op_type / name
    if blobs.is_dir():
        shutil.copytree(blobs, root / "blob" / "workloads" / op_type / name, dirs_exist_ok=True)

    # Bring the solutions that already exist for it. Without them the tuning candidates are
    # ranked only against the upstream baselines this script generates, so the report can
    # conclude "does not beat upstream" while the definition's own best kernel -- which may
    # beat upstream comfortably -- was never in the field.
    imported = 0
    for path in (dataset / "solutions").rglob("*.json"):
        rel = path.relative_to(dataset / "solutions")
        if name not in path.stem and name not in rel.parts:
            continue
        target_path = root / "solutions" / rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target_path)
        imported += 1
    logger.info(f"Imported {name} from {dataset} (op_type={op_type}, {imported} solution(s))")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HuggingFace repo id or local path.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--definition", default=None, help="Definition to optimize.")
    parser.add_argument("--prompt", default="Write a short poem about silicon.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument(
        "--dataset",
        default="tmp/flashinfer-trace",
        help="Dataset to take --definition from when the extractor does not emit it. The "
        "extractor hooks nn.Modules, so fused and inline families never appear in its "
        "output and could not otherwise be optimized.",
    )
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from flashinfer_bench.bench import Benchmark, BenchmarkConfig
    from flashinfer_bench.data import TraceSet
    from flashinfer_bench.integration import make_baseline_solutions

    root: Path = args.output

    # ---- 1. profile -------------------------------------------------------------
    if not args.skip_profile:
        logger.info("\n=== 1. Profile with unitrace ===")
        for name, pct in profile_model(args.model, args.prompt, args.max_new_tokens, top=6):
            logger.info(f"  {pct:6.2f}%  {name[:96]}")

    # ---- 2. extract -------------------------------------------------------------
    logger.info("\n=== 2. Extract definitions from the model ===")
    subprocess.run(
        [
            sys.executable,
            "scripts/extract_model_kernels_xpu.py",
            "--model",
            args.model,
            "--output",
            str(root),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ],
        check=True,
    )

    trace_set = TraceSet.from_path(str(root))
    target = args.definition or next(
        (n for n in sorted(trace_set.definitions) if n.startswith("rmsnorm")), None
    )
    if target is not None and target not in trace_set.definitions:
        # The extractor hooks nn.Modules, so whole families never appear in its output --
        # a fused add+norm spans two statements and a gated activation is written inline
        # inside an MLP's forward. Refusing to optimize those made the loop unable to reach
        # the families that actually beat the provider. Take the definition from the dataset
        # instead; everything downstream only needs it present in the working directory.
        _import_from_dataset(Path(args.dataset), root, target)
        trace_set = TraceSet.from_path(str(root))
    if target is None or target not in trace_set.definitions:
        raise SystemExit(
            f"No optimizable definition found (looked for {target!r}); it is neither in the "
            f"extraction output nor in {args.dataset}"
        )
    definition = trace_set.definitions[target]
    logger.info(f"Optimizing: {target}")

    # ---- 3. baselines -----------------------------------------------------------
    logger.info("\n=== 3. Add upstream baselines ===")
    baseline_dir = root / "solutions" / "baseline" / definition.op_type
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baselines = make_baseline_solutions(definition)
    for solution in baselines:
        (baseline_dir / f"{solution.name}.json").write_text(
            solution.model_dump_json(indent=2) + "\n"
        )
        logger.info(f"  baseline: {solution.name}")
    if not baselines:
        logger.info("  none available for this definition")

    # ---- 4. candidates ----------------------------------------------------------
    logger.info("\n=== 4. Generate tuning candidates ===")
    tuned_dir = root / "solutions" / "autotune" / definition.op_type
    tuned_dir.mkdir(parents=True, exist_ok=True)
    for candidate in viable_candidates(args.device):
        payload = _tuned_solution(target, candidate)
        (tuned_dir / f"{payload['name']}.json").write_text(json.dumps(payload, indent=2) + "\n")

    # ---- 5. benchmark the whole field -------------------------------------------
    logger.info("\n=== 5. Benchmark every candidate against the baselines ===")
    trace_set = TraceSet.from_path(str(root))
    config = BenchmarkConfig.default(num_trials=args.num_trials, definitions=[target])
    benchmark = Benchmark(trace_set, config)
    try:
        result = benchmark.run_all(dump_traces=True)
    finally:
        benchmark.close()

    # ---- 6. report --------------------------------------------------------------
    logger.info("\n=== 6. Result ===")
    rows: List[Tuple[str, float, str]] = []
    for trace in result.traces.get(target, []):
        ev = trace.evaluation
        if ev and ev.performance and trace.solution:
            author = "upstream" if "vllm" in trace.solution or "sgl" in trace.solution else "tuned"
            rows.append((trace.solution, ev.performance.speedup_factor, author))
    rows.sort(key=lambda r: -r[1])

    if not rows:
        logger.warning("No candidate passed correctness; nothing to report.")
        raise SystemExit(1)

    best_upstream = max((s for _, s, a in rows if a == "upstream"), default=None)
    logger.info(f"{'solution':<52}{'speedup':>9}  source")
    for name, speedup, author in rows:
        logger.info(f"{name[:51]:<52}{speedup:>8.2f}x  {author}")

    winner, best, source = rows[0]
    logger.info(f"\nBest: {winner} at {best:.2f}x ({source})")
    if best_upstream is not None:
        if best > best_upstream:
            logger.info(
                f"Beats the best upstream kernel ({best_upstream:.2f}x) by "
                f"{best / best_upstream:.2f}x."
            )
        else:
            logger.info(
                f"Does NOT beat the best upstream kernel ({best_upstream:.2f}x). "
                "The tuning space was exhausted without a win -- widen it, or change the "
                "algorithm rather than its launch geometry."
            )


if __name__ == "__main__":
    main()
