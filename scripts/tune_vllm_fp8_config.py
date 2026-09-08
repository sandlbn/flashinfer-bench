"""Tune vLLM's block-scaled FP8 Triton GEMM for the current device.

`w8a8_triton_block_scaled_mm` picks its tile shape from a JSON file named for the device
and the (N, K) of the projection. With no such file it falls back to one hardcoded config
-- `BLOCK_SIZE_M=64`, `num_warps=4` -- and says so:

    Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!
    Config file not found at .../N=4096,K=2560,device_name=Intel(R)_Arc(TM)_B580_Graphics,...

`BLOCK_SIZE_M=64` at decode, where M is 1, computes a 64-row tile to keep one row. Every
Intel part is in that position today: vLLM ships configs for NVIDIA and AMD parts only.

This sweeps the tile space per (N, K, M) and writes the file vLLM looks for. It is the
cheapest real contribution available for a quantized model on a new device -- no kernel is
written, and the result is upstreamable as data.

Run it in the environment that has vLLM installed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# vLLM's own grid of batch sizes: a config is chosen by nearest M, so the grid only needs
# to be dense where the shape of the answer changes, which is at the small end.
DEFAULT_M_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def candidate_configs(m: int, block_n: int, block_k: int) -> List[Dict[str, int]]:
    """Configs worth measuring at this M.

    BLOCK_SIZE_N must divide by the weight block's N extent and BLOCK_SIZE_K by its K
    extent -- the kernel indexes scales in units of those blocks, so anything else reads
    the wrong scale. BLOCK_SIZE_M is not so constrained, and is the axis that matters:
    tiling 64 rows to serve one is the default's whole problem.
    """
    if m <= 8:
        block_ms = (16, 32)
    elif m <= 64:
        block_ms = (16, 32, 64)
    else:
        block_ms = (32, 64, 128)
    out = []
    for bm, bn, bk, gm, warps, stages in itertools.product(
        block_ms, (block_n, block_n * 2), (block_k,), (1, 16), (4, 8, 16), (2, 3)
    ):
        out.append(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": gm,
                "num_warps": warps,
                "num_stages": stages,
            }
        )
    return out


def _time_config(fn, iters: int, device: str) -> float:
    """Median-of-three wall time per call, in milliseconds.

    Median rather than mean: a single scheduling excursion on an Intel part is large
    enough to pick the wrong config outright.
    """
    from flashinfer_bench.device import get_accelerator

    accel = get_accelerator(device)
    samples = []
    for _ in range(3):
        for _ in range(3):
            fn()
        accel.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        accel.synchronize(device)
        samples.append((time.perf_counter() - t0) / iters * 1e3)
    return sorted(samples)[1]


def tune_shape(
    n: int, k: int, block_n: int, block_k: int, m_grid, device: str, iters: int
) -> Dict[str, Dict[str, int]]:
    from vllm.model_executor.layers.quantization.utils import fp8_utils as F

    best: Dict[str, Dict[str, int]] = {}
    b = (torch.randn(n, k, device=device, dtype=torch.bfloat16) / 8).to(torch.float8_e4m3fn)
    bs = torch.rand(-(-n // block_n), -(-k // block_k), device=device, dtype=torch.float32) * 0.1 + 0.05

    for m in m_grid:
        a = (torch.randn(m, k, device=device, dtype=torch.bfloat16) / 8).to(torch.float8_e4m3fn)
        a_s = torch.rand(m, -(-k // block_k), device=device, dtype=torch.float32) * 0.1 + 0.05
        winner, winner_ms = None, float("inf")
        for cfg in candidate_configs(m, block_n, block_k):
            def call(cfg=cfg):
                # Patch the config lookup rather than reimplementing the launch: the point
                # is to measure the kernel vLLM will actually run, launch logic included.
                return _run_with_config(F, a, b, a_s, bs, block_n, block_k, cfg)

            try:
                call()
            except Exception:
                continue  # a config the compiler or the hardware refuses is simply out
            ms = _time_config(call, iters, device)
            if ms < winner_ms:
                winner, winner_ms = cfg, ms
        if winner is None:
            print(f"  N={n} K={k} M={m}: no config ran", flush=True)
            continue
        best[str(m)] = winner
        print(
            f"  N={n} K={k} M={m:>5}: {winner_ms:8.4f} ms  "
            f"BM={winner['BLOCK_SIZE_M']:<4} BN={winner['BLOCK_SIZE_N']:<4} "
            f"GM={winner['GROUP_SIZE_M']:<3} warps={winner['num_warps']:<3} "
            f"stages={winner['num_stages']}",
            flush=True,
        )
    return best


def shapes_from_definitions(root: Path) -> List[Tuple[int, int, int, int]]:
    """Every `(N, K, block_n, block_k)` a dataset's block-scaled FP8 definitions describe.

    The definitions already carry all of it -- `N` and `K` as constant axes, the block
    extents as `N / N_blocks` and `K / K_blocks` -- so tuning a model is a matter of
    pointing at what the extractor wrote rather than transcribing five shapes by hand and
    getting one wrong.

    Deduplicated, because two projections of the same shape need one config file.
    """
    seen: Dict[Tuple[int, int, int, int], None] = {}
    for path in sorted(Path(root).rglob("*.json")):
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("op_type") != "gemm":
            continue
        if not any(str(t).startswith("quantization:block") for t in d.get("tags", ())):
            continue
        axes = d.get("axes", {})

        def const(name):
            a = axes.get(name)
            return int(a["value"]) if isinstance(a, dict) and "value" in a else None

        n, k, nb, kb = const("N"), const("K"), const("N_blocks"), const("K_blocks")
        if not (n and k and nb and kb):
            continue
        seen.setdefault((n, k, -(-n // nb), -(-k // kb)), None)
    return list(seen)


def _run_with_config(F, a, b, a_s, b_s, block_n, block_k, cfg):
    """Call the kernel forcing one config, by making the lookup return exactly it."""
    original = F.get_w8a8_block_fp8_configs
    F.get_w8a8_block_fp8_configs = lambda *args, **kwargs: {0: cfg}
    try:
        return F.w8a8_triton_block_scaled_mm(
            a, b, a_s, b_s, [block_n, block_k], torch.bfloat16
        )
    finally:
        F.get_w8a8_block_fp8_configs = original


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-definitions", type=Path, metavar="DIR",
        help="Read the shapes to tune from a dataset's block-scaled FP8 definitions "
             "(the directory `extract_model_kernels_xpu.py --output` wrote). This is the "
             "normal way to use this: the definitions already state every projection the "
             "model uses and its block size, so nothing has to be transcribed.",
    )
    ap.add_argument(
        "--shape", action="append", metavar="N,K", default=None,
        help="Tune this shape explicitly, e.g. 4096,2560. Repeatable. Use instead of "
             "--from-definitions when there is no dataset to hand.",
    )
    ap.add_argument("--block-n", type=int, default=128, help="Ignored with --from-definitions.")
    ap.add_argument("--block-k", type=int, default=128, help="Ignored with --from-definitions.")
    ap.add_argument("--device", default="xpu:0")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--m-grid", default=",".join(str(m) for m in DEFAULT_M_GRID))
    ap.add_argument(
        "--output", type=Path, default=Path("tmp/vllm-fp8-configs"),
        help="Directory for the JSON files (default: tmp/vllm-fp8-configs).",
    )
    ap.add_argument(
        "--install", action="store_true",
        help="Also copy into vLLM's own configs directory, which is the only place the "
             "kernel reads them from. Without this the files are written and unused.",
    )
    args = ap.parse_args()
    if not args.shape and not args.from_definitions:
        ap.error("give --from-definitions DIR, or --shape N,K one or more times")

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        get_w8a8_block_fp8_configs,  # noqa: F401  (import proves vLLM is present)
    )
    from vllm.utils.platform_utils import get_device_name_as_file_name

    device_name = get_device_name_as_file_name()
    m_grid = [int(x) for x in args.m_grid.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)

    targets: List[Tuple[int, int, int, int]] = []
    if args.from_definitions:
        targets.extend(shapes_from_definitions(args.from_definitions))
        if not targets:
            raise SystemExit(
                f"No block-scaled FP8 gemm definitions under {args.from_definitions}. "
                "Extract them first with scripts/extract_model_kernels_xpu.py."
            )
        print(f"{len(targets)} shape(s) from {args.from_definitions}", flush=True)
    for shape in args.shape or []:
        n, k = (int(x) for x in shape.split(","))
        targets.append((n, k, args.block_n, args.block_k))

    for n, k, block_n, block_k in targets:
        print(f"tuning N={n} K={k} block={block_n}x{block_k} on {device_name}", flush=True)
        best = tune_shape(n, k, block_n, block_k, m_grid, args.device, args.iters)
        name = (
            f"N={n},K={k},device_name={device_name},dtype=fp8_w8a8,"
            f"block_shape=[{block_n},{block_k}].json"
        )
        path = args.output / name
        path.write_text(json.dumps(best, indent=4) + "\n")
        print(f"  wrote {path}", flush=True)
        if args.install:
            import vllm.model_executor.layers.quantization.utils.fp8_utils as F

            dest = Path(F.__file__).parent / "configs" / name
            # Say what is being replaced, and with how much. A quick run over a reduced
            # --m-grid installs a config covering fewer batch sizes than the one already
            # there, and the kernel picks by nearest M -- so a narrow config silently
            # serves batch sizes it was never measured at. Easy to do by accident.
            if dest.exists():
                try:
                    existing = len(json.loads(dest.read_text()))
                except (OSError, json.JSONDecodeError):
                    existing = None
                if existing is not None and existing > len(best):
                    print(
                        f"  WARNING replacing a config with {existing} batch sizes by one "
                        f"with {len(best)}; the kernel picks by nearest M, so the rest are "
                        f"now served by a config never measured at them",
                        flush=True,
                    )
                else:
                    print(f"  replacing existing config ({existing} batch sizes)", flush=True)
            dest.write_text(path.read_text())
            print(f"  installed {dest}", flush=True)


if __name__ == "__main__":
    main()
