"""Loop a kernel harness for a fixed wall time so a sampling profiler can attribute.

A harness file exposes ``Model``, ``get_inputs`` and ``get_init_inputs`` (the contract
``tools/kernel-harness`` and ``scripts/kernel_trials.py`` use). A tracing profiler records
every launch, but a sampling one attributes each sample to whatever kernel is running when
it fires, so a short kernel has to run for a while before it collects any.

Invocation:
    python -m flashinfer_bench.agents._harness_runner path/to/harness.py --seconds 5 --device xpu:0

Not a timer: the printed call count is for checking the loop ran, not for measuring.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys
import time

import torch

from flashinfer_bench.device import device_synchronize, parse_device


def _sync(device: str) -> None:
    dev_type, _ = parse_device(device)
    if dev_type == "xpu":
        # The stream sync is the one Level Zero tracers hook; the device-wide one is not.
        torch.xpu.current_stream().synchronize()
    else:
        device_synchronize(device)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Loop a kernel harness under a profiler")
    parser.add_argument("harness")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch", type=int, default=100, help="launches between syncs")
    args = parser.parse_args(argv)

    path = pathlib.Path(args.harness)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        print(f"cannot import {path}", file=sys.stderr)
        return 2
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dev_type, _ = parse_device(args.device)
    if dev_type != "cpu":
        from flashinfer_bench.device import get_accelerator

        get_accelerator(args.device).set_device(args.device)
    model = module.Model(*module.get_init_inputs())
    inputs = module.get_inputs()
    print(f"harness {path.name} on {args.device} pid {os.getpid()}", flush=True)

    calls = 0
    with torch.no_grad():
        for _ in range(args.warmup):
            model(*inputs)
        _sync(args.device)
        start = time.perf_counter()
        while time.perf_counter() - start < args.seconds:
            for _ in range(args.batch):
                model(*inputs)
            _sync(args.device)
            calls += args.batch
    print(f"{calls} calls in {time.perf_counter() - start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
