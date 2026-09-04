"""unitrace profiling for Intel GPUs, the counterpart of :mod:`flashinfer_bench.agents.ncu`.

Nsight Compute has no Intel equivalent, but Intel's ``unitrace`` (from intel/pti-gpu)
covers the same ground for kernel work: per-kernel device time, launch parameters, and
host/device transfer accounting, gathered through Level Zero and PTI.

Like the NCU tool, this profiles one Solution on one Workload by running the shared
solution runner under the profiler, so what is measured is exactly what the benchmark
measures.

unitrace is not a dependency and is not on PyPI. Build it from source:

.. code-block:: bash

    git clone https://github.com/intel/pti-gpu.git
    cd pti-gpu/tools/unitrace && mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j
    export PATH="$PWD:$PATH"
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from flashinfer_bench.data import Solution, Workload

logger = logging.getLogger(__name__)

DEFAULT_MODES = ("device-timing",)
"""Modes used when the caller does not ask for anything specific.

``device-timing`` is the summary that matters for kernel work: time on the device per
kernel, which is what a solution is trying to reduce.
"""

VALID_MODES = frozenset(
    {
        "device-timing",
        "host-timing",
        "kernel-submission",
        "device-timeline",
        "chrome-kernel-logging",
        "chrome-device-logging",
        "opencl",
        "level-zero",
    }
)
"""unitrace modes this wrapper will pass through.

Restricted on purpose: unitrace accepts many flags, and an agent passing an arbitrary
string through to a subprocess is not something to allow casually.
"""


def is_unitrace_available(unitrace_path: str = "unitrace") -> bool:
    """Whether the unitrace binary can be found."""
    return shutil.which(unitrace_path) is not None


def flashinfer_bench_list_unitrace_modes() -> str:
    """List the profiling modes this wrapper supports.

    Returns
    -------
    str
        One mode per line, with a short description of what it reports.
    """
    described = {
        "device-timing": "Time spent on the device, per kernel. The default.",
        "host-timing": "Time spent in host-side runtime calls.",
        "kernel-submission": "Submission, execution and completion breakdown per kernel.",
        "device-timeline": "Per-invocation device timeline.",
        "chrome-kernel-logging": "Chrome trace of kernel activity, for a timeline view.",
        "chrome-device-logging": "Chrome trace of device activity.",
        "opencl": "Trace OpenCL calls.",
        "level-zero": "Trace Level Zero calls.",
    }
    return "\n".join(f"{mode:<24} {text}" for mode, text in described.items())


def _build_unitrace_command(
    data_dir: Path,
    modes: List[str],
    device: str,
    trace_set_path: Optional[Path],
    unitrace_path: str,
    output_dir: Path,
) -> List[str]:
    """Build the unitrace command line."""
    cmd: List[str] = [unitrace_path]
    for mode in modes:
        cmd.append(f"--{mode}")
    cmd.extend(["--output", str(output_dir / "unitrace")])

    runner_cmd = [
        sys.executable,
        "-u",
        "-m",
        "flashinfer_bench.agents._solution_runner",
        "--data-dir",
        str(data_dir),
        "--device",
        device,
    ]
    if trace_set_path:
        runner_cmd.extend(["--trace-set-path", str(trace_set_path)])
    cmd.extend(runner_cmd)
    return cmd


def _truncate_output(output: str, max_lines: int) -> str:
    """Keep the head of a long report, noting what was dropped."""
    lines = output.split("\n")
    if len(lines) <= max_lines:
        return output
    kept = lines[:max_lines]
    kept.append(f"... [{len(lines) - max_lines} more lines truncated]")
    return "\n".join(kept)


def flashinfer_bench_run_unitrace(
    solution: Union[Solution, str],
    workload: Union[Workload, str],
    *,
    device: str = "xpu:0",
    trace_set_path: Optional[str] = None,
    modes: Optional[List[str]] = None,
    unitrace_path: str = "unitrace",
    timeout: int = 300,
    tmpdir: Optional[str] = None,
    max_lines: Optional[int] = 200,
) -> str:
    """Profile a solution on an Intel GPU with unitrace.

    All inputs and outputs are JSON-serializable, making this suitable as an LLM agent
    tool, matching :func:`flashinfer_bench.agents.flashinfer_bench_run_ncu`.

    Parameters
    ----------
    solution : Solution or str
        The solution to profile, or a path to its JSON file.
    workload : Workload or str
        The workload giving input dimensions, or a path to its JSON file.
    device : str
        Intel GPU to profile on.
    trace_set_path : Optional[str]
        Dataset root. Falls back to ``FIB_DATASET_PATH``.
    modes : Optional[List[str]]
        unitrace modes to enable; see :func:`flashinfer_bench_list_unitrace_modes`.
        Defaults to device timing.
    unitrace_path : str
        Path to the unitrace binary.
    timeout : int
        Seconds before the profiling run is abandoned.
    tmpdir : Optional[str]
        Working directory for the run. A temporary one is used when omitted.
    max_lines : Optional[int]
        Truncate the report to this many lines. ``None`` keeps everything.

    Returns
    -------
    str
        unitrace's report, or an error message explaining what went wrong.
    """
    from flashinfer_bench.agents.solution_handler import extract_solution_to_files

    if not is_unitrace_available(unitrace_path):
        return (
            f"unitrace not found at '{unitrace_path}'. It is not on PyPI; build it from "
            "https://github.com/intel/pti-gpu (tools/unitrace) and put it on PATH."
        )

    selected = list(modes) if modes else list(DEFAULT_MODES)
    invalid = [m for m in selected if m not in VALID_MODES]
    if invalid:
        return (
            f"Unsupported unitrace mode(s): {', '.join(invalid)}. "
            f"Supported: {', '.join(sorted(VALID_MODES))}"
        )

    with tempfile.TemporaryDirectory(dir=tmpdir) as work_dir:
        data_dir = Path(work_dir)
        try:
            extract_solution_to_files(solution, workload, data_dir)
        except Exception as e:
            return f"Failed to stage solution for profiling: {type(e).__name__}: {e}"

        cmd = _build_unitrace_command(
            data_dir=data_dir,
            modes=selected,
            device=device,
            trace_set_path=Path(trace_set_path) if trace_set_path else None,
            unitrace_path=unitrace_path,
            output_dir=data_dir,
        )

        logger.debug("Running: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=work_dir
            )
        except subprocess.TimeoutExpired:
            return f"unitrace timed out after {timeout}s"
        except Exception as e:
            return f"Failed to run unitrace: {type(e).__name__}: {e}"

        # unitrace writes its report to stderr; stdout carries the program's own output.
        report = result.stderr or result.stdout
        if result.returncode != 0 and not report.strip():
            return f"unitrace exited with code {result.returncode} and produced no report"

        if max_lines is not None:
            report = _truncate_output(report, max_lines)
        return report
