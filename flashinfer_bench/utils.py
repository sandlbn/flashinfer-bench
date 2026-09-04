"""Utility functions for FlashInfer-Bench."""

from __future__ import annotations

import os
import sys
import tempfile
from functools import cache
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    import torch

    from flashinfer_bench.data import Environment


@cache
def _get_dtype_str_to_python_dtype() -> Dict[str, type]:
    """Get dtype string to Python type mapping (cached)."""
    return {
        "float32": float,
        "float16": float,
        "bfloat16": float,
        "float8_e4m3fn": float,
        "float8_e5m2": float,
        "float4_e2m1": float,
        "int64": int,
        "int32": int,
        "int16": int,
        "int8": int,
        "bool": bool,
    }


def dtype_str_to_python_dtype(dtype_str: str) -> type:
    if not dtype_str:
        raise ValueError("dtype is None or empty")
    dtype = _get_dtype_str_to_python_dtype().get(dtype_str, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}'")
    return dtype


@cache
def _get_dtype_str_to_torch_dtype() -> Dict[str, torch.dtype]:
    """Lazily build dtype string to torch dtype mapping."""
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "float4_e2m1": torch.float4_e2m1fn_x2,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "bool": torch.bool,
    }


def dtype_str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    if not dtype_str:
        raise ValueError("dtype is None or empty")
    dtype = _get_dtype_str_to_torch_dtype().get(dtype_str, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}'")
    return dtype


@cache
def _get_integer_dtypes() -> frozenset:
    """Get frozenset of integer and boolean dtypes (cached)."""
    import torch

    return frozenset(
        (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
            torch.bool,
        )
    )


def is_dtype_integer(dtype: torch.dtype) -> bool:
    """Check if dtype is an integer or boolean type."""
    return dtype in _get_integer_dtypes()


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    from flashinfer_bench.device import CudaAccelerator

    return CudaAccelerator.is_available()


def list_cuda_devices() -> List[str]:
    """List visible CUDA devices.

    Deprecated in favour of :func:`list_devices`, which honours the configured backend.
    """
    from flashinfer_bench.device import CudaAccelerator

    return CudaAccelerator().list_devices()


def list_devices() -> List[str]:
    """List visible devices of the configured backend (see ``FIB_DEVICE_BACKEND``)."""
    from flashinfer_bench.device import list_devices as _list_devices

    return _list_devices()


def env_snapshot(device: str) -> Environment:
    import torch

    from flashinfer_bench.data import Environment
    from flashinfer_bench.device import get_accelerator

    libs: Dict[str, str] = {"torch": torch.__version__}
    try:
        import triton as _tr

        libs["triton"] = getattr(_tr, "__version__", "unknown")
    except Exception:
        pass

    try:
        import tilelang as _tl

        libs["tilelang"] = getattr(_tl, "__version__", "unknown")
    except Exception:
        pass

    accelerator = get_accelerator(device)
    libs.update(accelerator.lib_versions())
    # How the latency was measured, so numbers from different backends are never
    # silently compared as if one methodology produced both.
    libs["timing"] = accelerator.make_timer(device).name

    try:
        hardware_id: Optional[str] = accelerator.canonical_id(device)
    except Exception:
        hardware_id = None

    return Environment(hardware=hardware_from_device(device), hardware_id=hardware_id, libs=libs)


def hardware_from_device(device: str) -> str:
    """Raw vendor name of ``device``, as recorded in ``Environment.hardware``."""
    import torch

    from flashinfer_bench.device import get_accelerator

    d = torch.device(device)
    if d.type == "mps":
        return "Apple GPU (MPS)"
    try:
        return get_accelerator(device).device_name(device)
    except Exception:
        return d.type


def redirect_stdio_to_tempfile() -> str:
    """Redirect stdout/stderr to a temporary file.

    Returns the path to the temporary file.
    """

    sys.stdout.flush()
    sys.stderr.flush()
    fd, path = tempfile.mkstemp(suffix=".log", prefix="fib_")
    os.dup2(fd, 1)  # stdout -> fd
    os.dup2(fd, 2)  # stderr -> fd
    os.close(fd)
    sys.stdout = open(1, "w", encoding="utf-8", buffering=1, closefd=False)
    sys.stderr = open(2, "w", encoding="utf-8", buffering=1, closefd=False)
    return path
