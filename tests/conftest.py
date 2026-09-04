from pathlib import Path
from typing import List

import pytest


def _backend_available(backend: str) -> bool:
    """Check whether an accelerator backend is usable in this environment.

    Returns
    -------
    bool
        True if the backend is registered and reports itself available.
    """
    try:
        from flashinfer_bench.device import get_accelerator

        return type(get_accelerator(backend)).is_available()
    except Exception:
        return False


def _torch_cuda_available() -> bool:
    """Check if CUDA is available from PyTorch.

    Returns
    -------
    bool
        True if CUDA is available from PyTorch, False otherwise.
    """
    return _backend_available("cuda")


def _torch_xpu_available() -> bool:
    """Check if an Intel XPU is available from PyTorch.

    Returns
    -------
    bool
        True if an XPU is available from PyTorch, False otherwise.
    """
    return _backend_available("xpu")


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Skip tests whose required accelerator backend is not present."""
    has_cuda = _torch_cuda_available()
    has_xpu = _torch_xpu_available()

    skips = {
        "requires_torch_cuda": (
            has_cuda,
            pytest.mark.skip(reason="CUDA not available from PyTorch, skip test"),
        ),
        "requires_torch_xpu": (
            has_xpu,
            pytest.mark.skip(reason="Intel XPU not available from PyTorch, skip test"),
        ),
        "requires_accelerator": (
            has_cuda or has_xpu,
            pytest.mark.skip(reason="No accelerator available from PyTorch, skip test"),
        ),
    }

    for item in items:
        for marker_name, (available, skip_marker) in skips.items():
            if available:
                continue
            if any(item.iter_markers(name=marker_name)):
                item.add_marker(skip_marker)


@pytest.fixture
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use isolated temporary directory for cache in all tests.

    This fixture automatically sets FIB_CACHE_PATH to a unique temporary
    directory for each test, preventing cache pollution between tests.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))
    return cache_dir
