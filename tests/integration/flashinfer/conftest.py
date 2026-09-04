"""Skip the FlashInfer adapter tests when flashinfer-python is not installed.

`flashinfer-python` is an optional dependency (the `cuda` extra) because it is
CUDA-oriented and installing it on a non-NVIDIA machine pulls a toolchain that will
never be used. These tests exercise the adapters against the real package, so without
it there is nothing to test rather than something failing.
"""

import importlib.util

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if importlib.util.find_spec("flashinfer") is not None:
        return

    skip = pytest.mark.skip(
        reason="flashinfer-python not installed (pip install flashinfer-bench[cuda])"
    )
    here = __file__.rsplit("/", 1)[0]
    for item in items:
        if str(item.fspath).startswith(here):
            item.add_marker(skip)
