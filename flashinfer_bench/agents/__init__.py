"""Agent tools for LLM-based kernel development and debugging."""

from .ffi_prompt import FFI_PROMPT, FFI_PROMPT_SIMPLE
from .ncu import flashinfer_bench_list_ncu_options, flashinfer_bench_run_ncu
from .sanitizer import flashinfer_bench_run_sanitizer
from .schema import function_to_schema, get_all_tool_schemas
from .solution_handler import extract_solution_to_files, pack_solution_from_files
from .sycl_prompt import SYCL_PROMPT, SYCL_PROMPT_SIMPLE
from .unitrace import (
    UNITRACE_ENV,
    find_unitrace,
    flashinfer_bench_list_unitrace_modes,
    flashinfer_bench_run_unitrace,
    is_unitrace_available,
)

__all__ = [
    "flashinfer_bench_list_ncu_options",
    "flashinfer_bench_run_ncu",
    "flashinfer_bench_run_sanitizer",
    "flashinfer_bench_list_unitrace_modes",
    "flashinfer_bench_run_unitrace",
    "is_unitrace_available",
    "find_unitrace",
    "UNITRACE_ENV",
    "flashinfer_bench_list_vtune_modes",
    "flashinfer_bench_run_vtune",
    "profile_command_with_vtune",
    "is_vtune_available",
    "find_vtune",
    "VTUNE_ENV",
    "function_to_schema",
    "get_all_tool_schemas",
    "FFI_PROMPT_SIMPLE",
    "FFI_PROMPT",
    "SYCL_PROMPT_SIMPLE",
    "SYCL_PROMPT",
    "extract_solution_to_files",
    "pack_solution_from_files",
]

_VTUNE_EXPORTS = frozenset(
    {
        "VTUNE_ENV",
        "find_vtune",
        "flashinfer_bench_list_vtune_modes",
        "flashinfer_bench_run_vtune",
        "is_vtune_available",
        "profile_command_with_vtune",
    }
)


def __getattr__(name):
    # Resolved on first use rather than at package import, so that
    # `python -m flashinfer_bench.agents.vtune` runs the module once, not twice.
    if name in _VTUNE_EXPORTS:
        from . import vtune as _vtune

        return getattr(_vtune, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
