"""Builder for SYCL kernels, targeting Intel GPUs through oneAPI DPC++.

SYCL is where the fastest Intel kernels are written today, so it is a first-class
solution language rather than something bolted onto the C++ path. The programming model
mirrors CUDA's exactly: a solution receives DLPack tensors and asks the environment for
the framework's stream. On CUDA that stream is a ``cudaStream_t``; here it is a
``sycl::queue*``, already bound to the context that owns the tensors, so kernels run on
PyTorch's own queue with PyTorch's own USM pointers.

.. code-block:: cpp

    sycl::queue* q = static_cast<sycl::queue*>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));
    q->parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { out[i] = x[i] + y[i]; });

Sources use ordinary C++ extensions (``.cpp``); SYCL is C++. The ``sycl`` language tag on
the solution is what selects this builder and adds ``-fsycl``.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import shutil
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Callable, ClassVar, Dict, Iterator, List, Optional

from flashinfer_bench.compile.builder import Builder, BuildError
from flashinfer_bench.compile.runnable import Runnable, RunnableMetadata
from flashinfer_bench.compile.utils import write_sources_to_path
from flashinfer_bench.data import Definition, Solution, SupportedLanguages

logger = logging.getLogger(__name__)

_SYCL_COMPILERS = ("icpx", "dpcpp")
"""oneAPI DPC++ driver names, in preference order."""

_ONEAPI_SEARCH_GLOBS = ("{root}/compiler/*/bin/icpx", "{root}/compiler/latest/bin/icpx")
"""Where a oneAPI installation keeps the compiler, relative to its root."""

_ONEAPI_DEFAULT_ROOTS = ("/opt/intel/oneapi", "/usr/local/oneapi")
"""Standard install locations to check when ``ONEAPI_ROOT`` is not set."""

_DEPENDENCY_LDFLAGS: Dict[str, List[str]] = {
    "onemkl": ["-fsycl", "-lmkl_sycl", "-lmkl_intel_ilp64", "-lmkl_core", "-lmkl_tbb_thread"],
    "mkl": ["-fsycl", "-lmkl_sycl", "-lmkl_intel_ilp64", "-lmkl_core", "-lmkl_tbb_thread"],
    "onednn": ["-ldnnl"],
    "dnnl": ["-ldnnl"],
    "level_zero": ["-lze_loader"],
}
"""Link flags for dependencies a solution may declare."""

_env_lock = threading.Lock()
"""Serializes the temporary ``CXX`` override, which is process-global."""


@lru_cache(maxsize=1)
def find_sycl_compiler() -> Optional[str]:
    """Locate a oneAPI DPC++ compiler, or ``None`` if none is installed.

    Checks, in order: an explicit ``FIB_SYCL_COMPILER``; ``CXX`` when it already names a
    DPC++ driver; ``PATH``; then a oneAPI installation under ``ONEAPI_ROOT`` or a standard
    location. The last case matters because oneAPI is usually installed without being on
    ``PATH`` until ``setvars.sh`` is sourced.
    """
    explicit = os.environ.get("FIB_SYCL_COMPILER")
    if explicit and Path(explicit).exists():
        return explicit

    configured = os.environ.get("CXX", "")
    if any(Path(configured).name.startswith(name) for name in _SYCL_COMPILERS if configured):
        found = shutil.which(configured)
        if found:
            return found

    for name in _SYCL_COMPILERS:
        found = shutil.which(name)
        if found:
            return found

    roots = [os.environ["ONEAPI_ROOT"]] if os.environ.get("ONEAPI_ROOT") else []
    roots.extend(_ONEAPI_DEFAULT_ROOTS)
    for root in roots:
        for pattern in _ONEAPI_SEARCH_GLOBS:
            matches = sorted(glob.glob(pattern.format(root=root)), reverse=True)
            if matches:
                return matches[0]

    return None


@contextmanager
def _compiler_env(compiler: str) -> Iterator[None]:
    """Point the build at the SYCL compiler for the duration of one build.

    ``tvm_ffi.cpp.build`` selects its C++ compiler from ``CXX``, which is process-global,
    so the override is locked and restored.
    """
    with _env_lock:
        previous = os.environ.get("CXX")
        os.environ["CXX"] = compiler
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("CXX", None)
            else:
                os.environ["CXX"] = previous


class SyclBuilder(Builder):
    """Builder for SYCL solutions, compiled with oneAPI DPC++ and loaded via TVM-FFI.

    Kernels are compiled ahead of time when the target device's architecture is known, and
    otherwise to SPIR-V and JIT-compiled at load time. The SPIR-V path is what lets a
    solution run on a device that did not exist when it was written.
    """

    _PACKAGE_PREFIX: ClassVar[str] = "sycl_"
    """A unique prefix to prepend to the package name."""

    _BUILD_DIR_NAME: ClassVar[str] = "sycl"
    """A unique subdirectory under FIB_CACHE_PATH where build results are stored."""

    _CPP_EXTENSIONS: ClassVar[List[str]] = [".cpp", ".cc", ".cxx", ".sycl"]
    """Source extensions compiled as SYCL."""

    def __init__(self) -> None:
        super().__init__(self._PACKAGE_PREFIX, self._BUILD_DIR_NAME)

    @staticmethod
    def is_available() -> bool:
        """Check whether SYCL solutions can be built here.

        Requires a DPC++ compiler and TVM-FFI. Deliberately does not require an Intel GPU
        to be present: building and running are separate concerns, and a build machine
        need not have the target device.

        Returns
        -------
        bool
            True if a SYCL toolchain is available, False otherwise.
        """
        try:
            import tvm_ffi  # noqa: F401
        except ImportError:
            return False
        return find_sycl_compiler() is not None

    def can_build(self, solution: Solution) -> bool:
        """Check if this builder can handle the given solution.

        Parameters
        ----------
        solution : Solution
            Solution to check.

        Returns
        -------
        bool
            True if the solution's language is SYCL.
        """
        return solution.spec.language == SupportedLanguages.SYCL

    def _filter_sources(self, source_paths: List[Path]) -> List[str]:
        """Select the source files to compile."""
        return [str(p) for p in source_paths if p.suffix in self._CPP_EXTENSIONS]

    def _target_flags(self, solution: Solution) -> List[str]:
        """Ahead-of-time target flags for the devices this solution targets.

        Falls back to SPIR-V JIT when no target architecture is known, which is both the
        portable choice and the only way to run on a device whose AOT triple has not been
        published yet.
        """
        from flashinfer_bench.device import get_accelerator

        targets = []
        for hardware in solution.spec.target_hardware:
            try:
                accelerator = get_accelerator(hardware)
            except Exception:
                continue
            try:
                sycl_target = accelerator.capabilities(hardware).sycl_target
            except Exception:
                continue
            if sycl_target:
                targets.append(sycl_target)

        if not targets:
            return []
        return [f"-fsycl-targets={','.join(sorted(set(targets)))}"]

    def _link_flags(self, solution: Solution) -> List[str]:
        """Link flags implied by the solution's declared dependencies."""
        flags: List[str] = ["-fsycl"]
        for dependency in solution.spec.dependencies:
            extra = _DEPENDENCY_LDFLAGS.get(dependency.lower())
            if extra:
                flags.extend(f for f in extra if f not in flags)
            else:
                logger.debug("No known link flags for SYCL dependency '%s'", dependency)
        return flags

    def _get_cleaner(self, build_dir: Path) -> Callable[[], None]:
        def cleaner() -> None:
            shutil.rmtree(build_dir, ignore_errors=True)

        return cleaner

    def build(self, definition: Definition, solution: Solution) -> Runnable:
        """Build a SYCL solution into a runnable.

        Parameters
        ----------
        definition : Definition
            The problem definition specifying the expected interface.
        solution : Solution
            The SYCL solution to build.

        Returns
        -------
        Runnable
            An executable wrapper around the compiled kernel.

        Raises
        ------
        BuildError
            If no compiler is available, compilation fails, or the entry point is missing.
        """
        import tvm_ffi
        import tvm_ffi.cpp

        compiler = find_sycl_compiler()
        if compiler is None:
            raise BuildError(
                "No SYCL compiler found. Install oneAPI DPC++ and either put 'icpx' on "
                "PATH (source setvars.sh), set ONEAPI_ROOT, or set FIB_SYCL_COMPILER."
            )

        package_name, build_path = self._get_package_name_and_build_path(solution)
        build_path.mkdir(parents=True, exist_ok=True)

        source_paths = write_sources_to_path(build_path, solution.sources)
        sources = self._filter_sources(source_paths)
        if not sources:
            raise BuildError(
                f"Solution '{solution.name}' has no SYCL source files "
                f"(expected one of {', '.join(self._CPP_EXTENSIONS)})"
            )

        cflags = ["-fsycl", "-O3", *self._target_flags(solution)]

        try:
            with _compiler_env(compiler):
                output_lib_path = tvm_ffi.cpp.build(
                    name=package_name,
                    cpp_files=sources,
                    extra_cflags=cflags,
                    extra_ldflags=self._link_flags(solution),
                    extra_include_paths=[str(build_path)],
                    build_directory=build_path,
                )
        except Exception as e:
            raise BuildError(f"SYCL compilation failed for '{solution.name}': {e}") from e

        # Pin the library into the process before handing it to TVM-FFI.
        #
        # RTLD_NODELETE is the point: unloading a SYCL module runs its static destructors
        # against a SYCL runtime that may already have torn itself down, and the process
        # crashes in dlclose at exit once more than one such module is loaded. The work
        # itself completes correctly -- it is purely a teardown ordering problem -- but a
        # benchmark worker that dies while shutting down looks like a failed evaluation.
        # Keeping the mapping for the life of the process avoids the unload entirely.
        #
        # The load doubles as a symbol check: unresolved symbols surface here with a clear
        # message rather than as an opaque failure inside TVM-FFI.
        try:
            ctypes.CDLL(output_lib_path, mode=os.RTLD_NOW | os.RTLD_NODELETE)
        except OSError as e:
            raise BuildError(f"SYCL module has unresolved symbols: {e}") from e

        try:
            mod = tvm_ffi.load_module(output_lib_path)
        except Exception as e:
            raise BuildError(f"Failed to load compiled SYCL module: {e}") from e

        entry_symbol = solution.get_entry_symbol()
        try:
            callable = getattr(mod, entry_symbol)
        except AttributeError as e:
            raise BuildError(
                f"Entry point '{entry_symbol}' not found in module. Export it with "
                f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({entry_symbol}, YourFunction)."
            ) from e

        self._try_validate_signature(callable, definition, solution)

        metadata = RunnableMetadata(
            build_type="sycl",
            definition_name=definition.name,
            solution_name=solution.name,
            destination_passing_style=solution.spec.destination_passing_style,
            definition=definition,
            misc={
                "entry_symbol": entry_symbol,
                "binary": output_lib_path,
                "compiler": compiler,
                "cflags": " ".join(cflags),
            },
        )

        return Runnable(callable=callable, metadata=metadata, cleaner=self._get_cleaner(build_path))
