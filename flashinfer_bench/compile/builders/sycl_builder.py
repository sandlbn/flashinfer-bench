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
import sys
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

_AOT_DEVICE_ENV = "FIB_SYCL_AOT_DEVICE"
"""ocloc device name to compile ahead-of-time for (e.g. ``xe3``, ``bmg-g21``, ``pvc``).

Without an AOT target the kernel is JIT-compiled from generic SPIR-V, and CUTLASS-SYCL in
particular can fall back to generic code paths instead of the DPAS / 2D-block-IO ones.
``ocloc compile -device <name>`` lists what a given driver accepts.
"""

_LARGE_GRF_ENV = "FIB_SYCL_LARGE_GRF"
"""Set truthy to build AOT kernels in large-GRF mode (256 registers/thread, not 128).

Large tiles spill catastrophically in the default small-GRF mode, and spilling is
invisible to the correctness gate -- the kernel is right, just slow. An Xe-Fuse ``k2`` at
tile 256x256x32 on Battlemage spilled 8576 bytes/thread and ran 8.8 ms against oneDNN's
0.46 ms for the same work. Check ``Spill Memory Per Thread`` in unitrace's kernel
properties before reaching for this: a smaller tile is usually the better fix, since large
GRF halves the threads resident per EU.
"""

_ONEDNN_ENV = "FIB_ONEDNN_DIR"
"""Points at a oneDNN install, for solutions declaring the ``onednn`` dependency."""

_ONEDNN_DEFAULT_ROOTS = ("/opt/intel/oneapi/dnnl/latest", "/usr")
"""Where oneDNN usually lives when oneAPI is installed."""

_XE_FUSE_ENV = "FIB_XE_FUSE_DIR"
"""Points at an Xe-Fuse checkout, for solutions that declare the ``xe-fuse`` dependency."""

_SYCL_TLA_ENV = "FIB_SYCL_TLA_DIR"
"""Points at a sycl-tla (CUTLASS-SYCL) checkout. Defaults to Xe-Fuse's fetched copy."""

_CUTLASS_SYCL_SPIRV_EXTS = (
    "+SPV_INTEL_split_barrier",
    "+SPV_INTEL_2d_block_io",
    "+SPV_INTEL_subgroup_matrix_multiply_accumulate",
)
"""SPIR-V extensions CUTLASS-SYCL kernels emit instructions for.

They are off by default, and the failure is at *link* time, not compile:
``RequiresExtension: Feature requires the following SPIR-V extension``. The block-io and
matrix-multiply-accumulate extensions are what the XMX/DPAS paths need.
"""

_CUTLASS_SYCL_DEFINES = ("-DCUTLASS_ENABLE_SYCL", "-DSYCL_INTEL_TARGET")
"""Switches CUTLASS to its SYCL backend.

Without these, CUTLASS assumes CUDA and the compile fails looking for
``cuda_runtime_api.h`` -- a confusing error that has nothing to do with the kernel.
"""

_DEPENDENCY_LDFLAGS: Dict[str, List[str]] = {
    "onemkl": ["-fsycl", "-lmkl_sycl", "-lmkl_intel_ilp64", "-lmkl_core", "-lmkl_tbb_thread"],
    "mkl": ["-fsycl", "-lmkl_sycl", "-lmkl_intel_ilp64", "-lmkl_core", "-lmkl_tbb_thread"],
    # oneDNN's -L and -rpath are added by _link_flags from the resolved root, since the
    # install prefix is discovered at build time and may come from FIB_ONEDNN_DIR.
    "onednn": ["-ldnnl"],
    "dnnl": ["-ldnnl"],
    "onednn-sycl": ["-ldnnl"],
    "level_zero": ["-lze_loader"],
}
"""Link flags for dependencies a solution may declare."""

_ONEDNN_DEP_NAMES = frozenset({"onednn", "dnnl", "onednn-sycl"})
"""Dependency spellings that select oneDNN."""


@lru_cache(maxsize=1)
def find_onednn_root() -> Optional[str]:
    """Locate a oneDNN install, or ``None`` if none is present.

    ``FIB_ONEDNN_DIR`` wins, then the standard oneAPI and system prefixes. The header is
    what is probed, because a prefix carrying only the runtime library cannot build.
    """
    roots = [os.environ[_ONEDNN_ENV]] if os.environ.get(_ONEDNN_ENV) else []
    roots.extend(_ONEDNN_DEFAULT_ROOTS)
    for root in roots:
        if (Path(root) / "include" / "oneapi" / "dnnl" / "dnnl.hpp").exists():
            return root
    return None


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


def _ensure_build_tools_on_path() -> None:
    """Put the running interpreter's bin/ on PATH before invoking the compiler driver.

    `ninja` ships as a console script in the environment that installed it, so a process
    started as `/path/to/venv/bin/python script.py` -- which is how every harness and
    subprocess here starts one -- has the package importable but the executable off PATH.
    The build then fails with FileNotFoundError: 'ninja', which reads as a broken solution
    rather than an invisible toolchain, and each caller that hits it fixes it locally. It
    has surfaced three separate times in this repo; fixing it where the compiler is invoked
    means no caller has to know.
    """
    bindir = str(Path(sys.executable).parent)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if bindir not in parts:
        os.environ["PATH"] = os.pathsep.join([bindir, *parts])


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

    def _dependency_cflags(self, solution: Solution) -> List[str]:
        """Compile flags implied by the solution's declared dependencies.

        Currently only ``xe-fuse``, which needs CUTLASS-SYCL headers and the two defines
        that select its SYCL backend.
        """
        deps = {d.lower() for d in solution.spec.dependencies}
        flags: List[str] = []

        if deps & _ONEDNN_DEP_NAMES:
            root = find_onednn_root()
            if root is None:
                raise BuildError(
                    "Solution declares a oneDNN dependency but dnnl.hpp was not found. "
                    f"Install oneAPI or set {_ONEDNN_ENV} to a oneDNN prefix."
                )
            flags.append(f"-I{Path(root) / 'include'}")

        if "xe-fuse" not in deps and "xefuse" not in deps:
            return flags

        xe_fuse_dir = os.environ.get(_XE_FUSE_ENV)
        if not xe_fuse_dir or not Path(xe_fuse_dir).exists():
            raise BuildError(
                f"Solution declares the 'xe-fuse' dependency but {_XE_FUSE_ENV} is not set "
                "to an Xe-Fuse checkout. Clone https://github.com/IntelLabs/Xe-Fuse and "
                f"export {_XE_FUSE_ENV}=/path/to/Xe-Fuse"
            )

        xe_fuse = Path(xe_fuse_dir)
        sycl_tla = Path(
            os.environ.get(_SYCL_TLA_ENV) or xe_fuse / "build" / "_deps" / "sycl_tla-src"
        )
        if not sycl_tla.exists():
            raise BuildError(
                f"sycl-tla not found at {sycl_tla}. Configure Xe-Fuse once so its CMake "
                f"fetches it, or set {_SYCL_TLA_ENV} to an existing checkout."
            )

        aot_device = os.environ.get(_AOT_DEVICE_ENV)
        if aot_device:
            # Must be on the compile line too, not just the link line: the target gates
            # which device code paths the SYCL front end emits.
            flags.append("-fsycl-targets=spir64_gen")
        flags += list(_CUTLASS_SYCL_DEFINES)
        for include in (
            xe_fuse / "include",
            sycl_tla / "include",
            sycl_tla / "tools" / "util" / "include",
            sycl_tla / "examples" / "common",
            sycl_tla / "applications",
        ):
            flags.append(f"-I{include}")
        return flags

    def _aot_devices(self, solution: Solution) -> List[str]:
        """ocloc device names for the hardware this solution targets, sorted and deduped.

        Empty when no target architecture is known, which selects SPIR-V JIT -- both the
        portable choice and the only way to run on a device whose AOT device name has not
        been published yet.
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

        return sorted(set(targets))

    def _supports_large_grf(self, solution: Solution) -> bool:
        """Whether every target device can run in large-GRF mode.

        Asked of the device rather than assumed, so the flag is not passed to a backend
        that has no such mode.
        """
        from flashinfer_bench.device import get_accelerator

        for hardware in solution.spec.target_hardware:
            try:
                caps = get_accelerator(hardware).capabilities(hardware)
            except Exception:
                continue
            if caps.supports_large_grf:
                return True
        return False

    def _target_flags(self, solution: Solution) -> List[str]:
        """Compile-time AOT flags.

        ``sycl_target`` holds an ocloc device name (``bmg``), which is *not* a valid
        ``-fsycl-targets`` value -- passing it directly fails with ``invalid or
        unsupported offload target``. The offload target is the ``spir64_gen`` backend;
        the device name is a backend option, and belongs on the link step where device
        code is actually generated. See :meth:`_target_link_flags`.
        """
        if not self._aot_devices(solution):
            return []
        return ["-fsycl-targets=spir64_gen"]

    def _target_link_flags(self, solution: Solution) -> List[str]:
        """Link-time AOT flags, where ``spir64_gen`` generates the device binary.

        Each ``-Xs`` forwards exactly one token to the backend, so the option and its
        value are passed as two separate ``-Xs`` pairs. Writing it as a single
        ``-Xs "-device bmg"`` would need a quoted argument containing a space, which does
        not survive being joined into a ninja command line.
        """
        devices = self._aot_devices(solution)
        if not devices:
            return []
        flags = ["-fsycl-targets=spir64_gen", "-Xs", "-device", "-Xs", ",".join(devices)]
        if os.environ.get(_LARGE_GRF_ENV, "").lower() in ("1", "true", "yes", "on"):
            if not self._supports_large_grf(solution):
                logger.warning(
                    "%s is set but no target device reports large-GRF support; building "
                    "without it rather than passing a flag the backend may reject.",
                    _LARGE_GRF_ENV,
                )
                return flags
            # An ocloc backend option, so it rides the same -Xs channel as -device. Only
            # meaningful for an AOT build; a SPIR-V JIT gets its register mode from the
            # runtime instead.
            flags += ["-Xs", "-options", "-Xs", "-ze-opt-large-register-file"]
        return flags

    def _link_flags(self, solution: Solution) -> List[str]:
        """Link flags implied by the solution's declared dependencies."""
        flags: List[str] = ["-fsycl"]
        deps_lower = {d.lower() for d in solution.spec.dependencies}
        if deps_lower & _ONEDNN_DEP_NAMES:
            # -ldnnl alone cannot resolve: oneDNN lives outside the default search path.
            # The rpath must name the same prefix, so an install found via FIB_ONEDNN_DIR
            # is also the one loaded at runtime.
            root = find_onednn_root()
            if root is not None:
                lib = Path(root) / "lib"
                flags += [f"-L{lib}", f"-Wl,-rpath,{lib}"]
        if "xe-fuse" in deps_lower or "xefuse" in deps_lower:
            flags += ["-Xspirv-translator", f"-spirv-ext={','.join(_CUTLASS_SYCL_SPIRV_EXTS)}"]
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
        _ensure_build_tools_on_path()
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

        cflags = [
            "-fsycl",
            "-O3",
            *self._target_flags(solution),
            *self._dependency_cflags(solution),
        ]

        try:
            with _compiler_env(compiler):
                output_lib_path = tvm_ffi.cpp.build(
                    name=package_name,
                    cpp_files=sources,
                    extra_cflags=cflags,
                    extra_ldflags=[*self._link_flags(solution), *self._target_link_flags(solution)],
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
