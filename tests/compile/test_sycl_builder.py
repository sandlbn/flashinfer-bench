"""Tests for the SYCL builder.

The compiler-independent parts (discovery, language gating, flag derivation) run
anywhere. The end-to-end build runs only where oneAPI DPC++ and an Intel GPU are both
present.
"""

import shutil
from pathlib import Path

import pytest

from flashinfer_bench.compile.builder import BuildError
from flashinfer_bench.compile.builders import SyclBuilder
from flashinfer_bench.compile.builders import sycl_builder as sb
from flashinfer_bench.data import (
    BuildSpec,
    Definition,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
)

ADD_KERNEL = """#include <sycl/sycl.hpp>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

void AddSycl(tvm::ffi::TensorView x, tvm::ffi::TensorView y, tvm::ffi::TensorView out) {
  const int64_t n = x.size(0);
  DLDevice dev = x.device();
  sycl::queue* q = static_cast<sycl::queue*>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  TVM_FFI_ICHECK(q != nullptr) << "no SYCL queue for device";
  const float* xd = static_cast<const float*>(x.data_ptr());
  const float* yd = static_cast<const float*>(y.data_ptr());
  float* od = static_cast<float*>(out.data_ptr());
  q->parallel_for(sycl::range<1>(static_cast<size_t>(n)),
                  [=](sycl::id<1> i) { od[i] = xd[i] + yd[i]; });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_sycl, AddSycl);
"""


def _definition() -> Definition:
    return Definition(
        name="sycl_add",
        op_type="gemm",
        axes={"n": {"type": "var"}},
        inputs={
            "x": TensorSpec(shape=["n"], dtype="float32"),
            "y": TensorSpec(shape=["n"], dtype="float32"),
        },
        outputs={"out": TensorSpec(shape=["n"], dtype="float32")},
        reference="import torch\n\n\ndef run(x, y):\n    return x + y\n",
    )


def _solution(
    *,
    language: SupportedLanguages = SupportedLanguages.SYCL,
    target_hardware=None,
    dependencies=None,
    content: str = ADD_KERNEL,
    path: str = "add.cpp",
) -> Solution:
    return Solution(
        name="sycl_add_sol",
        definition="sycl_add",
        author="test",
        spec=BuildSpec(
            language=language,
            target_hardware=target_hardware or ["xpu"],
            entry_point=f"{path}::add_sycl",
            dependencies=dependencies or [],
            destination_passing_style=True,
        ),
        sources=[SourceFile(path=path, content=content)],
    )


def _has_toolchain() -> bool:
    return SyclBuilder.is_available()


class TestCompilerDiscovery:
    def test_explicit_override_wins(self, monkeypatch, tmp_path):
        fake = tmp_path / "my-icpx"
        fake.write_text("")
        monkeypatch.setenv("FIB_SYCL_COMPILER", str(fake))
        sb.find_sycl_compiler.cache_clear()
        assert sb.find_sycl_compiler() == str(fake)
        sb.find_sycl_compiler.cache_clear()

    def test_finds_an_unsourced_oneapi_install(self, monkeypatch, tmp_path):
        """oneAPI is normally installed without being on PATH until setvars.sh runs."""
        compiler = tmp_path / "compiler" / "2026.0" / "bin" / "icpx"
        compiler.parent.mkdir(parents=True)
        compiler.write_text("")
        monkeypatch.delenv("FIB_SYCL_COMPILER", raising=False)
        monkeypatch.setenv("ONEAPI_ROOT", str(tmp_path))
        monkeypatch.setattr(sb.shutil, "which", lambda name: None)
        monkeypatch.setenv("CXX", "")
        sb.find_sycl_compiler.cache_clear()
        assert sb.find_sycl_compiler() == str(compiler)
        sb.find_sycl_compiler.cache_clear()

    def test_absent_toolchain_reports_none(self, monkeypatch):
        monkeypatch.delenv("FIB_SYCL_COMPILER", raising=False)
        monkeypatch.delenv("ONEAPI_ROOT", raising=False)
        monkeypatch.setenv("CXX", "")
        monkeypatch.setattr(sb.shutil, "which", lambda name: None)
        monkeypatch.setattr(sb.glob, "glob", lambda pattern: [])
        sb.find_sycl_compiler.cache_clear()
        assert sb.find_sycl_compiler() is None
        sb.find_sycl_compiler.cache_clear()


class TestCanBuild:
    def test_accepts_sycl_solutions(self):
        assert SyclBuilder().can_build(_solution())

    @pytest.mark.parametrize(
        "language", [SupportedLanguages.CUDA, SupportedLanguages.CPP, SupportedLanguages.TRITON]
    )
    def test_rejects_other_languages(self, language):
        assert not SyclBuilder().can_build(_solution(language=language))


class TestBuildFlags:
    def test_unknown_target_falls_back_to_spirv_jit(self):
        """No AOT triple means SPIR-V, which is how an unreleased device runs."""
        assert SyclBuilder()._target_flags(_solution(target_hardware=["xpu"])) == []

    def test_known_dependencies_become_link_flags(self):
        flags = SyclBuilder()._link_flags(_solution(dependencies=["onednn"]))
        assert "-fsycl" in flags and "-ldnnl" in flags

    def test_unknown_dependency_is_ignored_not_fatal(self):
        flags = SyclBuilder()._link_flags(_solution(dependencies=["not-a-real-library"]))
        assert flags == ["-fsycl"]

    def test_link_flags_are_deduplicated(self):
        flags = SyclBuilder()._link_flags(_solution(dependencies=["onemkl", "mkl"]))
        assert flags.count("-fsycl") == 1

    def test_only_sycl_sources_are_compiled(self, tmp_path):
        paths = [tmp_path / "a.cpp", tmp_path / "b.h", tmp_path / "c.py", tmp_path / "d.cc"]
        selected = SyclBuilder()._filter_sources(paths)
        assert [Path(p).name for p in selected] == ["a.cpp", "d.cc"]


class TestBuildErrors:
    def test_missing_compiler_explains_how_to_fix_it(self, monkeypatch, tmp_cache_dir):
        monkeypatch.setattr(sb, "find_sycl_compiler", lambda: None)
        with pytest.raises(BuildError, match="No SYCL compiler found"):
            SyclBuilder().build(_definition(), _solution())

    @pytest.mark.skipif(not _has_toolchain(), reason="no SYCL toolchain")
    def test_solution_without_sycl_sources_is_rejected(self, tmp_cache_dir):
        solution = _solution(path="kernel.txt", content="not source")
        with pytest.raises(BuildError, match="no SYCL source files"):
            SyclBuilder().build(_definition(), solution)


@pytest.mark.requires_torch_xpu
@pytest.mark.skipif(not _has_toolchain(), reason="no SYCL toolchain")
@pytest.mark.skipif(shutil.which("ninja") is None, reason="ninja not on PATH")
class TestEndToEnd:
    def test_kernel_compiles_and_computes_correctly(self, tmp_cache_dir):
        import torch

        runnable = SyclBuilder().build(_definition(), _solution())
        assert runnable.metadata.build_type == "sycl"

        x = torch.randn(4096, device="xpu:0")
        y = torch.randn(4096, device="xpu:0")
        out = torch.empty_like(x)
        runnable(x, y, out)
        torch.xpu.synchronize()
        assert torch.equal(out, x + y)

    def test_missing_entry_point_names_the_export_macro(self, tmp_cache_dir):
        definition = _definition()
        solution = _solution()
        wrong = solution.model_copy(
            update={"spec": solution.spec.model_copy(update={"entry_point": "add.cpp::nope"})}
        )
        with pytest.raises(BuildError, match="TVM_FFI_DLL_EXPORT_TYPED_FUNC"):
            SyclBuilder().build(definition, wrong)

    def test_compilation_failure_is_reported_as_a_build_error(self, tmp_cache_dir):
        broken = _solution(content="#include <sycl/sycl.hpp>\nthis is not c++\n")
        with pytest.raises(BuildError, match="SYCL compilation failed"):
            SyclBuilder().build(_definition(), broken)
