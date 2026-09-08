"""Tests for the serving A/B harness.

The harness's job is to refuse to report a comparison it cannot justify, so what is worth
testing is the parsing that decides whether a run counted -- not the throughput arithmetic.
"""

import importlib.util
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "measure_serving_win",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "measure_serving_win.py",
)
msw = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(msw)


class TestParse:
    def test_reads_throughput_and_counters_through_vllm_worker_prefix(self):
        """vLLM tags worker output with ``(EngineCore pid=N) `` -- a space inside the parens.

        A parser anchored past that prefix finds no counters and the harness then reports
        the run as having no validity gate, discarding a measurement that was in fact fine.
        """
        out = "\n".join(
            [
                'FIB_RESULT {"generated_tokens": 2048, "seconds": 1.0, "tokens_per_sec": 2048.0}',
                "(EngineCore pid=596906) [flashinfer-bench] adapter dispatch:",
                "(EngineCore pid=596906)   rmsnorm: 8151 call(s), 143 applied (1.8%)",
                "(EngineCore pid=596906)   silu_and_mul: 4004 call(s), 4004 applied (100.0%)",
            ]
        )
        parsed = msw._parse(out)
        assert parsed["tokens_per_sec"] == 2048.0
        assert parsed["generated_tokens"] == 2048
        assert parsed["dispatch"] == {
            "rmsnorm": (8151, 143),
            "silu_and_mul": (4004, 4004),
        }

    def test_unprefixed_counters_parse_too(self):
        parsed = msw._parse("  rmsnorm: 10 call(s), 10 applied (100.0%)")
        assert parsed["dispatch"] == {"rmsnorm": (10, 10)}

    def test_run_without_counters_is_flagged_not_silently_accepted(self):
        parsed = msw._parse(
            'FIB_RESULT {"generated_tokens": 8, "seconds": 1.0, "tokens_per_sec": 8.0}'
        )
        assert parsed["tokens_per_sec"] == 8.0
        assert parsed["dispatch"] == {}  # caller must refuse to interpret this

    def test_failed_arm_reports_no_throughput(self):
        assert msw._parse("Engine core initialization failed")["tokens_per_sec"] is None


class TestSupported:
    def test_absent_vllm_is_fatal_not_undeterminable(self, monkeypatch):
        """A missing vLLM means this interpreter cannot run the harness at all.

        Returning None there would let the sweep report every model as "could not
        determine" while the real cause is a one-line environment mistake.
        """
        import builtins

        real_import = builtins.__import__

        def _no_vllm(name, *a, **kw):
            if name.startswith("vllm"):
                raise ImportError("No module named 'vllm'")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _no_vllm)
        with pytest.raises(SystemExit, match="not importable"):
            msw._supported("some/model")


class TestRootCause:
    def test_reports_the_worker_error_not_vllms_generic_wrapper(self):
        """vLLM's top-level message never names the cause, and a tail shows only it.

        The real reason sits far earlier in the worker's traceback, so a failure reported
        from the last few lines is the same uninformative string for every model.
        """
        output = "\n".join(
            [
                "INFO loading model",
                "(EngineCore pid=1) torch.OutOfMemoryError: XPU out of memory. Tried to "
                "allocate 2.00 GiB",
                'File "/x/vllm/engine.py", line 12, in run',
                "    raise RuntimeError(...)",
                "RuntimeError: Engine core initialization failed. See root cause above. "
                "Failed core proc(s): {}",
            ]
        )
        cause = msw._root_cause(output)
        assert "out of memory" in cause
        assert "See root cause above" not in cause

    def test_deduplicates_repeated_worker_lines(self):
        line = "(EngineCore pid=1) ValueError: unsupported quantization"
        assert msw._root_cause("\n".join([line] * 5)) == line.strip()

    def test_no_recognisable_error_yields_empty_so_caller_falls_back(self):
        assert msw._root_cause("INFO all good\nINFO done") == ""


class TestParseGluedOutput:
    def test_result_glued_to_a_newline_less_warning_is_still_read(self):
        """Dependencies print warnings without a trailing newline.

        The result line then arrives with that warning fused to its front, and a parser
        anchored at the start of the line throws away a run that in fact succeeded --
        reporting a working arm as a failure.
        """
        line = (
            "Warning: Triton requires a CUDA-enabled GPU. Falling back to reference "
            'implementation on CPU.FIB_RESULT {"generated_tokens": 32768, '
            '"seconds": 10.9344, "tokens_per_sec": 2996.77}'
        )
        parsed = msw._parse(line)
        assert parsed["tokens_per_sec"] == 2996.77
        assert parsed["generated_tokens"] == 32768


class TestMedian:
    def test_odd_and_even_lengths(self):
        assert msw._median([3.0, 1.0, 2.0]) == 2.0
        assert msw._median([4.0, 1.0, 3.0, 2.0]) == 2.5

    def test_single_run(self):
        assert msw._median([7.5]) == 7.5
