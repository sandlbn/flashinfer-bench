"""Tests for the per-part calibration, run without a GPU.

What is tested is the contract the deploy gate depends on: an unmeasurable dispatch cost is
reported as ``None`` and never as ``0.0``, a measurement that did not settle is refused, and
the two places the measurement reaches into the apply runtime put things back.
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from flashinfer_bench.device import calibration


class TestSettledMedian:
    def test_agreeing_rounds_report_their_median(self):
        assert calibration._settled_median([6.0, 6.1, 5.9, 6.05, 5.95]) == pytest.approx(6.0)

    def test_noise_around_zero_is_refused_not_reported_as_free(self):
        assert calibration._settled_median([0.1, -0.2, 0.05, 0.3, -0.1]) is None

    def test_negative_median_is_refused(self):
        assert calibration._settled_median([-6.0, -5.9, -6.1]) is None

    def test_exactly_zero_is_refused(self):
        assert calibration._settled_median([0.0, 0.0, 0.0]) is None

    def test_rounds_that_disagree_are_refused(self):
        # The pattern a thread bounced between core classes produces: a median exists,
        # but it describes the moment, not the machine.
        assert calibration._settled_median([6.0, 14.0, 6.1, 13.5, 9.4, 6.2, 12.0]) is None

    def test_tolerance_bounds_the_spread(self):
        deltas = [5.0, 6.0, 5.5, 4.5, 6.5]  # median 5.5, MAD 0.5 -> 9%
        assert calibration._settled_median(deltas, tolerance=0.10) == pytest.approx(5.5)
        assert calibration._settled_median(deltas, tolerance=0.05) is None

    def test_empty_is_none(self):
        assert calibration._settled_median([]) is None


class TestPairedDelta:
    def test_distinguishable_arms_report_a_positive_cost(self):
        slow = lambda: sum(range(3000))  # noqa: E731
        fast = lambda: sum(range(100))  # noqa: E731
        delta = calibration._paired_delta_us(slow, fast, calls=50, rounds=9, warmup_s=0.01)
        assert delta is not None and delta > 0

    def test_identical_arms_never_report_a_cost(self):
        fn = lambda: sum(range(300))  # noqa: E731
        for _ in range(3):
            delta = calibration._paired_delta_us(fn, fn, calls=50, rounds=9, warmup_s=0.01)
            assert delta is None or delta > 0  # never 0.0, and None is the expected answer
        # Whatever it returns, it is not a claim that a wrapper is free.
        assert delta != 0.0


class TestGetNeverFabricatesZero:
    """``get()`` on the CPU backend with ``measure`` replaced, so no device is needed."""

    @pytest.fixture
    def cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FIB_CACHE_PATH", str(tmp_path))
        return tmp_path / "calibration"

    def _record(self, dispatch):
        return calibration.Calibration(
            hardware_id="CPU",
            timer="wall",
            dispatch_us=dispatch,
            timing_floor_us=1.0,
            bandwidth_gbs=1.0,
        )

    def test_unmeasurable_dispatch_is_returned_as_none_and_not_cached(self, cache_dir, monkeypatch):
        monkeypatch.setattr(calibration, "measure", lambda device: self._record(None))
        result = calibration.get("cpu", refresh=True)
        assert result is not None
        assert result.dispatch_us is None
        assert not cache_dir.exists() or not list(cache_dir.glob("*.json"))

    def test_measured_dispatch_is_cached_and_round_trips(self, cache_dir, monkeypatch):
        monkeypatch.setattr(calibration, "measure", lambda device: self._record(5.9))
        first = calibration.get("cpu", refresh=True)
        assert first.dispatch_us == pytest.approx(5.9)
        files = list(cache_dir.glob(f"v{calibration.CACHE_VERSION}-*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text())["dispatch_us"] == pytest.approx(5.9)

        monkeypatch.setattr(calibration, "measure", lambda device: pytest.fail("cache not used"))
        assert calibration.get("cpu") == first

    def test_a_none_in_the_cache_is_never_read_as_zero(self, cache_dir, monkeypatch):
        monkeypatch.setattr(calibration, "measure", lambda device: self._record(4.0))
        calibration.get("cpu", refresh=True)
        (path,) = cache_dir.glob("*.json")
        path.write_text(json.dumps({**json.loads(path.read_text()), "dispatch_us": None}))
        cached = calibration.get("cpu")
        assert cached.dispatch_us is None

    def test_v1_records_are_not_read(self, cache_dir, monkeypatch):
        # v1 stored an unmeasurable cost as 0.0. A record like that must not be picked up.
        monkeypatch.setattr(calibration, "measure", lambda device: self._record(4.0))
        result = calibration.get("cpu", refresh=True)
        (path,) = cache_dir.glob("*.json")
        stale = path.with_name(path.name.replace(f"v{calibration.CACHE_VERSION}-", "v1-"))
        stale.write_text(json.dumps({**json.loads(path.read_text()), "dispatch_us": 0.0}))
        path.unlink()
        monkeypatch.setattr(calibration, "measure", lambda device: replace(result, dispatch_us=7.0))
        assert calibration.get("cpu").dispatch_us == pytest.approx(7.0)


class _FakeRunnable:
    def __init__(self, dps):
        self.metadata = SimpleNamespace(destination_passing_style=dps)
        self.calls = 0
        self._callable = self._kernel

    def _kernel(self, *args):
        self.calls += 1
        return ("real",)

    def _allocate_output_tensors(self, *inputs):
        return ["out"]

    def __call__(self, *args):
        return self._callable(*args)


class TestReachingIntoTheRuntimeIsUndone:
    def test_kernel_stub_is_restored_and_bench_call_has_the_bench_arity(self):
        r = _FakeRunnable(dps=True)
        with calibration._kernel_stubbed(r, ("x", "w")) as bench_call:
            assert bench_call() is None  # the stub, not the kernel
            assert r.calls == 0
        assert r._callable == r._kernel
        assert r("x", "w", "out") == ("real",) and r.calls == 1

    def test_value_returning_stub_returns_what_the_kernel_returned(self):
        r = _FakeRunnable(dps=False)
        with calibration._kernel_stubbed(r, ("x", "w")) as bench_call:
            assert r.calls == 1  # run once to learn the return value
            assert bench_call() == ("real",)
            assert r.calls == 1
        assert r._callable == r._kernel

    def test_kernel_stub_is_restored_when_the_block_raises(self):
        r = _FakeRunnable(dps=True)
        with pytest.raises(RuntimeError):
            with calibration._kernel_stubbed(r, ("x",)):
                raise RuntimeError("boom")
        assert r._callable == r._kernel

    def test_dispatched_runnable_observes_then_restores_try_build(self):
        class Runtime:
            def _try_build(self, definition, solution):
                return "runnable"

        rt = Runtime()
        original = Runtime._try_build
        seen = calibration._dispatched_runnable(rt, lambda: rt._try_build("d", "s"))
        assert seen == "runnable"
        assert "_try_build" not in vars(rt) and Runtime._try_build is original

    def test_dispatched_runnable_is_none_on_a_miss(self):
        class Runtime:
            def _try_build(self, definition, solution):
                return None

        rt = Runtime()
        assert calibration._dispatched_runnable(rt, lambda: calibration._MISS) is None
        assert "_try_build" not in vars(rt)
