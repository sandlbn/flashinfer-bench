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

    def _record(self, dispatch, launch=2.0, peak=None, period=5120):
        return calibration.Calibration(
            hardware_id="CPU",
            timer="wall",
            dispatch_us=dispatch,
            timing_floor_us=1.0,
            bandwidth_gbs=1.0,
            launch_floor_us=launch,
            matmul_peak_tflops={"bfloat16": 10.0} if peak is None else peak,
            channel_period_bytes=period,
        )

    def test_unresolved_channel_period_is_returned_as_none_and_not_cached(
        self, cache_dir, monkeypatch
    ):
        monkeypatch.setattr(calibration, "measure", lambda device: self._record(5.0, period=None))
        result = calibration.get("cpu", refresh=True)
        assert result.channel_period_bytes is None and not result.complete
        assert not cache_dir.exists() or not list(cache_dir.glob("*.json"))

    def test_v3_records_lacking_the_period_are_not_read(self, cache_dir, monkeypatch):
        monkeypatch.setattr(calibration, "measure", lambda device: self._record(4.0))
        result = calibration.get("cpu", refresh=True)
        (path,) = cache_dir.glob("*.json")
        stale = path.with_name(path.name.replace(f"v{calibration.CACHE_VERSION}-", "v3-"))
        old = json.loads(path.read_text())
        old.pop("channel_period_bytes")
        stale.write_text(json.dumps(old))
        path.unlink()
        monkeypatch.setattr(calibration, "measure", lambda device: replace(result, dispatch_us=7.0))
        assert calibration.get("cpu").dispatch_us == pytest.approx(7.0)

    def test_unsettled_launch_floor_is_returned_as_none_and_not_cached(
        self, cache_dir, monkeypatch
    ):
        monkeypatch.setattr(calibration, "measure", lambda device: self._record(5.0, launch=None))
        result = calibration.get("cpu", refresh=True)
        assert result.launch_floor_us is None and not result.complete
        assert not cache_dir.exists() or not list(cache_dir.glob("*.json"))

    def test_unsettled_matmul_peak_is_returned_as_none_and_not_cached(self, cache_dir, monkeypatch):
        monkeypatch.setattr(
            calibration, "measure", lambda device: self._record(5.0, peak={"bfloat16": None})
        )
        result = calibration.get("cpu", refresh=True)
        assert result.matmul_peak_tflops == {"bfloat16": None} and not result.complete
        assert not cache_dir.exists() or not list(cache_dir.glob("*.json"))

    def test_new_fields_round_trip_through_the_cache(self, cache_dir, monkeypatch):
        monkeypatch.setattr(
            calibration,
            "measure",
            lambda device: self._record(5.0, launch=3.5, peak={"float16": 42.0}),
        )
        first = calibration.get("cpu", refresh=True)
        monkeypatch.setattr(calibration, "measure", lambda device: pytest.fail("cache not used"))
        again = calibration.get("cpu")
        assert again == first
        assert again.launch_floor_us == pytest.approx(3.5)
        assert again.matmul_peak_tflops == {"float16": pytest.approx(42.0)}

    def test_v2_records_lacking_the_new_fields_are_not_read(self, cache_dir, monkeypatch):
        monkeypatch.setattr(calibration, "measure", lambda device: self._record(4.0))
        result = calibration.get("cpu", refresh=True)
        (path,) = cache_dir.glob("*.json")
        stale = path.with_name(path.name.replace(f"v{calibration.CACHE_VERSION}-", "v2-"))
        old = json.loads(path.read_text())
        old.pop("launch_floor_us")
        old.pop("matmul_peak_tflops")
        stale.write_text(json.dumps(old))
        path.unlink()
        monkeypatch.setattr(calibration, "measure", lambda device: replace(result, dispatch_us=7.0))
        assert calibration.get("cpu").dispatch_us == pytest.approx(7.0)

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


class TestSettledDeviceMeasurement:
    """The warm-up is by time and ends when the window agrees; nothing is reported before."""

    def _clock(self, per_call_sequence):
        # A fake clock advanced by the fake kernel: each region's per-call cost comes from
        # the sequence, so the ramp a gated clock produces can be scripted.
        state = {"t": 0.0, "i": 0}
        costs = list(per_call_sequence)

        def fn():
            i = min(state["i"], len(costs) - 1)
            state["t"] += costs[i] * 1e-6
            state["i"] += 1

        return state, fn, (lambda: state["t"])

    def test_ramping_clock_is_not_reported_until_it_settles(self):
        # Ten calls per region: regions 1-3 ramp (idle clock), regions 4+ are steady at 5 us.
        per_call = [50.0] * 11 + [30.0] * 10 + [15.0] * 10 + [5.0] * 500
        state, fn, clock = self._clock(per_call)
        us = calibration._settled_device_us(
            fn, lambda: None, calls=10, window=5, budget_s=10.0, clock=clock
        )
        assert us == pytest.approx(5.0)
        assert state["i"] > 1 + 3 * 10  # the three ramping regions were consumed, not reported

    def test_never_settling_returns_none_not_the_last_sample(self):
        import itertools

        noisy = itertools.cycle([5.0, 9.0, 5.0, 12.0, 6.0, 11.0])
        state = {"t": 0.0}

        def fn():
            state["t"] += next(noisy) * 1e-6

        us = calibration._settled_device_us(
            fn, lambda: None, calls=1, window=5, budget_s=0.001, clock=lambda: state["t"]
        )
        assert us is None

    def test_steady_regions_report_their_median(self):
        _, fn, clock = self._clock([7.0] * 1000)
        assert calibration._settled_device_us(
            fn, lambda: None, calls=10, window=5, budget_s=1.0, clock=clock
        ) == pytest.approx(7.0)


class TestStridedReadArguments:
    def test_pattern_arguments_are_validated_before_any_device_is_touched(self):
        with pytest.raises(ValueError):
            calibration.strided_read_bandwidth_gbs("cpu", 0, 64)
        with pytest.raises(ValueError):
            calibration.strided_read_bandwidth_gbs("cpu", 128, 64)
        with pytest.raises(ValueError):
            calibration.strided_read_bandwidth_gbs("cpu", 3, 64)
        with pytest.raises(ValueError):
            calibration.strided_read_bandwidth_gbs("cpu", 64, calibration.PATTERN_BUFFER_BYTES * 2)

    def test_complete_requires_every_field(self):
        full = calibration.Calibration("p", "t", 1.0, 1.0, 1.0, 1.0, {"bfloat16": 1.0}, 4096)
        assert full.complete
        assert not replace(full, dispatch_us=None).complete
        assert not replace(full, launch_floor_us=None).complete
        assert not replace(full, matmul_peak_tflops={"bfloat16": None}).complete
        assert not replace(full, channel_period_bytes=None).complete
        assert replace(full, matmul_peak_tflops={}).complete


STEP = 256
GRID = list(range(2048, 32768 + 1, STEP))
"""A pitch grid like the sweep's. The times below are synthetic: a flat baseline with the
spikes the period detector is meant to read, in units of the baseline."""


def _sweep(spikes, baseline=None, unsettled=()):
    """`spikes` maps a pitch to its time in baseline units; everything else is baseline."""
    baseline = baseline or (lambda p: 1.0)
    return {p: None if p in unsettled else baseline(p) * spikes.get(p, 1.0) for p in GRID}


def _multiples(period, factor, start=1):
    return {p: factor for p in GRID if p % period == 0 and p >= start * period}


class TestChannelPeriodFromSweep:
    """The detector reads the spacing of the slow pitches and nothing else; it refuses a
    sweep whose slow pitches do not repeat at one spacing, and never invents one."""

    def test_reads_the_spacing_of_the_slow_pitches(self):
        assert calibration.channel_period_from_sweep(_sweep(_multiples(5120, 1.3))) == 5120

    def test_a_deeper_spike_at_twice_the_period_does_not_hide_it(self):
        # Every other multiple is much slower than the ones between; the period is still the
        # spacing of all of them, not of the deep ones.
        spikes = {**_multiples(5120, 1.3), **_multiples(10240, 2.5)}
        assert calibration.channel_period_from_sweep(_sweep(spikes)) == 5120

    def test_a_residual_at_a_fraction_of_the_period_is_not_slow(self):
        # A pitch at an odd number of half periods shows a few percent; that is not
        # camping, and counting it would halve the period.
        half_periods = {p: 1.10 for p in GRID if p % 2560 == 0 and p % 5120 != 0}
        spikes = {**_multiples(5120, 1.3), **half_periods}
        assert calibration.channel_period_from_sweep(_sweep(spikes)) == 5120

    def test_a_single_slow_pitch_is_no_period(self):
        assert calibration.channel_period_from_sweep(_sweep({20480: 1.5})) is None

    def test_unrelated_slow_pitches_are_refused(self):
        # Two slow pitches whose common divisor is itself a fast pitch: no periodic
        # structure, so no period -- not the divisor, and not the smaller of the two.
        assert calibration.channel_period_from_sweep(_sweep({6144: 1.4, 9216: 1.4})) is None

    def test_a_fast_multiple_of_the_candidate_is_refused(self):
        spikes = {5120: 1.3, 15360: 1.3, 20480: 1.3}  # 10240 measured and fast
        assert calibration.channel_period_from_sweep(_sweep(spikes)) is None

    def test_unsettled_pitches_are_skipped_not_read_as_fast(self):
        spikes = _multiples(5120, 1.3)
        times = _sweep(spikes, unsettled={10240, 20480, 3072, 3328})
        assert calibration.channel_period_from_sweep(times) == 5120

    def test_a_drift_across_the_sweep_is_not_a_slow_pitch(self):
        drift = lambda p: 1.0 + 0.6 * (p - GRID[0]) / (GRID[-1] - GRID[0])  # noqa: E731
        assert calibration.channel_period_from_sweep(_sweep({}, baseline=drift)) is None
        assert (
            calibration.channel_period_from_sweep(_sweep(_multiples(5120, 1.3), baseline=drift))
            == 5120
        )

    def test_a_flat_sweep_has_no_period(self):
        assert calibration.channel_period_from_sweep(_sweep({})) is None

    def test_a_sweep_that_barely_settled_anywhere_has_no_period(self):
        times = {p: (1.3 if p % 5120 == 0 else 1.0) for p in GRID[:6]}
        assert calibration.channel_period_from_sweep(times) is None

    def test_the_deficit_is_the_stated_one(self):
        just_under = 1.0 + calibration.CAMPING_DEFICIT - 0.01
        just_over = 1.0 + calibration.CAMPING_DEFICIT + 0.01
        assert calibration.channel_period_from_sweep(_sweep(_multiples(5120, just_under))) is None
        assert calibration.channel_period_from_sweep(_sweep(_multiples(5120, just_over))) == 5120


class TestAuthoredStreamProbe:
    """The authored-stream rate is a sidecar of the record: measured once per (part, timer,
    language), cached only when it measured, and never a number from anywhere else."""

    @pytest.fixture
    def cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FIB_CACHE_PATH", str(tmp_path))
        return tmp_path / "calibration"

    RECORD = {"gbs": 400.0, "language": "triton", "config": {"block": 2048}, "triton": "x"}

    def test_an_unmeasurable_probe_is_none_and_leaves_no_sidecar(self, cache_dir, monkeypatch):
        monkeypatch.setattr(calibration, "_measure_authored_stream", lambda device: None)
        assert calibration.authored_stream_probe("cpu") is None
        assert calibration.authored_stream_gbs("cpu") is None
        assert not cache_dir.exists() or not list(cache_dir.glob("*-authored.json"))

    def test_a_measured_probe_is_cached_and_round_trips(self, cache_dir, monkeypatch):
        monkeypatch.setattr(calibration, "_measure_authored_stream", lambda device: self.RECORD)
        assert calibration.authored_stream_probe("cpu") == self.RECORD
        (path,) = cache_dir.glob(f"v{calibration.CACHE_VERSION}-*-authored.json")
        assert json.loads(path.read_text())["triton"]["gbs"] == 400.0
        monkeypatch.setattr(
            calibration, "_measure_authored_stream", lambda device: pytest.fail("cache not used")
        )
        assert calibration.authored_stream_gbs("cpu") == pytest.approx(400.0)
        assert calibration.authored_stream_probe("cpu")["config"] == {"block": 2048}

    def test_refresh_measures_again(self, cache_dir, monkeypatch):
        monkeypatch.setattr(calibration, "_measure_authored_stream", lambda device: self.RECORD)
        calibration.authored_stream_probe("cpu")
        monkeypatch.setattr(
            calibration, "_measure_authored_stream", lambda device: {**self.RECORD, "gbs": 410.0}
        )
        assert calibration.authored_stream_gbs("cpu", refresh=True) == pytest.approx(410.0)
        assert calibration.authored_stream_gbs("cpu") == pytest.approx(410.0)

    def test_a_failing_probe_is_none_not_an_exception(self, cache_dir, monkeypatch):
        def boom(device):
            raise RuntimeError("no backend")

        monkeypatch.setattr(calibration, "_measure_authored_stream", boom)
        assert calibration.authored_stream_probe("cpu") is None
