"""A solution that cannot be built here must fall back, not raise.

A trace dataset is shared across hardware, so most of its solutions target something
else: on an Intel GPU every CUDA solution fails to build, and some fail on a missing
Python dependency rather than a missing device. ``apply()`` takes a ``fallback`` precisely
so the caller keeps working when the fast path is unavailable -- letting a BuildError
escape defeats that.

This is not hypothetical. Enabling apply inside vLLM killed the EngineCore at startup:
one `cublaslt_fp4_e2m1_scaled_mm` solution in the dataset needs `torchao`, which is not
installed on an Intel box, and the import error propagated out of `apply()` and took down
the server.
"""

import importlib
import types

import pytest

from flashinfer_bench.compile.builder import BuildError


class _Boom:
    """A builder registry whose build always fails, as a foreign solution would."""

    def __init__(self, exc):
        self._exc = exc

    def build(self, definition, solution):
        raise self._exc


@pytest.fixture
def runtime_with_failing_builder(monkeypatch):
    from flashinfer_bench.apply import runtime as rt

    made = []

    class _Registry:
        @staticmethod
        def get_instance():
            return made[0]

    monkeypatch.setattr(rt, "BuilderRegistry", _Registry)
    return rt, made


@pytest.mark.parametrize(
    "exc",
    [
        BuildError("no compiler for this device"),
        ModuleNotFoundError("No module named 'torchao'"),
        RuntimeError("something else entirely"),
    ],
)
def test_unbuildable_solution_returns_none(runtime_with_failing_builder, exc):
    """Any build failure is a miss, not an exception -- whatever its type."""
    rt, made = runtime_with_failing_builder
    made.append(_Boom(exc))

    instance = rt.ApplyRuntime.__new__(rt.ApplyRuntime)
    instance._unbuildable = set()

    definition = type("D", (), {"name": "some_def"})()
    solution = type("S", (), {"name": "some_sol"})()
    assert instance._try_build(definition, solution) is None


def test_each_unbuildable_solution_is_reported_once(runtime_with_failing_builder, caplog):
    """Otherwise every call re-reports it and the log drowns the process."""
    rt, made = runtime_with_failing_builder
    made.append(_Boom(BuildError("nope")))

    instance = rt.ApplyRuntime.__new__(rt.ApplyRuntime)
    instance._unbuildable = set()
    definition = type("D", (), {"name": "d"})()
    solution = type("S", (), {"name": "s"})()

    with caplog.at_level("WARNING"):
        for _ in range(5):
            instance._try_build(definition, solution)
    assert sum("cannot be built here" in r.message for r in caplog.records) == 1
    assert "s" in instance._unbuildable


def _local_solution(name):
    """A solution declaring whatever backend this host actually runs on."""
    from flashinfer_bench.device import default_device_type

    return types.SimpleNamespace(
        name=name, spec=types.SimpleNamespace(target_hardware=[default_device_type()])
    )


class TestWarmUpNeverRaises:
    """Warm-up runs at table construction, inside a vLLM EngineCore's startup.

    Anything that escapes here kills the engine before it serves a request, so the
    handler is exercised directly rather than through a path that might not reach it.
    An earlier version of this handler referenced an undefined ``logger`` and turned
    every build failure into a ``NameError`` that propagated exactly as the original
    exception would have.
    """

    @pytest.mark.parametrize(
        "exc",
        [
            ModuleNotFoundError("No module named 'torchao'"),
            RuntimeError("BuildError: failed importing generated module"),
            KeyboardInterrupt(),
        ],
        ids=["missing-dependency", "build-error", "non-Exception"],
    )
    def test_a_failing_build_does_not_escape(self, exc):
        table = importlib.import_module("flashinfer_bench.apply.table")

        class Failing:
            def build(self, definition, solution):
                raise exc

        definition = types.SimpleNamespace(name="fp4_scaled_mm")
        solution = _local_solution("cublaslt")

        if isinstance(exc, Exception):
            table._warm(Failing(), definition, solution)
        else:
            # KeyboardInterrupt is not an Exception and must still reach the operator.
            with pytest.raises(KeyboardInterrupt):
                table._warm(Failing(), definition, solution)

    def test_a_successful_build_is_still_performed(self):
        table = importlib.import_module("flashinfer_bench.apply.table")
        built = []

        class Working:
            def build(self, definition, solution):
                built.append((definition.name, solution.name))

        table._warm(Working(), types.SimpleNamespace(name="d"), _local_solution("s"))
        assert built == [("d", "s")], "warm-up must not have been skipped entirely"


class TestWarmUpSkipsOtherHardware:
    """A solution for another backend must not even be handed to the builder.

    This is not the same failure as an unbuildable solution. `cpp_extension.load` guards
    its build directory with a `FileBaton` that polls a lock file with no timeout, so a
    baton orphaned by a killed process hangs the next builder forever -- and a hang is
    invisible to the `except` clause above. The only defence is not to start the build.
    """

    def _solution(self, targets):
        return types.SimpleNamespace(
            name="cuda_only", spec=types.SimpleNamespace(target_hardware=targets)
        )

    def test_a_cuda_solution_is_not_built_on_an_xpu_host(self, monkeypatch):
        table = importlib.import_module("flashinfer_bench.apply.table")
        monkeypatch.setattr(
            "flashinfer_bench.device.default_device_type", lambda: "xpu", raising=False
        )
        attempted = []

        class Registry:
            def build(self, definition, solution):
                attempted.append(solution.name)

        table._warm(Registry(), types.SimpleNamespace(name="d"), self._solution(["cuda"]))
        assert attempted == [], "a CUDA solution was handed to the builder on an XPU host"

    def test_a_matching_solution_is_still_built(self, monkeypatch):
        table = importlib.import_module("flashinfer_bench.apply.table")
        monkeypatch.setattr(
            "flashinfer_bench.device.default_device_type", lambda: "xpu", raising=False
        )
        attempted = []

        class Registry:
            def build(self, definition, solution):
                attempted.append(solution.name)

        table._warm(Registry(), types.SimpleNamespace(name="d"), self._solution(["xpu"]))
        assert attempted == ["cuda_only"], "warm-up must still build what does match"

    @pytest.mark.parametrize("targets", [["XPU"], ["cuda", "xpu"]])
    def test_matching_is_case_insensitive_and_accepts_multiple_targets(self, monkeypatch, targets):
        table = importlib.import_module("flashinfer_bench.apply.table")
        monkeypatch.setattr(
            "flashinfer_bench.device.default_device_type", lambda: "xpu", raising=False
        )
        built = []

        class Registry:
            def build(self, definition, solution):
                built.append(solution.name)

        table._warm(Registry(), types.SimpleNamespace(name="d"), self._solution(targets))
        assert built == ["cuda_only"]

    def test_an_undeclared_target_is_skipped_rather_than_guessed(self, monkeypatch):
        table = importlib.import_module("flashinfer_bench.apply.table")
        monkeypatch.setattr(
            "flashinfer_bench.device.default_device_type", lambda: "xpu", raising=False
        )
        attempted = []

        class Registry:
            def build(self, definition, solution):
                attempted.append(solution.name)

        table._warm(Registry(), types.SimpleNamespace(name="d"), self._solution([]))
        assert attempted == []


class TestDtypeMismatchNeverDispatches:
    """A definition names a shape, not a dtype, so the lookup alone cannot keep them apart.

    `rmsnorm_h1024` extracted from an fp16 HuggingFace run and `rmsnorm_h1024` as vLLM
    serves it in bf16 are the same name. The apply index keys on axes only and
    `use_def_best` skips the key, so the fp16 solution was selected for a bf16 model: it
    returned fp16 activations and the next matmul died with
    "expected mat1 and mat2 to have the same dtype". Wrong dtype is a wrong kernel, and
    the caller's own implementation is the right answer.
    """

    def _definition(self, dtype):
        import torch

        return types.SimpleNamespace(
            name="rmsnorm_h1024",
            inputs={"hidden_states": None, "weight": None},
            outputs={"output": None},
            torch_input_dtypes=lambda: [dtype, dtype],
        )

    def test_mismatched_dtype_falls_back(self):
        import torch

        from flashinfer_bench.apply.runtime import _dtypes_match

        d = self._definition(torch.float16)
        args = (torch.zeros(4, 8, dtype=torch.bfloat16), torch.zeros(8, dtype=torch.bfloat16))
        assert not _dtypes_match(d, args), "an fp16 definition accepted bf16 tensors"

    def test_matching_dtype_dispatches(self):
        import torch

        from flashinfer_bench.apply.runtime import _dtypes_match

        d = self._definition(torch.bfloat16)
        args = (torch.zeros(4, 8, dtype=torch.bfloat16), torch.zeros(8, dtype=torch.bfloat16))
        assert _dtypes_match(d, args)

    def test_a_non_tensor_argument_is_not_treated_as_a_mismatch(self):
        import torch

        from flashinfer_bench.apply.runtime import _dtypes_match

        d = self._definition(torch.bfloat16)
        args = (torch.zeros(4, 8, dtype=torch.bfloat16), 1e-6)
        assert _dtypes_match(d, args), "a scalar argument must not block dispatch"


class TestDtypeGuardAgainstARealDefinition:
    """The guard must work on a real Definition, not only on a test double.

    `torch_input_dtypes` is a `cached_property`. The first version of the guard called it
    as a method, raised TypeError, and swallowed that in a broad `except` -- so it was a
    no-op against every real definition while passing tests that mocked it as callable.
    Exercising the real class is the only thing that catches that.
    """

    def _definition(self, dtype: str):
        from flashinfer_bench.data import Definition

        return Definition.model_validate(
            {
                "name": f"rmsnorm_h8_{dtype}",
                "description": "d",
                "op_type": "rmsnorm",
                "axes": {
                    "batch_size": {"type": "var"},
                    "hidden_size": {"type": "const", "value": 8},
                },
                "inputs": {
                    "hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": dtype},
                    "weight": {"shape": ["hidden_size"], "dtype": dtype},
                },
                "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": dtype}},
                "reference": "import torch\n\ndef run(hidden_states, weight):\n    return hidden_states\n",
            }
        )

    def test_the_property_is_read_not_called(self):
        d = self._definition("bfloat16")
        assert not callable(d.torch_input_dtypes), "assumption changed; guard must follow"

    def test_bf16_tensors_are_rejected_by_an_fp16_definition(self):
        import torch

        from flashinfer_bench.apply.runtime import _dtypes_match

        d = self._definition("float16")
        args = (torch.zeros(2, 8, dtype=torch.bfloat16), torch.zeros(8, dtype=torch.bfloat16))
        assert not _dtypes_match(d, args), "guard is a no-op against a real Definition"

    def test_matching_tensors_are_accepted(self):
        import torch

        from flashinfer_bench.apply.runtime import _dtypes_match

        d = self._definition("bfloat16")
        args = (torch.zeros(2, 8, dtype=torch.bfloat16), torch.zeros(8, dtype=torch.bfloat16))
        assert _dtypes_match(d, args)
