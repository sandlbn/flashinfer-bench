"""The command PROVENANCE.md tells the reader to run next must be one the CLI accepts."""

import importlib.util
import pathlib
import re
import sys

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_provenance_next_step_parses_against_the_real_cli():
    pks = _load("pull_kernel_source")
    kt = _load("kernel_trials")
    text = pks._PROVENANCE.format(
        op="_C::rms_norm",
        series="_C_rms_norm",
        device="xpu:0",
        keys="XPU",
        registered="x.so",
        project="p",
        schema="s",
        files="f",
    )
    commands = re.findall(r"`python scripts/kernel_trials\.py ([^`]*)`", text)
    assert commands, text
    for command in commands:
        args = kt.build_parser().parse_args(command.split())
        assert args.cmd == "init"
        assert args.name == "_C_rms_norm"
        assert args.baseline.endswith("harness.py")
