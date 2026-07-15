"""CLI commands via Typer's CliRunner."""

from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from qf_agentos.cli import app
from qf_test_utils import make_spec

runner = CliRunner()


@pytest.fixture
def spec_file(tmp_path):
    def _write(name="spec.yaml", **kwargs):
        spec = make_spec(**kwargs)
        p = tmp_path / name
        p.write_text(yaml.safe_dump(spec.model_dump(mode="json")))
        return p

    return _write


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "QF-AgentOS" in result.stdout


def test_skills():
    result = runner.invoke(app, ["skills"])
    assert result.exit_code == 0
    assert "collateral-optimizer" in result.stdout


def test_explain(spec_file):
    result = runner.invoke(app, ["explain", str(spec_file())])
    assert result.exit_code == 0
    assert "binary_milp" in result.stdout


def test_plan(spec_file):
    result = runner.invoke(app, ["plan", str(spec_file())])
    assert result.exit_code == 0
    assert "instance" in result.stdout.lower()


def test_solve_writes_evidence(spec_file, tmp_path):
    spec = spec_file(allow_gate_model=False)  # skip QAOA for speed
    out = tmp_path / "evidence"
    result = runner.invoke(app, ["solve", str(spec), "--out", str(out), "--quiet"])
    assert result.exit_code == 0
    assert "FINAL DECISION" in result.stdout
    runs = list(out.glob("run-*"))
    assert runs and (runs[0] / "manifest.json").exists()


def test_solve_bad_spec_exit_code(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("problem: [unclosed")
    result = runner.invoke(app, ["solve", str(bad)])
    assert result.exit_code == 2  # SpecError.exit_code


def test_solve_infeasible_exit_code(spec_file, tmp_path):
    spec = spec_file(required=10**12, allow_gate_model=False)
    result = runner.invoke(app, ["solve", str(spec), "--out", str(tmp_path / "ev"), "--quiet"])
    assert result.exit_code == 4  # InfeasibleProblemError.exit_code


def test_runs_command(spec_file, tmp_path):
    out = tmp_path / "ev"
    spec = spec_file(allow_gate_model=False)
    runner.invoke(app, ["solve", str(spec), "--out", str(out), "--quiet"])
    result = runner.invoke(app, ["runs", "--out", str(out)])
    assert result.exit_code == 0
    assert "run-" in result.stdout
