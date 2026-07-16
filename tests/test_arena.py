"""QF-Arena benchmark: runner, leaderboard, and the honest no-advantage headline."""

from __future__ import annotations

import pytest

from qf_agentos.arena import load_example_specs, render_leaderboard, run_arena
from qf_test_utils import make_fraud_spec, make_spec


def test_arena_runs_and_scores_each_problem():
    specs = {
        "collateral": make_spec(required=4_000_000, allow_gate_model=False),
        "fraud": make_fraud_spec(allow_gate_model=False),
    }
    result = run_arena(specs)
    assert result.n == 2
    names = {e.name for e in result.entries}
    assert names == {"collateral", "fraud"}
    tasks = {e.problem: e.task_type for e in result.entries}
    assert tasks["collateral_allocation"] == "optimization"
    assert tasks["fraud_detection"] == "classification"
    for e in result.entries:
        assert e.verdict
        assert e.classical_score is not None  # a classical baseline always runs
    # Counts partition the entries.
    assert result.n_advantage + result.n_parity + result.n_classical == result.n


def test_leaderboard_renders_markdown():
    specs = {"collateral": make_spec(required=4_000_000, allow_gate_model=False)}
    md = render_leaderboard(run_arena(specs))
    assert "# QF-Arena leaderboard" in md
    assert "collateral_allocation" in md
    assert "quantum helps" in md


def test_load_example_specs_returns_the_bundled_suite():
    specs = load_example_specs()
    assert len(specs) >= 5
    problems = {s.problem for s in specs.values()}
    assert {"collateral_allocation", "fraud_detection", "settlement_netting"} <= problems


@pytest.mark.slow
def test_full_suite_has_no_quantum_advantage():
    # The honest headline: across the whole shipped suite, quantum never beats the
    # exact classical comparator (parity at best). A regression here would mean the
    # platform started over-claiming.
    result = run_arena(load_example_specs())
    assert result.n >= 5
    assert result.n_advantage == 0
    assert all(e.verdict for e in result.entries)
