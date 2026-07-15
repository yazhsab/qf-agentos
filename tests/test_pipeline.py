"""End-to-end pipeline behaviour: decisions, determinism, robustness, policy."""

from __future__ import annotations

import pytest

from qf_agentos import solve
from qf_agentos.core.result import DecisionCategory
from qf_test_utils import make_spec


@pytest.mark.slow
def test_full_pipeline_produces_decision_and_bundle(small_spec):
    ctx = solve(small_spec)
    assert ctx.state.audit is not None and ctx.state.audit.rendered
    assert ctx.state.bundle is not None
    assert ctx.state.bundle.manifest["run_id"] == ctx.run_id
    assert ctx.state.classical_milp is not None and ctx.state.classical_milp.feasible
    assert "classical_milp" in ctx.state.verification
    assert not ctx.state.errors  # no step failed


@pytest.mark.slow
def test_determinism_same_digest(small_spec):
    a = solve(small_spec)
    b = solve(small_spec)
    assert a.state.reproducibility.evidence_digest == b.state.reproducibility.evidence_digest
    assert a.state.audit.category == b.state.audit.category
    assert a.state.audit.objective_gap_pct == b.state.audit.objective_gap_pct


def test_infeasible_problem_flagged(spec_factory):
    spec = spec_factory(required=10**12, allow_gate_model=False)
    ctx = solve(spec)
    audit = ctx.state.audit
    assert audit.problem_infeasible
    assert audit.category == DecisionCategory.CLASSICAL_PREFERRED
    assert "infeasible" in audit.recommended_method.lower()


def test_empty_inventory_does_not_crash():
    spec = make_spec(inventory=[], required=1_000_000, allow_gate_model=False)
    ctx = solve(spec)
    assert ctx.state.bundle is not None
    assert not ctx.state.errors  # graceful, not a crash


def test_abstention_when_gate_model_disabled(spec_factory):
    spec = spec_factory(required=4_000_000, allow_gate_model=False)
    ctx = solve(spec)
    plan = ctx.state.hardware_plan
    assert plan.abstain
    assert ctx.state.instance_qaoa is None
    assert ctx.state.audit.category == DecisionCategory.QUANTUM_NOT_FEASIBLE


def test_policy_blocks_simulator_below_l2(spec_factory):
    # At L1, RUN_SIMULATOR is not authorised, so QAOA must not execute.
    spec = spec_factory(required=4_000_000, autonomy="L1")
    ctx = solve(spec)
    assert ctx.state.instance_qaoa is None
    assert any("QAOA not executed" in w for w in ctx.warnings)


@pytest.mark.slow
def test_quantum_never_silently_beats_exact(small_spec):
    ctx = solve(small_spec)
    qaoa = ctx.state.instance_qaoa
    inst_milp = ctx.state.instance_milp
    if qaoa and qaoa.feasible and qaoa.objective is not None and inst_milp.objective is not None:
        assert qaoa.objective >= inst_milp.objective - 1e-6


@pytest.mark.slow
def test_noisy_simulation_completes_the_ladder(spec_factory):
    spec = spec_factory(required=4_000_000)
    spec.execution_policy.noisy_simulation = True
    ctx = solve(spec)
    assert ctx.state.instance_qaoa_noisy is not None
    assert "qaoa_noisy_sim" in ctx.state.verification
    assert ctx.state.instance_qaoa_noisy.metadata.get("noise_model") is not None
    assert "qaoa_noisy_sim" in ctx.state.bundle.manifest["results"]
    assert not ctx.state.errors
