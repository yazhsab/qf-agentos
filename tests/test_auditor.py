"""Auditor decision categories — the honest-classification logic.

The auditor is exercised directly on hand-crafted pipeline states so every
decision branch is covered deterministically."""

from __future__ import annotations

from qf_agentos.agents.auditor import auditor_agent
from qf_agentos.core.artifacts import HardwarePlan
from qf_agentos.core.policy import PolicyEngine
from qf_agentos.core.result import (
    Allocation,
    AuditDecision,
    DecisionCategory,
    SolveResult,
    VerificationReport,
)
from qf_agentos.core.workflow import RunContext
from qf_test_utils import make_spec


def _result(method, kind, feasible, objective):
    return SolveResult(
        method=method,
        kind=kind,
        backend="x",
        scope="research_instance",
        feasible=feasible,
        objective=objective,
        allocation=Allocation(x={}),
    )


def _report(method, feasible, objective, *, contributed=None):
    rep = VerificationReport(
        method=method, scope="research_instance", feasible=feasible, recomputed_objective=objective
    )
    if contributed is not None:
        rep.quantum_contribution = {"contributed": contributed, "reached_ground_state": True}
    return rep


def _ctx(
    *,
    abstain=False,
    full_feasible=True,
    c_obj=100.0,
    c_feasible=True,
    q_obj=None,
    q_feasible=None,
    contributed=True,
    autonomy="L2",
    approved=False,
) -> RunContext:
    spec = make_spec(autonomy=autonomy)
    ctx = RunContext(
        spec=spec,
        policy=PolicyEngine(spec.execution_policy, human_approved=approved),
        run_id="test",
        seed=7,
    )
    ctx.state.classical_milp = _result(
        "classical_milp", "classical", full_feasible, 50.0 if full_feasible else None
    )
    if not full_feasible:
        ctx.state.classical_milp.metadata = {"status": "infeasible"}
    ctx.state.hardware_plan = HardwarePlan(
        n_qubits=8,
        qubo_density=1.0,
        target=None if abstain else "gate_model_statevector_sim",
        abstain=abstain,
        reasons=["policy"] if abstain else [],
        estimated_two_qubit_depth=1,
        estimated_cost_usd=0.0,
        real_qpu="gated",
        instance_target=100.0,
    )
    if not abstain and q_obj is not None:
        ctx.state.instance_milp = _result("instance_milp", "classical", c_feasible, c_obj)
        ctx.state.instance_qaoa = _result("qaoa_sim", "quantum", q_feasible, q_obj)
        ctx.state.verification["instance_milp"] = _report("instance_milp", c_feasible, c_obj)
        ctx.state.verification["qaoa_sim"] = _report(
            "qaoa_sim", q_feasible, q_obj, contributed=contributed
        )
    return ctx


def _decide(**kw) -> AuditDecision:
    ctx = _ctx(**kw)
    auditor_agent(ctx)
    return ctx.state.audit


def test_problem_infeasible():
    d = _decide(full_feasible=False)
    assert d.problem_infeasible
    assert d.category == DecisionCategory.CLASSICAL_PREFERRED


def test_abstention():
    d = _decide(abstain=True)
    assert d.category == DecisionCategory.QUANTUM_NOT_FEASIBLE


def test_quantum_infeasible_prefers_classical():
    d = _decide(q_obj=100.0, q_feasible=False)
    assert d.category == DecisionCategory.CLASSICAL_PREFERRED


def test_quantum_parity():
    d = _decide(c_obj=100.0, q_obj=100.0, q_feasible=True)
    assert d.category == DecisionCategory.QUANTUM_PARITY
    assert d.objective_gap_pct == 0.0


def test_quantum_worse_prefers_classical():
    d = _decide(c_obj=100.0, q_obj=120.0, q_feasible=True)
    assert d.category == DecisionCategory.CLASSICAL_PREFERRED
    assert d.objective_gap_pct and d.objective_gap_pct > 0


def test_quantum_improvement_flags_reproduction():
    d = _decide(c_obj=100.0, q_obj=90.0, q_feasible=True)
    assert d.category == DecisionCategory.QUANTUM_IMPROVEMENT_OBSERVED
    assert "reproduction" in d.recommended_method.lower()


def test_research_candidate_when_classical_missing():
    d = _decide(c_obj=None, c_feasible=False, q_obj=80.0, q_feasible=True)
    assert d.category == DecisionCategory.QUANTUM_RESEARCH_CANDIDATE


def test_production_gate_blocked_without_l4():
    d = _decide(q_obj=100.0, q_feasible=True, autonomy="L2")
    assert any("Production sign-off: BLOCKED" in r for r in d.rationale)


def test_production_gate_authorised_at_l4():
    d = _decide(q_obj=100.0, q_feasible=True, autonomy="L4", approved=True)
    assert any("Production sign-off: authorised" in r for r in d.rationale)


def test_zero_cost_tie_is_parity_not_classical_preferred():
    # Regression: c_obj == 0.0 must not be treated as falsy (was misclassified).
    d = _decide(c_obj=0.0, q_obj=0.0, q_feasible=True)
    assert d.category == DecisionCategory.QUANTUM_PARITY
    assert d.objective_gap_pct == 0.0


def test_qaoa_none_with_selected_target_is_not_called_abstention():
    # Regression: a policy-blocked QAOA must not be reported as "planner abstained".
    ctx = _ctx(q_obj=None)  # abstain=False → target selected, but no QAOA result
    ctx.warnings.append(
        "QAOA not executed: Autonomy L1 is below the L2 required for run_simulator."
    )
    auditor_agent(ctx)
    d = ctx.state.audit
    assert d.category == DecisionCategory.QUANTUM_NOT_FEASIBLE
    assert any("selected but produced no result" in r for r in d.rationale)
    assert any("QAOA not executed" in r for r in d.rationale)
    assert not any("abstained" in r.lower() for r in d.rationale)
