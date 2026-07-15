"""Settlement / liquidity-saving problem family: IR, MILP, slack-encoded QUBO, pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from qf_agentos import solve
from qf_agentos.core.errors import SpecError
from qf_agentos.core.ir import parse_spec
from qf_agentos.core.result import Allocation, DecisionCategory
from qf_agentos.finance import get_domain
from qf_agentos.finance.settlement import (
    bits_to_settlement_allocation,
    build_settlement_qubo,
    check_settlement_constraints,
    reduce_to_settlement_instance,
    solve_settlement_milp,
)
from qf_test_utils import make_settlement_spec

# --- IR validation --------------------------------------------------------


def test_settlement_spec_validates():
    spec = make_settlement_spec()
    assert spec.problem == "settlement_netting"
    assert len(spec.participants) == 4 and len(spec.obligations) == 4


def test_missing_settlement_block_rejected():
    with pytest.raises(SpecError):
        parse_spec(
            {
                "problem": "settlement_netting",
                "participants": [{"id": "A", "balance": 1}],
                "obligations": [{"id": "O", "payer": "A", "payee": "B", "amount": 1}],
            }
        )


def test_unknown_payer_or_payee_rejected():
    with pytest.raises(SpecError):
        parse_spec(
            {
                "problem": "settlement_netting",
                "settlement": {},
                "participants": [{"id": "A", "balance": 1}],
                "obligations": [{"id": "O", "payer": "A", "payee": "GHOST", "amount": 1}],
            }
        )


def test_same_payer_and_payee_rejected():
    with pytest.raises(SpecError):
        parse_spec(
            {
                "problem": "settlement_netting",
                "settlement": {},
                "participants": [{"id": "A", "balance": 1}],
                "obligations": [{"id": "O", "payer": "A", "payee": "A", "amount": 1}],
            }
        )


def test_duplicate_obligation_ids_rejected():
    with pytest.raises(SpecError):
        parse_spec(
            {
                "problem": "settlement_netting",
                "settlement": {},
                "participants": [{"id": "A", "balance": 1}, {"id": "B", "balance": 1}],
                "obligations": [
                    {"id": "O", "payer": "A", "payee": "B", "amount": 1},
                    {"id": "O", "payer": "B", "payee": "A", "amount": 1},
                ],
            }
        )


# --- MILP -----------------------------------------------------------------


def test_milp_resolves_the_gridlock_cycle():
    # Each cycle bank has only 10 liquidity but owes 100; the batch settles the
    # cycle anyway because inflow funds outflow simultaneously.
    spec = make_settlement_spec()
    out = solve_settlement_milp(spec.obligations, spec.participants, None, integer=True)
    assert out.feasible
    assert {"O_AB", "O_BC", "O_CA"} <= set(out.allocation.x)
    assert out.objective == pytest.approx(0.0)  # everything settles → nothing unsettled


def test_lp_is_upper_bound_on_settled_value():
    # LP relaxation settles at least as much value as the binary MILP,
    # so its *unsettled* objective is <= the MILP's.
    spec = make_settlement_spec()
    lp = solve_settlement_milp(spec.obligations, spec.participants, None, integer=False)
    mp = solve_settlement_milp(spec.obligations, spec.participants, None, integer=True)
    assert lp.feasible and mp.feasible
    assert lp.objective <= mp.objective + 1e-6


def test_unfundable_obligation_stays_unsettled():
    # E owes F 100 but has no liquidity and no incoming payment → cannot settle.
    participants = [{"id": "E", "balance": 0}, {"id": "F", "balance": 0}]
    obligations = [{"id": "O_EF", "payer": "E", "payee": "F", "amount": 100}]
    spec = make_settlement_spec(participants=participants, obligations=obligations, max_qubits=6)
    out = solve_settlement_milp(spec.obligations, spec.participants, None, integer=True)
    assert out.feasible  # settling nothing is always feasible
    assert out.allocation.x == {}
    assert out.objective == pytest.approx(100.0)  # the whole 100 is unsettled


def test_min_settled_ratio_can_make_it_infeasible():
    participants = [{"id": "E", "balance": 0}, {"id": "F", "balance": 0}]
    obligations = [{"id": "O_EF", "payer": "E", "payee": "F", "amount": 100}]
    # Forcing 100% settled is impossible when the only obligation is unfundable.
    spec = make_settlement_spec(
        participants=participants,
        obligations=obligations,
        settlement={"penalty_scale": 8.0, "min_settled_ratio": 1.0},
        max_qubits=6,
    )
    out = solve_settlement_milp(spec.obligations, spec.participants, 1.0, integer=True)
    assert not out.feasible


# --- Constraint checking --------------------------------------------------


def test_check_detects_liquidity_violation():
    # E pays 100 with no liquidity and no inflow → liquidity check fails.
    participants = [{"id": "E", "balance": 0}, {"id": "F", "balance": 0}]
    obligations = [{"id": "O_EF", "payer": "E", "payee": "F", "amount": 100}]
    spec = make_settlement_spec(participants=participants, obligations=obligations, max_qubits=6)
    alloc = Allocation(x={"O_EF": 1.0})
    feasible, unsettled, checks = check_settlement_constraints(
        spec.obligations, spec.participants, None, alloc
    )
    liq = next(c for c in checks if c.name == "liquidity")
    assert not liq.satisfied and not feasible
    assert unsettled == pytest.approx(0.0)  # everything was (wrongly) marked settled


# --- QUBO -----------------------------------------------------------------


def test_qubo_ground_state_respects_liquidity():
    # The whole point of the slack encoding: the QUBO optimum is liquidity-feasible.
    spec = make_settlement_spec(max_qubits=12)
    inst = reduce_to_settlement_instance(spec, 12)
    qubo = build_settlement_qubo(inst)
    from qf_agentos.backends.heuristic import brute_force_qubo

    bits, _, _ = brute_force_qubo(qubo)
    alloc = bits_to_settlement_allocation(inst, bits)
    feasible, _, _ = check_settlement_constraints(
        inst.obligations, inst.participants, inst.min_settled_ratio, alloc
    )
    assert feasible


def test_qubo_ground_state_settles_the_cycle():
    # With enough slack resolution the QUBO recovers the true optimum (settle all).
    spec = make_settlement_spec(max_qubits=16)
    inst = reduce_to_settlement_instance(spec, 16)
    qubo = build_settlement_qubo(inst)
    from qf_agentos.backends.heuristic import brute_force_qubo

    bits, _, _ = brute_force_qubo(qubo)
    alloc = bits_to_settlement_allocation(inst, bits)
    assert {"O_AB", "O_BC", "O_CA"} <= set(alloc.x)


def test_reduce_respects_qubit_budget():
    for mq in (6, 10, 12, 16):
        inst = reduce_to_settlement_instance(make_settlement_spec(max_qubits=mq), mq)
        qubo = build_settlement_qubo(inst)
        assert inst.n_qubits <= mq
        assert qubo.n <= mq
        assert len(inst.obligations) >= 1


def test_qubo_ising_energy_equivalence():
    spec = make_settlement_spec(max_qubits=10)
    inst = reduce_to_settlement_instance(spec, 10)
    qubo = build_settlement_qubo(inst)
    from qf_agentos.finance.collateral import qubo_energy, qubo_to_ising

    const, h, J = qubo_to_ising(qubo)
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.integers(0, 2, size=qubo.n)
        z = 1 - 2 * x
        ising = const + float(h @ z) + sum(c * z[i] * z[j] for (i, j), c in J.items())
        assert qubo_energy(qubo, x) == pytest.approx(ising, rel=1e-9, abs=1e-9)


# --- Pipeline reuse -------------------------------------------------------


def test_domain_registered():
    assert get_domain("settlement_netting").problem == "settlement_netting"


def test_pipeline_runs_on_settlement_without_quantum():
    spec = make_settlement_spec(allow_gate_model=False)
    ctx = solve(spec)
    assert ctx.state.classical_milp.feasible
    assert ctx.state.bundle is not None
    assert not ctx.state.errors
    assert ctx.state.audit.category == DecisionCategory.QUANTUM_NOT_FEASIBLE


@pytest.mark.slow
def test_pipeline_runs_on_settlement_with_qaoa():
    spec = make_settlement_spec(max_qubits=12, qaoa_reps=2)
    ctx = solve(spec)
    assert ctx.state.audit is not None
    qaoa = ctx.state.instance_qaoa
    inst_milp = ctx.state.instance_milp
    # QAOA can never beat the exact MILP optimum on a feasible instance solution.
    if qaoa and qaoa.feasible and qaoa.objective is not None and inst_milp.objective is not None:
        assert qaoa.objective >= inst_milp.objective - 1e-6


def test_determinism_on_settlement():
    spec = make_settlement_spec(allow_gate_model=False)
    a = solve(spec).state.bundle.manifest["evidence_digest"]
    b = solve(spec).state.bundle.manifest["evidence_digest"]
    assert a == b
