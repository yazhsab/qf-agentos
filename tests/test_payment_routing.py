"""Payment-routing problem family: IR, GAP MILP, QUBO, pipeline reuse."""

from __future__ import annotations

import numpy as np
import pytest

from qf_agentos import solve
from qf_agentos.core.errors import SpecError
from qf_agentos.core.ir import parse_spec
from qf_agentos.core.result import DecisionCategory
from qf_agentos.finance import get_domain
from qf_agentos.finance.payment_routing import (
    build_routing_qubo,
    check_routing_constraints,
    reduce_to_routing_instance,
    solve_routing_milp,
)
from qf_test_utils import make_routing_spec

# --- IR validation --------------------------------------------------------


def test_routing_spec_validates():
    spec = make_routing_spec()
    assert spec.problem == "payment_routing"
    assert len(spec.routes) == 4 and len(spec.transactions) == 6


def test_missing_routing_block_rejected():
    with pytest.raises(SpecError):
        parse_spec({"problem": "payment_routing", "routes": [{"id": "R", "approval_rate": 0.9}]})


def test_unknown_eligible_route_rejected():
    with pytest.raises(SpecError):
        parse_spec(
            {
                "problem": "payment_routing",
                "routing": {},
                "routes": [{"id": "R1", "approval_rate": 0.9}],
                "transactions": [{"id": "T1", "amount": 100, "eligible_routes": ["NOPE"]}],
            }
        )


def test_bad_approval_rate_rejected():
    with pytest.raises(SpecError):
        parse_spec(
            {
                "problem": "payment_routing",
                "routing": {},
                "routes": [{"id": "R1", "approval_rate": 1.5}],
            }
        )


# --- MILP (generalized assignment) ---------------------------------------


def test_milp_routes_every_transaction_once():
    spec = make_routing_spec()
    out = solve_routing_milp(spec.transactions, spec.routes, spec.routing, integer=True)
    assert out.feasible and out.objective is not None
    # exactly one assignment per transaction
    by_tx: dict[str, int] = {}
    for key in out.allocation.x:
        tx_id = key.split("=>", 1)[0]
        by_tx[tx_id] = by_tx.get(tx_id, 0) + 1
    assert all(v == 1 for v in by_tx.values())
    assert len(by_tx) == len(spec.transactions)


def test_milp_infeasible_when_capacity_too_small():
    # Total capacity across routes < number of transactions => infeasible.
    routes = [{"id": "R1", "approval_rate": 0.9, "capacity": 1, "cost_bps": 10}]
    spec = make_routing_spec(
        routes=routes,
        transactions=[{"id": "T1", "amount": 100}, {"id": "T2", "amount": 100}],
        max_qubits=8,
    )
    out = solve_routing_milp(spec.transactions, spec.routes, spec.routing, integer=True)
    assert not out.feasible


def test_lp_is_lower_bound_on_milp():
    spec = make_routing_spec()
    lp = solve_routing_milp(spec.transactions, spec.routes, spec.routing, integer=False)
    mp = solve_routing_milp(spec.transactions, spec.routes, spec.routing, integer=True)
    assert lp.feasible and mp.feasible
    assert lp.objective <= mp.objective + 1e-6


# --- Constraint checking --------------------------------------------------


def test_check_detects_capacity_violation():
    from qf_agentos.core.result import Allocation

    spec = make_routing_spec(
        routes=[{"id": "R1", "approval_rate": 0.9, "capacity": 1, "cost_bps": 5}],
        transactions=[{"id": "T1", "amount": 100}, {"id": "T2", "amount": 100}],
        routing={},
        max_qubits=8,
    )
    # Force both onto the single capacity-1 route.
    alloc = Allocation(x={"T1=>R1": 1.0, "T2=>R1": 1.0})
    feasible, _, checks = check_routing_constraints(
        spec.transactions, spec.routes, spec.routing, alloc
    )
    cap = next(c for c in checks if c.name == "capacity")
    assert not cap.satisfied and not feasible


# --- QUBO -----------------------------------------------------------------


def test_qubo_ground_state_is_a_valid_assignment():
    spec = make_routing_spec(max_qubits=12)
    inst = reduce_to_routing_instance(spec, 12)
    qubo = build_routing_qubo(inst)
    from qf_agentos.backends.heuristic import brute_force_qubo
    from qf_agentos.finance.payment_routing import bits_to_routing_allocation

    bits, _, _ = brute_force_qubo(qubo)
    alloc = bits_to_routing_allocation(inst, bits)
    # Each instance transaction assigned exactly one route at the QUBO optimum.
    counts: dict[str, int] = {}
    for key in alloc.x:
        counts[key.split("=>", 1)[0]] = counts.get(key.split("=>", 1)[0], 0) + 1
    assert all(v == 1 for v in counts.values())
    assert len(counts) == len(inst.transactions)


def test_reduce_respects_qubit_budget():
    spec = make_routing_spec(max_qubits=12)
    inst = reduce_to_routing_instance(spec, 12)
    assert inst.n_qubits <= 12
    assert len(inst.routes) >= 1 and len(inst.transactions) >= 1


# --- Pipeline reuse -------------------------------------------------------


def test_domain_registered():
    assert get_domain("payment_routing").problem == "payment_routing"


def test_pipeline_runs_on_routing_without_quantum():
    # allow_gate_model=False => fast (no QAOA); exercises the full agent pipeline.
    spec = make_routing_spec(allow_gate_model=False)
    ctx = solve(spec)
    assert ctx.state.classical_milp.feasible
    assert ctx.state.bundle is not None
    assert not ctx.state.errors
    assert ctx.state.audit.category == DecisionCategory.QUANTUM_NOT_FEASIBLE


@pytest.mark.slow
def test_pipeline_runs_on_routing_with_qaoa():
    spec = make_routing_spec(max_qubits=12)
    ctx = solve(spec)
    assert ctx.state.audit is not None
    qaoa = ctx.state.instance_qaoa
    inst_milp = ctx.state.instance_milp
    # QAOA can never beat the exact MILP optimum on a feasible instance solution.
    if qaoa and qaoa.feasible and qaoa.objective is not None and inst_milp.objective is not None:
        assert qaoa.objective >= inst_milp.objective - 1e-6


def test_determinism_on_routing():
    spec = make_routing_spec(allow_gate_model=False)
    a = solve(spec).state.bundle.manifest["evidence_digest"]
    b = solve(spec).state.bundle.manifest["evidence_digest"]
    assert a == b


def test_qubo_ising_energy_equivalence_routing():
    spec = make_routing_spec(max_qubits=9)
    inst = reduce_to_routing_instance(spec, 9)
    qubo = build_routing_qubo(inst)
    from qf_agentos.finance.collateral import qubo_energy, qubo_to_ising

    const, h, J = qubo_to_ising(qubo)
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.integers(0, 2, size=qubo.n)
        z = 1 - 2 * x
        ising = const + float(h @ z) + sum(c * z[i] * z[j] for (i, j), c in J.items())
        assert qubo_energy(qubo, x) == pytest.approx(ising, rel=1e-9, abs=1e-9)
