"""Collateral formulations: MILP, QUBO/Ising, slack encoding, reductions, checks."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qf_agentos.core.ir import Security
from qf_agentos.finance.collateral import (
    ResearchInstance,
    bits_to_allocation,
    build_qubo,
    check_constraints,
    qubo_energy,
    qubo_to_ising,
    reduce_to_instance,
    solve_binary_milp,
    solve_lp_relaxation,
)
from qf_test_utils import make_spec


def test_milp_feasible_and_matches_recomputation():
    spec = make_spec(required=4_000_000, concentration={"issuer": 0.6})
    out = solve_binary_milp(spec)
    assert out.feasible and out.objective is not None
    feasible, cost, _ = check_constraints(
        spec.eligible_inventory,
        out.allocation,
        spec.constraints.required_collateral,
        spec.constraints.minimum_hqla,
        spec.constraints.concentration,
    )
    assert feasible
    assert cost == pytest.approx(out.objective, rel=1e-6)


def test_lp_is_a_lower_bound_on_milp():
    spec = make_spec(required=4_000_000)
    lp = solve_lp_relaxation(spec)
    milp = solve_binary_milp(spec)
    assert lp.feasible and milp.feasible
    assert lp.objective <= milp.objective + 1e-6


def test_milp_infeasible_when_required_exceeds_coverage():
    spec = make_spec(required=10**12)
    out = solve_binary_milp(spec)
    assert not out.feasible


def test_milp_scales_with_many_unique_issuers():
    # Each security has a distinct issuer + a concentration cap — the previous
    # dense formulation was O(n^2); the sparse (auxiliary-T) form is O(n).
    inv = [
        {
            "id": f"S{i}",
            "issuer": f"ISS{i}",
            "counterparty": "CP",
            "market_value": 1_000_000,
            "haircut": 0.0,
            "cost_bps": 1 + (i % 5),
        }
        for i in range(400)
    ]
    spec = make_spec(required=40_000_000, concentration={"issuer": 0.10}, inventory=inv)
    out = solve_binary_milp(spec)
    assert out.feasible and out.objective is not None
    feasible, _, _ = check_constraints(
        spec.eligible_inventory,
        out.allocation,
        spec.constraints.required_collateral,
        0.0,
        {"issuer": 0.10},
    )
    assert feasible


def test_reduce_empty_inventory_is_degenerate():
    spec = make_spec(inventory=[])
    inst = reduce_to_instance(spec, 8)
    assert inst.degenerate
    assert inst.n_qubits == 0


def test_build_qubo_degenerate_is_empty():
    inst = ResearchInstance(
        securities=[], required_collateral=0.0, minimum_hqla=0.0, concentration={}, degenerate=True
    )
    qubo = build_qubo(inst)
    assert qubo.n == 0 and qubo.Q == {}


def test_slack_encoding_ground_state_meets_coverage():
    """With slack bits, the exact QUBO optimum should satisfy coverage >= R'."""
    spec = make_spec(required=4_000_000)
    inst = reduce_to_instance(spec, 9, slack_bits=4)
    qubo = build_qubo(inst, slack_bits=4)
    # brute force the QUBO
    best_bits, _, _ = _brute(qubo)
    alloc = bits_to_allocation(qubo.ids, best_bits)
    total_cov = sum(s.coverage * alloc.x.get(s.id, 0.0) for s in inst.securities)
    assert total_cov >= inst.required_collateral - 1e-6


def _brute(qubo):
    from qf_agentos.backends.heuristic import brute_force_qubo

    return brute_force_qubo(qubo)


def test_bits_to_allocation_ignores_slack_bits():
    ids = ["A", "B"]
    bits = [1, 0, 1, 1]  # extra slack bits beyond the 2 securities
    alloc = bits_to_allocation(ids, bits)
    assert alloc.x == {"A": 1.0}


def test_concentration_check_detects_violation():
    secs = [
        Security(id="A", issuer="X", market_value=1_000_000, haircut=0.0, cost_bps=1),
        Security(id="B", issuer="X", market_value=1_000_000, haircut=0.0, cost_bps=1),
    ]
    from qf_agentos.core.result import Allocation

    alloc = Allocation(x={"A": 1.0, "B": 1.0})
    feasible, _, checks = check_constraints(secs, alloc, 1_000_000, 0.0, {"issuer": 0.5})
    conc = next(c for c in checks if c.name.startswith("concentration"))
    assert not conc.satisfied and not feasible


@settings(max_examples=40, deadline=None)
@given(
    covs=st.lists(st.floats(min_value=0.5, max_value=5.0), min_size=2, max_size=5),
    seed=st.integers(min_value=0, max_value=9),
)
def test_qubo_ising_energy_equivalence(covs, seed):
    """QUBO energy must equal Ising energy for every bitstring (x = (1-z)/2)."""
    secs = [
        Security(id=f"S{i}", issuer="I", market_value=v * 1_000_000, haircut=0.0, cost_bps=5 + i)
        for i, v in enumerate(covs)
    ]
    inst = ResearchInstance(
        securities=secs,
        required_collateral=sum(s.coverage for s in secs) * 0.6,
        minimum_hqla=0.0,
        concentration={},
    )
    qubo = build_qubo(inst, slack_bits=2)
    const, h, J = qubo_to_ising(qubo)
    rng = np.random.default_rng(seed)
    for _ in range(8):
        x = rng.integers(0, 2, size=qubo.n)
        z = 1 - 2 * x
        ising = const + float(h @ z) + sum(c * z[i] * z[j] for (i, j), c in J.items())
        assert qubo_energy(qubo, x) == pytest.approx(ising, rel=1e-9, abs=1e-9)
