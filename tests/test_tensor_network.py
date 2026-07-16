"""Tensor-network baseline: entanglement entropy, bond dimension, MPS simulability."""

from __future__ import annotations

import numpy as np
import pytest

from qf_agentos.backends import quantum_available
from qf_agentos.finance.tensor_network import (
    bond_dimension_for_fidelity,
    entanglement_entropy,
    schmidt_values,
    simulability_analysis,
    truncated_mps_fidelity,
)


def test_product_state_has_zero_entanglement():
    psi = np.zeros(16, dtype=complex)
    psi[0] = 1.0
    a = simulability_analysis(psi, 4)
    assert a["max_entanglement_entropy_bits"] == pytest.approx(0.0, abs=1e-9)
    assert a["bond_dimension_for_fidelity"] == 1
    assert a["truncated_mps_fidelity"] == pytest.approx(1.0)
    assert a["classically_simulable"] is True


def test_bell_state_has_one_bit_of_entanglement():
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert entanglement_entropy(schmidt_values(bell, 2, 1)) == pytest.approx(1.0)


def test_ghz_is_low_bond_dimension():
    ghz = np.zeros(16, dtype=complex)
    ghz[0] = ghz[15] = 1 / np.sqrt(2)
    a = simulability_analysis(ghz, 4)
    assert a["max_entanglement_entropy_bits"] == pytest.approx(1.0)
    assert a["bond_dimension_for_fidelity"] == 2
    assert a["truncated_mps_fidelity"] == pytest.approx(1.0)


def test_truncated_mps_recovers_full_state_at_max_bond():
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    psi = psi / np.linalg.norm(psi)
    # Exact-rank bond dimension reproduces the state exactly.
    assert truncated_mps_fidelity(psi, 4, 4) == pytest.approx(1.0, abs=1e-9)
    # A too-small bond dimension loses fidelity on a generic (highly entangled) state.
    assert truncated_mps_fidelity(psi, 4, 1) < 0.99


def test_bond_dimension_monotonic_in_fidelity():
    ghz = np.zeros(16, dtype=complex)
    ghz[0] = ghz[15] = 1 / np.sqrt(2)
    sv = schmidt_values(ghz, 4, 2)
    assert bond_dimension_for_fidelity(sv, 0.4) <= bond_dimension_for_fidelity(sv, 0.99)


def test_bond_dimension_never_exceeds_rank_at_fidelity_one():
    # Regression: fidelity=1.0 must not return more than the Schmidt rank.
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert bond_dimension_for_fidelity(schmidt_values(bell, 2, 1), 1.0) <= 2
    maxent = np.ones(16, dtype=complex) / 4.0  # near-maximal 4-qubit entanglement
    sv = schmidt_values(maxent, 4, 2)
    assert bond_dimension_for_fidelity(sv, 1.0) <= len(sv)


def test_max_entangled_state_is_not_declared_mps_simulable():
    # Regression / honesty: a maximally-entangled state must NOT be called
    # "classically simulable by a tensor network" (the MPS is bigger than the
    # statevector) and must not misattribute Vidal 2003.
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    a = simulability_analysis(psi, 4)
    # High-entanglement state: the MPS is no smaller than the statevector, so no
    # tensor-network advantage and no Vidal misattribution.
    assert a["mps_compresses"] is False
    assert a["classically_simulable"] is False
    assert "Vidal" not in a["verdict"]
    assert "no advantage over exact statevector" in a["verdict"]


@pytest.mark.skipif(not quantum_available(), reason="qiskit not installed")
@pytest.mark.slow
def test_qaoa_statevector_simulability_runs():
    from qf_agentos.core.ir import ProblemSpec
    from qf_agentos.finance.collateral import build_qubo, reduce_to_instance
    from qf_agentos.finance.tensor_network import qaoa_statevector
    from qf_test_utils import default_inventory

    spec = ProblemSpec.model_validate(
        {
            "problem": "collateral_allocation",
            "constraints": {"required_collateral": 4_000_000},
            "inventory": default_inventory(),
        }
    )
    qubo = build_qubo(reduce_to_instance(spec, 9, slack_bits=4), slack_bits=4)
    a = simulability_analysis(qaoa_statevector(qubo, reps=1, seed=7), qubo.n)
    assert a["n_qubits"] == qubo.n
    assert 1 <= a["bond_dimension_for_fidelity"] <= a["exact_max_bond_dimension"]
    assert 0.0 <= a["truncated_mps_fidelity"] <= 1.0 + 1e-9
    assert isinstance(a["classically_simulable"], bool)
    assert "verdict" in a
