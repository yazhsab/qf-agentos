"""Quantum Amplitude Estimation: oracle correctness, MLAE accuracy, honest resources."""

from __future__ import annotations

import pytest

from qf_agentos.backends import quantum_available
from qf_agentos.finance.qae import (
    RiskInstance,
    classical_monte_carlo,
    exact_expectation,
    make_normal_loss_instance,
    mlae,
    resource_analysis,
)

pytestmark = pytest.mark.skipif(not quantum_available(), reason="qiskit not installed")


def test_instance_validation():
    import numpy as np

    with pytest.raises(ValueError):
        RiskInstance(np.array([0.5, 0.4]), np.array([0.0, 1.0]), m_qubits=1)  # probs don't sum to 1
    with pytest.raises(ValueError):
        RiskInstance(np.array([0.5, 0.5]), np.array([0.0, 2.0]), m_qubits=1)  # payoff out of [0,1]


def test_amplitude_oracle_encodes_expectation_exactly():
    # The whole method rests on this: the good-state probability == E[f].
    from qf_agentos.finance.qae import _good_state_probability, build_amplitude_oracle

    inst = make_normal_loss_instance(4, mean=0.5, std=0.2)
    gsp = _good_state_probability(build_amplitude_oracle(inst), inst.m_qubits)
    assert abs(gsp - exact_expectation(inst)) < 1e-9


@pytest.mark.slow
def test_mlae_estimates_expected_loss():
    inst = make_normal_loss_instance(4, mean=0.5, std=0.2)
    r = mlae(inst, shots=200, seed=7)
    assert abs(r.good_state_prob_check - r.exact) < 1e-9
    assert r.abs_error < 0.03  # converges near the exact expectation
    assert r.oracle_calls > 0 and r.max_grover_power == max(r.schedule)


@pytest.mark.slow
def test_mlae_estimates_tail_probability():
    inst = make_normal_loss_instance(4, mean=0.5, std=0.2, tail_threshold=0.7)
    assert "P(loss" in inst.label
    r = mlae(inst, shots=200, seed=7)
    assert abs(r.estimate - r.exact) < 0.03


def test_classical_monte_carlo_is_close():
    inst = make_normal_loss_instance(4, mean=0.5, std=0.2)
    est, se = classical_monte_carlo(inst, 5000, seed=7)
    assert abs(est - exact_expectation(inst)) < 0.05
    assert se > 0


def test_resource_analysis_shows_proven_quadratic_ratio():
    inst = make_normal_loss_instance(4)
    ra = resource_analysis(inst, target_eps=1e-3)
    # QAE O(1/eps) vs MC O(1/eps^2): the ratio is ~ 1/eps.
    assert ra["classical_mc_samples"] == ra["qae_oracle_queries"] ** 2
    assert ra["state_preparation_gates"] == 2**inst.m_qubits
    assert "CLASSICAL PREFERRED" in ra["verdict"]
    assert "fault tolerance" in ra["verdict"].lower()
