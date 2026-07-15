"""Backend registry, solver protocol, and QUBO solver correctness."""

from __future__ import annotations

import pytest

from qf_agentos.backends import quantum_available
from qf_agentos.backends.base import QuboRunConfig, QuboSolver
from qf_agentos.backends.registry import (
    all_solvers,
    discover_capabilities,
    get_solver,
    solver_names,
)
from qf_agentos.core.errors import BackendUnavailableError
from qf_agentos.finance.collateral import build_qubo, reduce_to_instance
from qf_test_utils import make_spec


@pytest.fixture
def qubo():
    spec = make_spec(required=4_000_000)
    inst = reduce_to_instance(spec, 9, slack_bits=4)
    return build_qubo(inst, slack_bits=4)


def test_registry_lists_all_backends():
    names = solver_names()
    assert {
        "qubo_exact_optimum",
        "simulated_annealing",
        "qaoa_sim",
        "qaoa_ibm",
        "dwave_hybrid",
        "qaoa_pennylane",
    } <= set(names)


def test_get_unknown_solver_raises():
    with pytest.raises(BackendUnavailableError):
        get_solver("does_not_exist")


def test_all_solvers_conform_to_protocol():
    for s in all_solvers():
        assert isinstance(s, QuboSolver)
        assert s.kind in {"classical", "heuristic", "quantum"}
        available, detail = s.is_available()
        assert isinstance(available, bool) and isinstance(detail, str)


def test_capabilities_include_classical_and_are_truthful():
    caps = {c.name: c for c in discover_capabilities()}
    assert caps["classical_cpu"].available
    # Credentialed backends are unavailable in CI (no tokens).
    assert not caps["qaoa_ibm"].available
    assert not caps["dwave_hybrid"].available


def test_exact_matches_simulated_annealing(qubo):
    exact = get_solver("qubo_exact_optimum").solve(qubo, QuboRunConfig(seed=7))
    sa = get_solver("simulated_annealing").solve(qubo, QuboRunConfig(seed=7))
    assert sa.energy == pytest.approx(exact.energy, rel=1e-6)


def test_remote_backends_raise_without_credentials(qubo):
    for name in ("qaoa_ibm", "dwave_hybrid"):
        with pytest.raises(BackendUnavailableError):
            get_solver(name).solve(qubo, QuboRunConfig())


@pytest.mark.skipif(not quantum_available(), reason="qiskit not installed")
@pytest.mark.slow
def test_qaoa_sim_reaches_ground_state(qubo):
    exact = get_solver("qubo_exact_optimum").solve(qubo, QuboRunConfig(seed=7))
    qaoa = get_solver("qaoa_sim").solve(qubo, QuboRunConfig(seed=7, shots=2048, reps=2))
    # QAOA can only equal, never beat, the exact ground state.
    assert qaoa.energy >= exact.energy - 1e-6


@pytest.mark.slow
def test_pennylane_backend_is_provider_neutral(qubo):
    pytest.importorskip("pennylane")
    exact = get_solver("qubo_exact_optimum").solve(qubo, QuboRunConfig(seed=7))
    sol = get_solver("qaoa_pennylane").solve(qubo, QuboRunConfig(seed=7, shots=512, reps=1))
    assert len(sol.best_bits) == qubo.n
    assert sol.energy >= exact.energy - 1e-6
