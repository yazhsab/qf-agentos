"""Solver backends. Provider-neutral: each exposes the uniform ``QuboSolver``
surface so the agents can select among them without knowing vendor specifics.

The gate-model, IBM, D-Wave, and PennyLane backends require optional extras and
(for hardware) credentials; they are imported lazily and report their status via
:func:`~qf_agentos.backends.registry.discover_capabilities`.
"""

from .base import QuboRunConfig, QuboSolution, QuboSolver
from .heuristic import brute_force_qubo, simulated_annealing_qubo
from .registry import all_solvers, discover_capabilities, get_solver, solver_names


def quantum_available() -> bool:
    """True if the local gate-model simulator (qiskit) can run."""
    try:
        import qiskit  # noqa: F401
        import qiskit_aer  # noqa: F401

        return True
    except Exception:
        return False


__all__ = [
    "QuboRunConfig",
    "QuboSolution",
    "QuboSolver",
    "all_solvers",
    "brute_force_qubo",
    "discover_capabilities",
    "get_solver",
    "quantum_available",
    "simulated_annealing_qubo",
    "solver_names",
]
