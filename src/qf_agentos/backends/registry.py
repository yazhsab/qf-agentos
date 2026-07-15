"""Backend registry — the single source of truth for available QUBO solvers.

Centralises capability discovery (previously duplicated between the planner and
the backends package) and solver instantiation by name.
"""

from __future__ import annotations

from collections.abc import Callable

from ..core.artifacts import BackendCapability
from ..core.errors import BackendUnavailableError
from .base import QuboSolver
from .dwave import DwaveHybridSolver
from .ibm_runtime import IbmRuntimeQaoaSolver
from .pennylane_backend import PennyLaneQaoaSolver
from .solvers import ExactQuboSolver, QaoaSimSolver, SimulatedAnnealingSolver

# name -> zero-arg factory
_SOLVER_FACTORIES: dict[str, Callable[[], QuboSolver]] = {
    "qubo_exact_optimum": ExactQuboSolver,
    "simulated_annealing": SimulatedAnnealingSolver,
    "qaoa_sim": QaoaSimSolver,
    "qaoa_ibm": IbmRuntimeQaoaSolver,
    "dwave_hybrid": DwaveHybridSolver,
    "qaoa_pennylane": PennyLaneQaoaSolver,
}


def solver_names() -> list[str]:
    return list(_SOLVER_FACTORIES)


def get_solver(name: str) -> QuboSolver:
    """Instantiate a solver by name, or raise :class:`BackendUnavailableError`."""
    factory = _SOLVER_FACTORIES.get(name)
    if factory is None:
        raise BackendUnavailableError(
            name, f"unknown backend; known: {', '.join(_SOLVER_FACTORIES)}"
        )
    return factory()


def all_solvers() -> list[QuboSolver]:
    return [factory() for factory in _SOLVER_FACTORIES.values()]


def discover_capabilities() -> list[BackendCapability]:
    """Truthfully report which backends are usable right now (deps + credentials)."""
    caps = [BackendCapability(name="classical_cpu", available=True, detail="scipy/HiGHS MILP + LP")]
    for solver in all_solvers():
        available, detail = solver.is_available()
        suffix = " (requires credentials)" if solver.requires_credentials else ""
        caps.append(
            BackendCapability(name=solver.name, available=available, detail=detail + suffix)
        )
    return caps


__all__ = ["all_solvers", "discover_capabilities", "get_solver", "solver_names"]
