"""Local QUBO solvers: exact brute force, simulated annealing, and QAOA (sim).

These are always available (QAOA requires the ``qiskit`` extra). Each wraps the
underlying numerical routine in the uniform :class:`QuboSolver` surface.
"""

from __future__ import annotations

from ..finance.collateral import Qubo
from .base import QuboRunConfig, QuboSolution
from .heuristic import brute_force_qubo, simulated_annealing_qubo


class ExactQuboSolver:
    """Exhaustive 2^n enumeration — the QUBO ground truth for small instances."""

    name = "qubo_exact_optimum"
    kind = "classical"
    requires_credentials = False

    def is_available(self) -> tuple[bool, str]:
        return True, "exact brute-force enumeration (CPU)"

    def solve(self, qubo: Qubo, config: QuboRunConfig) -> QuboSolution:
        bits, energy, evaluated = brute_force_qubo(qubo)
        return QuboSolution(
            best_bits=[int(b) for b in bits],
            energy=energy,
            metadata={"qubo_energy": energy, "states_evaluated": evaluated},
        )


class SimulatedAnnealingSolver:
    """Multi-restart simulated annealing — a strong classical heuristic."""

    name = "simulated_annealing"
    kind = "heuristic"
    requires_credentials = False

    def is_available(self) -> tuple[bool, str]:
        return True, "multi-restart simulated annealing (CPU)"

    def solve(self, qubo: Qubo, config: QuboRunConfig) -> QuboSolution:
        bits, energy, info = simulated_annealing_qubo(qubo, seed=config.seed)
        return QuboSolution(
            best_bits=[int(b) for b in bits],
            energy=energy,
            metadata={"qubo_energy": energy, **info},
        )


class QaoaSimSolver:
    """QAOA on a Qiskit statevector simulator."""

    name = "qaoa_sim"
    kind = "quantum"
    requires_credentials = False

    def is_available(self) -> tuple[bool, str]:
        try:
            import qiskit  # noqa: F401
            import qiskit_aer  # noqa: F401
        except Exception:
            return False, "gate-model simulator unavailable (install qf-agentos[qiskit])"
        return True, "qiskit statevector QAOA"

    def solve(self, qubo: Qubo, config: QuboRunConfig) -> QuboSolution:
        from .quantum import run_qaoa

        ws = list(config.warm_start) if config.warm_start is not None else None
        raw = run_qaoa(
            qubo,
            reps=config.reps,
            shots=config.shots,
            seed=config.seed,
            warm_start=ws,
            noisy=config.noisy,
            two_qubit_error=config.noise_two_qubit_error,
            readout_error=config.readout_error,
        )
        return QuboSolution(
            best_bits=[int(b) for b in raw["best_bits"]],
            energy=float(raw["best_energy"]),
            metadata=raw,
            qpu_time_s=0.0,
            cost_usd=0.0,
        )


__all__ = ["ExactQuboSolver", "QaoaSimSolver", "SimulatedAnnealingSolver"]
