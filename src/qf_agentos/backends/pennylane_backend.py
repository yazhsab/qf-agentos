"""PennyLane QAOA backend — a differentiable-framework simulator alternative.

Demonstrates provider-neutrality: the same QUBO is solved by an independent
quantum-software stack. Available whenever the ``pennylane`` extra is installed
(no credentials required). Not on the default critical path.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import BackendError
from ..finance.collateral import Qubo, qubo_energy, qubo_to_ising
from .base import QuboRunConfig, QuboSolution


class PennyLaneQaoaSolver:
    name = "qaoa_pennylane"
    kind = "quantum"
    requires_credentials = False

    def is_available(self) -> tuple[bool, str]:
        try:
            import pennylane  # noqa: F401
        except Exception:
            return False, "install qf-agentos[pennylane]"
        return True, "PennyLane default.qubit QAOA"

    def solve(self, qubo: Qubo, config: QuboRunConfig) -> QuboSolution:
        try:
            import pennylane as qml
            from scipy.optimize import minimize
        except Exception as exc:
            raise BackendError(f"PennyLane unavailable: {exc}") from exc

        n = qubo.n
        _const, h, J = qubo_to_ising(qubo)

        coeffs: list[float] = []
        obs: list[object] = []
        for i in range(n):
            if abs(h[i]) > 1e-12:
                coeffs.append(float(h[i]))
                obs.append(qml.PauliZ(i))
        for (i, j), jij in J.items():
            if abs(jij) > 1e-12:
                coeffs.append(float(jij))
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
        if not obs:
            raise BackendError("Degenerate QUBO: no cost Hamiltonian terms.")

        h_cost = qml.Hamiltonian(coeffs, obs)
        h_mix = qml.Hamiltonian([1.0] * n, [qml.PauliX(i) for i in range(n)])
        reps = config.reps
        dev = qml.device("default.qubit", wires=n)

        def _ansatz(params: np.ndarray) -> None:
            for w in range(n):
                qml.Hadamard(w)
            gammas, betas = params[:reps], params[reps:]
            for layer in range(reps):
                qml.qaoa.cost_layer(gammas[layer], h_cost)
                qml.qaoa.mixer_layer(betas[layer], h_mix)

        @qml.qnode(dev)
        def expval(params: np.ndarray) -> float:
            _ansatz(params)
            return qml.expval(h_cost)

        @qml.qnode(dev)
        def probs(params: np.ndarray) -> np.ndarray:
            _ansatz(params)
            return qml.probs(wires=range(n))

        rng = np.random.default_rng(config.seed)
        best_params, best_ev = None, float("inf")
        for _ in range(3):
            x0 = rng.uniform(0.0, np.pi, size=2 * reps)
            res = minimize(
                lambda p: float(expval(p)), x0, method="COBYLA", options={"maxiter": 150}
            )
            if float(res.fun) < best_ev:
                best_ev, best_params = float(res.fun), np.asarray(res.x)

        p = np.asarray(probs(best_params), dtype=float)
        p = p / p.sum()
        draws = rng.choice(len(p), size=config.shots, p=p)

        def idx_to_bits(k: int) -> np.ndarray:
            # PennyLane basis index: wire 0 is the most-significant bit.
            return np.array([(k >> (n - 1 - w)) & 1 for w in range(n)], dtype=int)

        best_bits, best_energy = None, float("inf")
        for k in {int(d) for d in draws}:
            e = qubo_energy(qubo, idx_to_bits(k))
            if e < best_energy:
                best_energy, best_bits = e, idx_to_bits(k)

        assert best_bits is not None
        return QuboSolution(
            best_bits=[int(b) for b in best_bits],
            energy=best_energy,
            metadata={"framework": "pennylane", "reps": reps, "expectation": best_ev},
        )


__all__ = ["PennyLaneQaoaSolver"]
