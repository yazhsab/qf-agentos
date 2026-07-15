"""Classical QUBO solvers: exact brute force and simulated annealing.

These are deliberately strong. The project's credibility depends on never
comparing a quantum method against a weak baseline, so the classical comparator
for the QUBO is either its *exact* optimum (small n) or a well-tuned annealer.
"""

from __future__ import annotations

import numpy as np

from ..finance.collateral import Qubo, qubo_energy


def brute_force_qubo(qubo: Qubo) -> tuple[np.ndarray, float, int]:
    """Exhaustively minimise the QUBO. Returns (best_bits, best_energy, evaluated)."""
    n = qubo.n
    if n == 0:
        return np.zeros(0, dtype=int), qubo.offset, 0
    best_bits = np.zeros(n, dtype=int)
    best_e = float("inf")
    for mask in range(1 << n):
        bits = np.array([(mask >> k) & 1 for k in range(n)], dtype=int)
        e = qubo_energy(qubo, bits)
        if e < best_e:
            best_e, best_bits = e, bits
    return best_bits, best_e, (1 << n)


def simulated_annealing_qubo(
    qubo: Qubo,
    *,
    seed: int = 7,
    sweeps: int = 2000,
    restarts: int = 20,
    t_hi: float = 2.0,
    t_lo: float = 0.02,
) -> tuple[np.ndarray, float, dict[str, int]]:
    """Multi-restart simulated annealing over the QUBO. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    n = qubo.n
    if n == 0:
        return np.zeros(0, dtype=int), qubo.offset, {"sweeps": 0, "restarts": 0, "seed": seed}

    # Precompute a dense symmetric matrix for O(1) delta evaluation.
    W = np.zeros((n, n))
    for (i, j), c in qubo.Q.items():
        if i == j:
            W[i, i] = c
        else:
            W[i, j] += c
            W[j, i] += c  # symmetric off-diagonal halves; delta uses full row

    best_bits = np.zeros(n, dtype=int)
    best_e = float("inf")
    temps = np.geomspace(t_hi, t_lo, sweeps)

    for _ in range(restarts):
        b = rng.integers(0, 2, size=n)
        e = qubo_energy(qubo, b)
        for T in temps:
            i = int(rng.integers(0, n))
            # delta for flipping bit i: diagonal + interactions with current bits
            si = 1 - 2 * b[i]  # +1 if 0->1, -1 if 1->0
            delta = si * W[i, i]
            # off-diagonal contribution (row i excluding diagonal)
            inter = 0.0
            row = W[i]
            for j in range(n):
                if j != i and b[j]:
                    inter += row[j]
            delta += si * inter
            if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
                b[i] = 1 - b[i]
                e += delta
            if e < best_e:
                best_e, best_bits = e, b.copy()

    # Recompute exactly to shed any incremental float drift.
    best_e = qubo_energy(qubo, best_bits)
    return best_bits, best_e, {"sweeps": sweeps, "restarts": restarts, "seed": seed}
