"""Gate-model quantum backend: QAOA on a simulator (qiskit).

This is a *real* QAOA: it maps the QUBO to an Ising Hamiltonian, builds a
``QAOAAnsatz``, optimises the variational parameters against an exact statevector
expectation, then samples measurement outcomes with shot noise. No results are
faked. Requires the optional ``qiskit`` extra.

The ansatz is transpiled to rotation + CX gates *once* so the cost-layer
evolution is never re-synthesised (a slow matrix-exponential path) inside the
optimiser loop. The building blocks are exposed as functions so real-hardware
adapters can reuse the cheap simulator optimisation and only sample the final
circuit on a QPU.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from ..core.errors import BackendError
from ..finance.collateral import Qubo, qubo_energy, qubo_to_ising

_FAST_BASIS = ["rz", "ry", "rx", "h", "cx", "sx", "x"]
_METRIC_BASIS = ["rz", "sx", "x", "cx"]


def _ising_terms(qubo: Qubo) -> tuple[float, list[tuple[str, list[int], float]]]:
    """Return (ising constant, sparse Pauli term list) for the QUBO."""
    n = qubo.n
    const, h, J = qubo_to_ising(qubo)
    terms: list[tuple[str, list[int], float]] = []
    for i in range(n):
        if abs(h[i]) > 1e-12:
            terms.append(("Z", [i], float(h[i])))
    for (i, j), jij in J.items():
        if abs(jij) > 1e-12:
            terms.append(("ZZ", [i, j], float(jij)))
    return const, terms


def build_qaoa_circuit(qubo: Qubo, reps: int, seed: int) -> tuple[Any, Any, int]:
    """Build (transpiled ansatz, cost operator, n_params). Cost layer synthesised once."""
    from qiskit import transpile
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp

    _const, terms = _ising_terms(qubo)
    if not terms:
        raise BackendError("Degenerate QUBO: Ising Hamiltonian has no non-trivial terms.")
    cost_op = SparsePauliOp.from_sparse_list(terms, num_qubits=qubo.n)
    ansatz = QAOAAnsatz(cost_operator=cost_op, reps=reps)
    isa = transpile(
        ansatz, basis_gates=_FAST_BASIS, optimization_level=1, seed_transpiler=int(seed)
    )
    return isa, cost_op, int(isa.num_parameters)


def optimize_qaoa(
    qubo: Qubo, *, reps: int, seed: int, restarts: int = 4, maxiter: int = 150
) -> dict[str, Any]:
    """Optimise QAOA parameters on the statevector simulator (cheap).

    Returns best params, the transpiled circuit, the cost operator, the Ising
    constant, and optimiser diagnostics.
    """
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize

    const, _terms = _ising_terms(qubo)
    isa, cost_op, num_params = build_qaoa_circuit(qubo, reps, seed)

    def expectation(vals: np.ndarray) -> float:
        sv = Statevector(isa.assign_parameters(vals))
        return float(np.real(sv.expectation_value(cost_op)))

    rng = np.random.default_rng(seed)
    best_params, best_ev, total_evals = None, float("inf"), 0
    for _ in range(restarts):
        x0 = rng.uniform(0.0, np.pi, size=num_params)
        res = minimize(expectation, x0, method="COBYLA", options={"maxiter": maxiter})
        total_evals += int(res.get("nfev", 0))
        if float(res.fun) < best_ev:
            best_ev, best_params = float(res.fun), np.asarray(res.x)

    return {
        "best_params": best_params,
        "isa": isa,
        "cost_op": cost_op,
        "ising_const": const,
        "num_parameters": num_params,
        "best_ev": best_ev,
        "optimizer_evals": total_evals,
        "restarts": restarts,
    }


def _sample_statevector(isa: Any, params: np.ndarray, shots: int, seed: int) -> Counter[str]:
    from qiskit.quantum_info import Statevector

    sv = Statevector(isa.assign_parameters(params))
    probs = sv.probabilities_dict()
    keys = list(probs)
    p = np.array([probs[k] for k in keys], dtype=float)
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(keys), size=shots, p=p)
    return Counter(keys[i].replace(" ", "") for i in draws)


def _rank_bitstrings(
    qubo: Qubo, counts: Counter[str]
) -> tuple[np.ndarray, float, dict[str, float]]:
    n = qubo.n

    def key_to_bits(key: str) -> np.ndarray:
        return np.array([int(key[n - 1 - i]) for i in range(n)], dtype=int)

    energy_cache = {k: qubo_energy(qubo, key_to_bits(k)) for k in counts}
    best_key = min(counts, key=lambda k: energy_cache[k])
    return key_to_bits(best_key), energy_cache[best_key], energy_cache


def run_qaoa(
    qubo: Qubo,
    *,
    reps: int = 1,
    shots: int = 4096,
    seed: int = 7,
    restarts: int = 4,
    maxiter: int = 150,
) -> dict[str, Any]:
    """Run QAOA end-to-end on the statevector simulator. Returns a result dict."""
    from qiskit import transpile

    n = qubo.n
    _const, terms = _ising_terms(qubo)
    if not terms:  # degenerate Hamiltonian; nothing for QAOA to optimise
        return {
            "best_bits": np.zeros(n, dtype=int),
            "best_energy": qubo_energy(qubo, np.zeros(n)),
            "degenerate": True,
            "n_qubits": n,
            "reps": reps,
        }

    opt = optimize_qaoa(qubo, reps=reps, seed=seed, restarts=restarts, maxiter=maxiter)
    counts = _sample_statevector(opt["isa"], opt["best_params"], shots, seed)
    best_bits, best_energy, energy_cache = _rank_bitstrings(qubo, counts)

    sample_energies = np.array(
        [energy_cache[k] for k, c in counts.items() for _ in range(c)], dtype=float
    )

    # Transpilation metrics (structure only; independent of parameter values).
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp

    cost_op = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
    ansatz = QAOAAnsatz(cost_operator=cost_op, reps=reps)
    tqc = transpile(
        ansatz, basis_gates=_METRIC_BASIS, optimization_level=1, seed_transpiler=int(seed)
    )
    two_qubit_depth = tqc.depth(lambda inst: inst.operation.num_qubits == 2)

    return {
        "degenerate": False,
        "best_bits": best_bits,
        "best_energy": best_energy,
        "expectation_ising": opt["best_ev"] + opt["ising_const"],
        "n_qubits": n,
        "reps": reps,
        "num_parameters": opt["num_parameters"],
        "optimizer": "COBYLA",
        "optimizer_evals": opt["optimizer_evals"],
        "restarts": opt["restarts"],
        "shots": int(sum(counts.values())),
        "counts": dict(counts),
        "sample_mean_energy": float(sample_energies.mean()),
        "transpile": {
            "total_depth": int(tqc.depth()),
            "two_qubit_depth": int(two_qubit_depth),
            "cx_count": int(tqc.count_ops().get("cx", 0)),
            "basis_gates": _METRIC_BASIS,
        },
    }


__all__ = ["build_qaoa_circuit", "optimize_qaoa", "run_qaoa"]
