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


def _build_ansatz(cost_op: Any, reps: int, n: int, warm_start: list[float] | None) -> Any:
    """QAOA ansatz, optionally warm-started (Egger et al. 2021).

    A warm start biases the initial state toward a classical relaxation solution
    ``c_i in [0,1]`` via RY(2·arcsin(√c_i)) and uses the corresponding warm-start
    mixer, so the classical solution is the mixer's ground state.
    """
    from qiskit.circuit import Parameter, QuantumCircuit
    from qiskit.circuit.library import QAOAAnsatz

    if warm_start is None or len(warm_start) != n:
        return QAOAAnsatz(cost_operator=cost_op, reps=reps)

    eps = 0.1  # regularisation: keep angles away from the poles
    c = np.clip(np.asarray(warm_start, dtype=float), eps, 1.0 - eps)
    theta = 2.0 * np.arcsin(np.sqrt(c))

    init = QuantumCircuit(n)
    for i in range(n):
        init.ry(float(theta[i]), i)

    beta = Parameter("beta_ws")
    mixer = QuantumCircuit(n)
    for i in range(n):
        mixer.ry(-float(theta[i]), i)
        mixer.rz(-2.0 * beta, i)
        mixer.ry(float(theta[i]), i)

    return QAOAAnsatz(cost_operator=cost_op, reps=reps, initial_state=init, mixer_operator=mixer)


def build_qaoa_circuit(
    qubo: Qubo, reps: int, seed: int, warm_start: list[float] | None = None
) -> tuple[Any, Any, int]:
    """Build (transpiled ansatz, cost operator, n_params). Cost layer synthesised once."""
    from qiskit import transpile
    from qiskit.quantum_info import SparsePauliOp

    _const, terms = _ising_terms(qubo)
    if not terms:
        raise BackendError("Degenerate QUBO: Ising Hamiltonian has no non-trivial terms.")
    cost_op = SparsePauliOp.from_sparse_list(terms, num_qubits=qubo.n)
    ansatz = _build_ansatz(cost_op, reps, qubo.n, warm_start)
    isa = transpile(
        ansatz, basis_gates=_FAST_BASIS, optimization_level=1, seed_transpiler=int(seed)
    )
    return isa, cost_op, int(isa.num_parameters)


def optimize_qaoa(
    qubo: Qubo,
    *,
    reps: int,
    seed: int,
    restarts: int = 4,
    maxiter: int = 150,
    warm_start: list[float] | None = None,
) -> dict[str, Any]:
    """Optimise QAOA parameters on the statevector simulator (cheap).

    Returns best params, the transpiled circuit, the cost operator, the Ising
    constant, and optimiser diagnostics.
    """
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize

    const, _terms = _ising_terms(qubo)
    isa, cost_op, num_params = build_qaoa_circuit(qubo, reps, seed, warm_start)

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


def _readout_mitigate(counts: dict[str, int], p: float, n: int, shots: int) -> Counter[str]:
    """Tensored inverse-confusion-matrix readout mitigation (symmetric error p).

    Skipped for very high readout error (p close to 0.5) where 1/(1-2p) explodes and
    the inverse is numerically meaningless; the raw noisy counts are returned instead.
    """
    if not 0 < p <= 0.4 or n > 16 or shots == 0:
        return Counter(counts)
    vec = np.zeros(2**n)
    for key, c in counts.items():
        vec[int(key.replace(" ", ""), 2)] += c
    total = vec.sum() or 1.0
    vec /= total
    a_inv = np.array([[1 - p, -p], [-p, 1 - p]]) / (1.0 - 2.0 * p)
    tensor = vec.reshape([2] * n)
    for q in range(n):
        tensor = np.moveaxis(np.tensordot(a_inv, tensor, axes=([1], [q])), 0, q)
    mvec = np.clip(tensor.reshape(2**n), 0.0, None)
    mvec = mvec / (mvec.sum() or 1.0)
    return Counter(
        {format(i, f"0{n}b"): int(round(mvec[i] * shots)) for i in range(2**n) if mvec[i] > 1e-6}
    )


def _noisy_evaluate(
    qubo: Qubo,
    isa: Any,
    params: np.ndarray,
    *,
    shots: int,
    seed: int,
    two_qubit_error: float,
    readout_error: float,
    mitigate: bool,
) -> dict[str, Any]:
    """Sample the optimised circuit under a depolarising + readout noise model,
    then optionally apply readout error mitigation. Requires qiskit-aer."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error, 2), ["cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error / 5.0, 1), ["sx", "x"])
    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - readout_error, readout_error], [readout_error, 1 - readout_error]])
    )
    sim = AerSimulator(noise_model=nm)
    circ = isa.assign_parameters(params).copy()
    circ.measure_all()
    tqc = transpile(
        circ,
        sim,
        basis_gates=["cx", "rz", "sx", "x", "id"],
        optimization_level=0,
        seed_transpiler=int(seed),
    )
    result = sim.run(tqc, shots=shots, seed_simulator=int(seed)).result()
    noisy_counts = {k.replace(" ", ""): v for k, v in result.get_counts().items()}

    # Report the DISTRIBUTION mean energy (noise-sensitive) as the headline metric.
    # best-of-shots (min over the sampled support) is nearly noise-insensitive for
    # small n with many shots — it would falsely imply noise is harmless — so it is
    # kept only as a secondary decoded solution, not the degradation signal.
    n_best, n_energy, ecache = _rank_bitstrings(qubo, Counter(noisy_counts))
    total = int(sum(noisy_counts.values())) or 1
    noisy_mean = sum(c * ecache[k] for k, c in noisy_counts.items()) / total
    out: dict[str, Any] = {
        "noisy_best_bits": n_best,
        "noisy_best_energy": n_energy,  # best-of-shots (secondary; not noise-robust)
        "noisy_mean_energy": float(noisy_mean),  # noise-sensitive headline metric
        "noisy_shots": total,
        "noise_model": {"two_qubit_depolarising": two_qubit_error, "readout": readout_error},
    }
    if mitigate:
        mit = _readout_mitigate(noisy_counts, readout_error, qubo.n, total)
        if mit:
            m_best, m_energy, mcache = _rank_bitstrings(qubo, mit)
            m_total = int(sum(mit.values())) or 1
            out["mitigated_best_bits"] = m_best
            out["mitigated_best_energy"] = m_energy
            out["mitigated_mean_energy"] = float(
                sum(c * mcache[k] for k, c in mit.items()) / m_total
            )
    return out


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
    warm_start: list[float] | None = None,
    noisy: bool = False,
    two_qubit_error: float = 0.02,
    readout_error: float = 0.03,
    mitigate: bool = True,
) -> dict[str, Any]:
    """Run QAOA end-to-end on the statevector simulator. Returns a result dict.

    When ``noisy`` is set, the optimised circuit is additionally sampled under a
    depolarising + readout noise model and (optionally) readout-error-mitigated,
    so the report can show how the result degrades on present-day hardware.
    """
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

    opt = optimize_qaoa(
        qubo, reps=reps, seed=seed, restarts=restarts, maxiter=maxiter, warm_start=warm_start
    )
    counts = _sample_statevector(opt["isa"], opt["best_params"], shots, seed)
    best_bits, best_energy, energy_cache = _rank_bitstrings(qubo, counts)

    sample_energies = np.array(
        [energy_cache[k] for k, c in counts.items() for _ in range(c)], dtype=float
    )

    # Transpilation metrics (structure only; independent of parameter values).
    from qiskit.quantum_info import SparsePauliOp

    cost_op = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
    ansatz = _build_ansatz(cost_op, reps, n, warm_start)
    tqc = transpile(
        ansatz, basis_gates=_METRIC_BASIS, optimization_level=1, seed_transpiler=int(seed)
    )
    two_qubit_depth = tqc.depth(lambda inst: inst.operation.num_qubits == 2)

    noisy_fields: dict[str, Any] = {}
    if noisy:
        noisy_fields = _noisy_evaluate(
            qubo,
            opt["isa"],
            opt["best_params"],
            shots=shots,
            seed=seed,
            two_qubit_error=two_qubit_error,
            readout_error=readout_error,
            mitigate=mitigate,
        )

    return {
        "degenerate": False,
        "best_bits": best_bits,
        "best_energy": best_energy,
        "expectation_ising": opt["best_ev"] + opt["ising_const"],
        "warm_started": warm_start is not None,
        "n_qubits": n,
        "reps": reps,
        **noisy_fields,
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
