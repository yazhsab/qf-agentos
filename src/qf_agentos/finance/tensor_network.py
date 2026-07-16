r"""Tensor-network baseline: is the QAOA circuit classically simulable?

Tensor networks are both a quantum ansatz and a *strong classical competitor*. A
matrix-product state (MPS) represents an :math:`n`-qubit state with :math:`O(n\chi^2)`
parameters, where the bond dimension :math:`\chi` is set by the entanglement across
each cut. If a QAOA output state has low bond dimension, a classical MPS reproduces
it with polynomial resources — so the "quantum" circuit offers **no advantage**; a
laptop matches it.

This module measures exactly that. From a QAOA statevector it computes, at every
bipartition, the Schmidt spectrum, the bipartite entanglement entropy, and the bond
dimension needed to reach a target fidelity — then reconstructs a truncated bond-
:math:`\chi` MPS and reports its fidelity with the exact state. The honest verdict
(central to this platform): small QAOA research instances are classically simulable,
so any quantum-advantage claim from them is undercut by the tensor-network baseline
(cf. Vidal 2003 on efficient MPS simulation of slightly-entangled circuits).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def schmidt_values(psi: ComplexArray, n: int, cut: int) -> FloatArray:
    """Normalised Schmidt coefficients across a contiguous bipartition of ``cut``
    qubits vs the remaining ``n-cut`` (the entropy is symmetric, so which side is
    which does not affect any reported metric)."""
    mat = psi.reshape(2**cut, 2 ** (n - cut))
    sv = np.linalg.svd(mat, compute_uv=False)
    norm = float(np.linalg.norm(sv)) or 1.0
    return np.asarray(sv / norm, dtype=float)


def entanglement_entropy(schmidt: FloatArray) -> float:
    """Bipartite von-Neumann entropy in bits: -sum lambda^2 log2 lambda^2."""
    p = schmidt**2
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log2(p)))


def bond_dimension_for_fidelity(schmidt: FloatArray, fidelity: float = 0.99) -> int:
    """Smallest number of Schmidt values capturing >= ``fidelity`` of the weight.

    Clamped to the actual Schmidt rank: fidelity 1.0 (or float round-off above the
    total weight) must never return a bond dimension larger than the rank.
    """
    weight = np.cumsum(schmidt**2)
    return min(int(np.searchsorted(weight, fidelity) + 1), len(schmidt))


def truncated_mps_state(psi: ComplexArray, n: int, chi: int) -> ComplexArray:
    """Reconstruct the dense state of a bond-``chi`` left-canonical MPS of ``psi``."""
    residual = psi.reshape(1, 2**n).astype(complex)
    tensors: list[ComplexArray] = []
    left = 1
    for site in range(n - 1):
        mat = residual.reshape(left * 2, 2 ** (n - site - 1))
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        k = min(chi, len(s))
        tensors.append(u[:, :k].reshape(left, 2, k))
        residual = np.diag(s[:k]) @ vh[:k, :]
        left = k
    tensors.append(residual.reshape(left, 2, 1))

    res = tensors[0].reshape(2, tensors[0].shape[2])
    for t in tensors[1:]:
        res = np.tensordot(res, t, axes=([1], [0]))  # (D, 2, r)
        res = res.reshape(res.shape[0] * 2, t.shape[2])
    return np.asarray(res.reshape(-1), dtype=complex)


def truncated_mps_fidelity(psi: ComplexArray, n: int, chi: int) -> float:
    """Fidelity |<psi_MPS|psi>|^2 of the bond-``chi`` MPS approximation."""
    rec = truncated_mps_state(psi, n, chi)
    norm = float(np.linalg.norm(rec))
    if norm == 0.0:
        return 0.0
    rec = rec / norm
    return float(abs(np.vdot(rec, psi)) ** 2)


def simulability_analysis(psi: ComplexArray, n: int, *, fidelity: float = 0.99) -> dict[str, Any]:
    """Full classical-simulability report for an ``n``-qubit state."""
    psi = np.asarray(psi, dtype=complex)
    psi = psi / (float(np.linalg.norm(psi)) or 1.0)

    entropies: list[float] = []
    chis: list[int] = []
    for cut in range(1, n):
        sv = schmidt_values(psi, n, cut)
        entropies.append(entanglement_entropy(sv))
        chis.append(bond_dimension_for_fidelity(sv, fidelity))

    chi_needed = max(chis) if chis else 1
    exact_max_bond = 2 ** (n // 2)
    mps_fid = truncated_mps_fidelity(psi, n, chi_needed)
    mps_params = n * chi_needed * chi_needed * 2  # ~ O(n chi^2 d)
    statevector_params = 2**n

    # "Efficiently classically simulable BY A TENSOR NETWORK" means the MPS actually
    # compresses (a genuinely low-bond state), NOT merely that n is small. A state
    # whose bond dimension reaches the exact-rank maximum needs an MPS at least as
    # large as the statevector — no tensor-network advantage, regardless of n.
    mps_compresses = mps_params < statevector_params
    classically_simulable = bool(chi_needed < exact_max_bond and mps_compresses)

    if classically_simulable:
        note = (
            "This circuit is CLASSICALLY SIMULABLE by a low-bond tensor network — an MPS "
            "reproduces it with fewer parameters than the statevector, so it offers no quantum "
            "advantage (cf. Vidal 2003)."
        )
    else:
        note = (
            "The MPS does NOT compress this state (bond dimension near the exact-rank maximum), "
            "so a tensor network gives no advantage over exact statevector simulation — which "
            "already solves this small instance classically. Not evidence of hardware advantage."
        )
    return {
        "n_qubits": n,
        "max_entanglement_entropy_bits": max(entropies) if entropies else 0.0,
        "bond_dimension_for_fidelity": chi_needed,
        "fidelity_target": fidelity,
        "exact_max_bond_dimension": exact_max_bond,
        "truncated_mps_fidelity": mps_fid,
        "mps_parameters": mps_params,
        "statevector_parameters": statevector_params,
        "compression_ratio": statevector_params / max(1, mps_params),
        "mps_compresses": mps_compresses,
        "classically_simulable": classically_simulable,
        "verdict": (
            f"The QAOA output state needs bond dimension chi={chi_needed} for "
            f"{fidelity:.0%} fidelity (exact-rank max {exact_max_bond}); a bond-{chi_needed} "
            f"MPS reproduces it to fidelity {mps_fid:.4f} with ~{mps_params:,} parameters vs "
            f"{statevector_params:,} amplitudes. " + note
        ),
    }


def qaoa_statevector(qubo: Any, *, reps: int = 1, seed: int = 7) -> ComplexArray:
    """Optimised QAOA output statevector for a QUBO (via the shared optimiser)."""
    from qiskit.quantum_info import Statevector

    from ..backends.quantum import optimize_qaoa

    opt = optimize_qaoa(qubo, reps=reps, seed=seed)
    circuit = opt["isa"].assign_parameters(opt["best_params"])
    return np.asarray(Statevector(circuit).data, dtype=complex)


__all__ = [
    "bond_dimension_for_fidelity",
    "entanglement_entropy",
    "qaoa_statevector",
    "schmidt_values",
    "simulability_analysis",
    "truncated_mps_fidelity",
    "truncated_mps_state",
]
