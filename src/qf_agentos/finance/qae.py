r"""Quantum Amplitude Estimation for risk / expectation estimation — honestly.

The canonical quantum-finance primitive. To estimate an expectation
:math:`\mathbb{E}[f(X)] = \sum_i p_i f_i` (an expected loss, a tail probability, an
option payoff), QAE encodes it as the amplitude of a marked state and estimates it
with RMSE :math:`\epsilon` in :math:`O(1/\epsilon)` oracle queries — a **proven
quadratic speedup** over classical Monte Carlo's :math:`O(1/\epsilon^2)` (Brassard,
Hoyer, Mosca, Tapp 2002; Montanaro 2015).

This module builds the amplitude oracle exactly, validates it on a statevector,
runs **Maximum-Likelihood Amplitude Estimation** (Suzuki et al. 2020) with a real
Grover operator, and compares it head-to-head with exact summation and classical
Monte Carlo — then reports the honest verdict and a resource analysis.

The honesty this platform is built for applies in full:

* The quadratic speedup is real and proven, but **asymptotic**. At the sizes a
  statevector can hold, exact classical summation gives the answer with zero error
  instantly; classical MC is trivial. QAE wins nothing here.
* **State preparation is the binding constraint.** Loading an arbitrary
  :math:`2^m`-outcome distribution costs :math:`O(2^m)` gates, which destroys the
  speedup unless amortised across many pricings (Stamatopoulos et al. 2020).
* Reaching a useful advantage needs **early fault tolerance** — thousands of
  logical qubits and deep coherent Grover circuits (Chakrabarti et al. 2021), a
  10+ year horizon. Nothing here runs on NISQ hardware at useful precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backends import quantum_available


@dataclass
class RiskInstance:
    """A discretised risk/expectation problem: probabilities + a payoff in [0,1]."""

    probabilities: NDArray[np.float64]  # length 2^m, sums to 1
    payoff: NDArray[np.float64]  # length 2^m, each in [0,1]
    m_qubits: int
    label: str = "expectation"

    def __post_init__(self) -> None:
        n = len(self.probabilities)
        if n != 2**self.m_qubits or len(self.payoff) != n:
            raise ValueError("probabilities/payoff length must be 2**m_qubits.")
        if abs(float(self.probabilities.sum()) - 1.0) > 1e-9:
            raise ValueError("probabilities must sum to 1.")
        if self.payoff.min() < -1e-9 or self.payoff.max() > 1 + 1e-9:
            raise ValueError("payoff must be normalised to [0, 1].")


def make_normal_loss_instance(
    m_qubits: int = 4,
    *,
    mean: float = 0.5,
    std: float = 0.2,
    tail_threshold: float | None = None,
) -> RiskInstance:
    """A discretised truncated-normal loss over ``2^m`` levels in [0,1].

    ``tail_threshold=None`` -> the payoff is the (normalised) loss itself, so the
    amplitude is the **expected loss**. Otherwise the payoff is the tail indicator
    ``1[loss > threshold]``, so the amplitude is the **exceedance probability**
    (a VaR-style quantity).
    """
    n = 2**m_qubits
    x = np.linspace(0.0, 1.0, n)
    pdf = np.exp(-0.5 * ((x - mean) / std) ** 2)
    probs = pdf / pdf.sum()
    if tail_threshold is None:
        payoff = x.copy()
        label = "expected_loss"
    else:
        payoff = (x > tail_threshold).astype(float)
        label = f"P(loss > {tail_threshold:.2f})"
    return RiskInstance(probabilities=probs, payoff=payoff, m_qubits=m_qubits, label=label)


# ---------------------------------------------------------------------------
# Classical baselines (the fair comparators)
# ---------------------------------------------------------------------------


def exact_expectation(inst: RiskInstance) -> float:
    """The ground truth: sum_i p_i f_i. Exact and instant at these sizes."""
    return float(inst.probabilities @ inst.payoff)


def classical_monte_carlo(inst: RiskInstance, n_samples: int, seed: int) -> tuple[float, float]:
    """Plain Monte Carlo estimate of E[f] and its standard error (RMSE ~ 1/sqrt(N))."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(inst.probabilities), size=n_samples, p=inst.probabilities)
    samples = inst.payoff[idx]
    est = float(samples.mean())
    stderr = float(samples.std(ddof=1) / np.sqrt(n_samples)) if n_samples > 1 else float("inf")
    return est, stderr


# ---------------------------------------------------------------------------
# Quantum amplitude oracle (A) — exact encoding
# ---------------------------------------------------------------------------


def build_amplitude_oracle(inst: RiskInstance) -> Any:
    """Build A: |0> -> sum_i sqrt(p_i)|i>(sqrt(1-f_i)|0> + sqrt(f_i)|1>).

    The good-state (objective qubit = 1) probability is exactly E[f]. State prep
    and the per-level payoff rotation are both O(2^m) — the honest binding cost.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    m = inst.m_qubits
    qc = QuantumCircuit(m + 1)  # m distribution qubits + 1 objective ancilla
    qc.append(StatePreparation(np.sqrt(inst.probabilities)), range(m))
    obj = m
    for i, f in enumerate(inst.payoff):
        angle = 2.0 * np.arcsin(float(np.sqrt(np.clip(f, 0.0, 1.0))))
        if angle == 0.0:
            continue
        bits = [(i >> b) & 1 for b in range(m)]  # little-endian control pattern
        zeros = [b for b in range(m) if bits[b] == 0]
        for b in zeros:
            qc.x(b)
        qc.mcry(angle, list(range(m)), obj)
        for b in zeros:
            qc.x(b)
    return qc


def _good_state_probability(circuit: Any, obj_qubit: int) -> float:
    from qiskit.quantum_info import Statevector

    probs = Statevector(circuit).probabilities([obj_qubit])
    return float(probs[1])


# ---------------------------------------------------------------------------
# Maximum-Likelihood Amplitude Estimation (Suzuki et al. 2020)
# ---------------------------------------------------------------------------


@dataclass
class QaeResult:
    estimate: float
    exact: float
    abs_error: float
    oracle_calls: int
    max_grover_power: int
    schedule: list[int]
    good_state_prob_check: float  # A-operator good-state prob == exact E[f]
    metadata: dict[str, Any] = field(default_factory=dict)


def mlae(
    inst: RiskInstance,
    *,
    powers: list[int] | None = None,
    shots: int = 100,
    seed: int = 7,
    grid: int = 4000,
) -> QaeResult:
    """Estimate E[f] via MLAE with a real Grover operator on the statevector.

    For each Grover power k the good-state probability is exactly
    ``sin^2((2k+1)theta)``; we read it from the statevector (what a hardware
    sampler would sample), draw ``shots`` binomial outcomes, then maximise the
    combined log-likelihood over theta on a grid. ``a = sin^2(theta)`` is the
    estimate. Oracle-call count = sum_k shots*(2k+1) — the QAE query complexity.
    """
    from qiskit.circuit.library import GroverOperator

    powers = powers or [0, 1, 2, 3, 5, 8]
    exact = exact_expectation(inst)
    a_op = build_amplitude_oracle(inst)
    obj = inst.m_qubits
    check = _good_state_probability(a_op, obj)  # must equal E[f]

    # Oracle marks the good state (objective qubit = |1>) with a phase flip.
    from qiskit import QuantumCircuit

    oracle = QuantumCircuit(inst.m_qubits + 1)
    oracle.z(obj)
    grover = GroverOperator(oracle, state_preparation=a_op)

    rng = np.random.default_rng(seed)
    hits: list[int] = []
    for k in powers:
        circ = a_op.copy()
        for _ in range(k):
            circ = circ.compose(grover)
        p_k = _good_state_probability(circ, obj)
        hits.append(int(rng.binomial(shots, min(max(p_k, 0.0), 1.0))))

    # Maximum-likelihood over theta in (0, pi/2).
    thetas = np.linspace(1e-6, np.pi / 2 - 1e-6, grid)
    loglik = np.zeros(grid)
    for k, h in zip(powers, hits, strict=True):
        angle = (2 * k + 1) * thetas
        p = np.clip(np.sin(angle) ** 2, 1e-12, 1 - 1e-12)
        loglik += h * np.log(p) + (shots - h) * np.log(1 - p)
    theta_hat = float(thetas[int(np.argmax(loglik))])
    estimate = float(np.sin(theta_hat) ** 2)

    oracle_calls = int(sum(shots * (2 * k + 1) for k in powers))
    return QaeResult(
        estimate=estimate,
        exact=exact,
        abs_error=abs(estimate - exact),
        oracle_calls=oracle_calls,
        max_grover_power=max(powers),
        schedule=list(powers),
        good_state_prob_check=check,
        metadata={"shots_per_power": shots, "label": inst.label},
    )


# ---------------------------------------------------------------------------
# Honest resource analysis
# ---------------------------------------------------------------------------


def resource_analysis(inst: RiskInstance, target_eps: float = 1e-3) -> dict[str, Any]:
    """Query/depth counts to reach RMSE target_eps, and the fault-tolerance verdict."""
    qae_queries = int(np.ceil(1.0 / target_eps))  # O(1/eps), proven
    mc_queries = int(np.ceil(1.0 / target_eps**2))  # O(1/eps^2)
    state_prep_gates = 2**inst.m_qubits  # exact loading is O(2^m) — the bottleneck
    return {
        "target_rmse": target_eps,
        "qae_oracle_queries": qae_queries,
        "classical_mc_samples": mc_queries,
        "quadratic_query_ratio": mc_queries / max(1, qae_queries),
        "state_preparation_gates": state_prep_gates,
        "max_coherent_grover_power_for_eps": qae_queries,
        "verdict": (
            "QAE's quadratic query advantage is PROVEN but asymptotic. At this scale "
            "exact classical summation is instant and error-free, so quantum wins "
            "nothing. State preparation costs O(2^m) gates (the binding constraint; "
            "the speedup survives only if amortised across many pricings), and reaching "
            "a useful advantage needs early fault tolerance — thousands of logical "
            "qubits and deep coherent Grover circuits (Chakrabarti et al. 2021), a 10+ "
            "year horizon. CLASSICAL PREFERRED."
        ),
    }


def quantum_available_for_qae() -> bool:
    """QAE needs qiskit (statevector). Reuses the backend availability check."""
    return quantum_available()


__all__ = [
    "QaeResult",
    "RiskInstance",
    "build_amplitude_oracle",
    "classical_monte_carlo",
    "exact_expectation",
    "make_normal_loss_instance",
    "mlae",
    "quantum_available_for_qae",
    "resource_analysis",
]
