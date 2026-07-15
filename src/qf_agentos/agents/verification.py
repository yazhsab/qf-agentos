"""Agent 7 — Verification (the most important agent).

Never trusts a solver. It independently recomputes every objective, re-checks
every candidate against the exact constraint set, and — for quantum results —
performs *quantum-contribution accounting*: did the circuit concentrate
probability mass on low-energy (good) solutions beyond what random sampling
would, and did it reach the true QUBO ground state?

The ORBIT-Q benchmark shows autonomous agents still trail human experts, so the
platform relies on this deterministic check rather than trusting an LLM's answer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.artifacts import QaoaResult, ReproducibilityInfo
from ..core.domain import ProblemDomain, ProblemInstance
from ..core.result import VerificationReport
from ..core.workflow import RunContext
from ..finance import get_domain
from ..finance.collateral import Qubo, qubo_energy


def _key_to_bits(key: str, n: int) -> NDArray[np.int_]:
    s = key.replace(" ", "")
    return np.array([int(s[n - 1 - i]) for i in range(n)], dtype=int)


def _quantum_contribution(
    qubo: Qubo, qaoa: QaoaResult, exact_energy: float | None, seed: int
) -> dict[str, object]:
    """Compare the QAOA measurement distribution to uniform random sampling."""
    n = qubo.n
    counts = qaoa.counts
    total = sum(counts.values()) or 1

    q_list: list[float] = []
    for key, c in counts.items():
        e = qubo_energy(qubo, _key_to_bits(key, n))
        q_list.extend([e] * c)
    q_energies: NDArray[np.float64] = np.array(q_list, dtype=float)

    rng = np.random.default_rng(seed + 1)
    r_energies: NDArray[np.float64] = np.array(
        [qubo_energy(qubo, rng.integers(0, 2, size=n)) for _ in range(total)], dtype=float
    )

    tol = 1e-6 * max(1.0, abs(exact_energy)) if exact_energy is not None else 1e-6
    if exact_energy is not None:
        p_opt_q = float(np.mean(q_energies <= exact_energy + tol))
        p_opt_r = float(np.mean(r_energies <= exact_energy + tol))
        reached_ground_state = bool(q_energies.min() <= exact_energy + tol)
    else:
        p_opt_q = p_opt_r = float("nan")
        reached_ground_state = False

    contributed = bool(
        q_energies.mean() < r_energies.mean() - tol and (np.isnan(p_opt_q) or p_opt_q >= p_opt_r)
    )

    return {
        "qaoa_mean_energy": float(q_energies.mean()),
        "qaoa_min_energy": float(q_energies.min()),
        "random_mean_energy": float(r_energies.mean()),
        "random_min_energy": float(r_energies.min()),
        "exact_ground_energy": exact_energy,
        "reached_ground_state": reached_ground_state,
        "p_optimal_qaoa": p_opt_q,
        "p_optimal_random": p_opt_r,
        "shots": int(total),
        "verdict": (
            "QPU contributed: measurement distribution is shifted toward low-energy "
            "solutions beyond chance."
            if contributed
            else "No measurable quantum contribution: QAOA sampling is not better than random."
        ),
        "contributed": contributed,
    }


def verification_agent(ctx: RunContext) -> str:
    spec = ctx.spec
    st = ctx.state
    domain = get_domain(spec.problem)
    assert isinstance(domain, ProblemDomain)
    instance: ProblemInstance | None = st.instance
    reports: dict[str, VerificationReport] = {}

    if st.classical_milp is not None:
        reports["classical_milp"] = domain.verify_full(spec, st.classical_milp)

    if instance is not None:
        for res in (st.instance_milp, st.instance_qubo_exact, st.instance_sa, st.instance_qaoa):
            if res is None:
                continue
            reports[res.method] = domain.verify_instance(instance, res)

    if st.qaoa_raw is not None and not st.qaoa_raw.degenerate and st.qubo is not None:
        contrib = _quantum_contribution(
            st.qubo, st.qaoa_raw, st.qubo_exact_energy, spec.execution_policy.seed
        )
        if "qaoa_sim" in reports:
            reports["qaoa_sim"].quantum_contribution = contrib
            reports["qaoa_sim"].notes.append(
                f"QAOA reached QUBO ground state: {contrib['reached_ground_state']}."
            )

    st.verification = reports
    st.reproducibility = ReproducibilityInfo(
        deterministic=True,
        seed=spec.execution_policy.seed,
        note="Same spec + same seed reproduce this evidence bundle exactly (timestamps aside).",
    )

    q_txt = ""
    if "qaoa_sim" in reports and reports["qaoa_sim"].quantum_contribution:
        contributed = reports["qaoa_sim"].quantum_contribution["contributed"]
        q_txt = f" Quantum contribution: {'yes' if contributed else 'no'}."
    feas = {m: r.feasible for m, r in reports.items()}
    return f"Verified {len(reports)} solution(s); feasibility {feas}.{q_txt}"
