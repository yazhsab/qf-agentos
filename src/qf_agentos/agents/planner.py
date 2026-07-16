"""Agents 4 & 5 — Formulation reduction + Hardware Planning (Agent-Hardware Negotiation).

Reverses the usual flow: instead of forcing the problem onto a chosen backend,
it analyses problem structure, discovers *actually available* hardware (via the
backend registry), estimates cost/depth/qubits, and decides whether quantum
execution is warranted at all. Abstention is a first-class outcome.
"""

from __future__ import annotations

from ..backends.registry import discover_capabilities
from ..core.artifacts import HardwarePlan
from ..core.domain import ProblemDomain
from ..core.workflow import RunContext
from ..finance import get_domain

_SLACK_BITS = 4


def hardware_planner_agent(ctx: RunContext) -> str:
    spec = ctx.spec
    pol = spec.execution_policy
    statevector_limit = ctx.settings.statevector_qubit_limit
    domain = get_domain(spec.problem)
    assert isinstance(domain, ProblemDomain)

    instance = domain.reduce_to_instance(spec, pol.max_effective_qubits)
    slack = max(0, min(_SLACK_BITS, pol.max_effective_qubits - instance.n_qubits))
    qubo = domain.build_qubo(instance, slack_bits=slack)
    ctx.state.instance = instance
    ctx.state.qubo = qubo

    caps = discover_capabilities()
    gate_sim_available = any(c.name == "qaoa_sim" and c.available for c in caps)

    total_qubits = qubo.n
    n_pairs = sum(1 for (i, j) in qubo.Q if i != j)
    density = (2 * n_pairs) / max(1, total_qubits * (total_qubits - 1))

    reasons: list[str] = []
    abstain = False
    target: str | None = None

    if instance.degenerate or total_qubits == 0:
        abstain = True
        reasons.append(str(instance.provenance.get("reason", "degenerate instance")))
    elif not pol.allow_gate_model and not pol.allow_quantum_annealing:
        abstain = True
        reasons.append("policy disallows all quantum backends")
    elif total_qubits > pol.max_effective_qubits:
        abstain = True
        reasons.append(
            f"{total_qubits} qubits exceeds policy max_effective_qubits={pol.max_effective_qubits}"
        )
    elif not pol.allow_gate_model:
        abstain = True
        reasons.append("policy disallows the gate model; no annealer credentials configured")
    elif not gate_sim_available:
        abstain = True
        reasons.append("gate-model simulator unavailable (install qf-agentos[qiskit])")
    elif total_qubits > statevector_limit:
        abstain = True
        reasons.append(
            f"{total_qubits} qubits beyond exact statevector budget (<= {statevector_limit})"
        )
    else:
        target = "gate_model_statevector_sim"

    # Optionally route the FINAL optimised circuit to real IBM hardware (opt-in via
    # execution_policy.qpu_backend="ibm"). Requires credentials; otherwise fall back
    # to the simulator with a recorded reason. L3 + approval + budget are enforced
    # later by the policy engine at the actual call site (RUN_PAID_QPU).
    real_qpu = "not attempted (requires credentials + L3 approval + budget)"
    estimated_cost = 0.0
    if target == "gate_model_statevector_sim" and pol.qpu_backend == "ibm":
        ibm_cap = next((c for c in caps if c.name == "qaoa_ibm"), None)
        if ibm_cap is not None and ibm_cap.available:
            target = "gate_model_ibm_runtime"
            real_qpu = (
                "IBM Quantum via Qiskit Runtime — final optimised circuit only "
                "(parameters trained on the local simulator)"
            )
        else:
            detail = ibm_cap.detail if ibm_cap is not None else "qaoa_ibm not registered"
            reasons.append(
                f"qpu_backend='ibm' requested but IBM is unavailable ({detail}); "
                "running on the local simulator instead."
            )

    est_two_qubit_depth = int(pol.qaoa_reps * n_pairs / max(1, total_qubits / 2))

    ctx.state.hardware_plan = HardwarePlan(
        n_qubits=total_qubits,
        qubo_density=round(density, 3),
        target=target,
        abstain=abstain,
        reasons=reasons,
        estimated_two_qubit_depth=est_two_qubit_depth,
        estimated_cost_usd=estimated_cost,
        real_qpu=real_qpu,
        capabilities=caps,
        encoding_losses=qubo.encoding_losses,
        instance_provenance=instance.provenance,
        instance_target=instance.target,
    )

    if abstain:
        return f"Hardware plan: ABSTAIN from quantum — {'; '.join(reasons)}. Use classical result."
    return (
        f"Hardware plan: {total_qubits}-qubit instance ({instance.n_qubits} decisions + "
        f"{qubo.slack_bits} slack) on {target}; est. 2-qubit depth {est_two_qubit_depth}, "
        f"est. cost ${0.0:.2f}. Real QPU gated (L3 + approval)."
    )
