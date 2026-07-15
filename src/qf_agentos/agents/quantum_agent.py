"""Agents 4 & 6 — Quantum Algorithm selection + Execution.

The Quantum Algorithm agent picks a technique from problem structure rather than
reflexively reaching for QAOA. The Execution agent runs the instance-level ladder
that makes the eventual comparison fair, selecting QUBO solvers from the backend
registry, delegating decode+check to the problem domain, and enforcing the policy
engine before any (simulated or paid) run:

    instance classical optimum (all constraints)
        -> QUBO exact ground state (reference for quantum contribution)
        -> simulated annealing (strong classical heuristic)
        -> QAOA on the statevector simulator (authorised at L2)

Real-QPU execution reuses the same registry path, gated behind L3 + approval.
"""

from __future__ import annotations

import time

from ..backends.base import QuboRunConfig
from ..backends.registry import get_solver
from ..core.artifacts import QaoaResult, QuantumSelection, TranspileMetrics
from ..core.domain import ProblemDomain
from ..core.policy import Action
from ..core.workflow import RunContext
from ..finance import get_domain

# Maps the planner's abstract target to a concrete registry backend name.
_TARGET_TO_BACKEND = {"gate_model_statevector_sim": "qaoa_sim"}


def quantum_algorithm_agent(ctx: RunContext) -> str:
    plan = ctx.state.hardware_plan
    pol = ctx.spec.execution_policy

    if plan is None or plan.abstain:
        reasons = plan.reasons if plan else ["planner did not run"]
        ctx.state.quantum_selection = QuantumSelection(
            algorithm=None, reason="planner abstained", rationale=reasons
        )
        return "Quantum algorithm: none selected (planner abstained from quantum)."

    ctx.state.quantum_selection = QuantumSelection(
        algorithm="QAOA",
        reps=pol.qaoa_reps,
        mixer="transverse-field X",
        optimizer="COBYLA (multi-restart)",
        warm_start="Egger warm-start from the classical relaxation (active)",
        alternatives_considered=[
            "D-Wave hybrid (needs credentials)",
            "IBM QPU (needs credentials)",
        ],
        rationale=[
            "Small, dense QUBO — a natural gate-model fit for QAOA.",
            "Annealing is preferred if a D-Wave backend and budget are available.",
        ],
    )
    return f"Quantum algorithm: QAOA(reps={pol.qaoa_reps}) for the {plan.n_qubits}-qubit instance."


def execution_agent(ctx: RunContext) -> str:
    spec = ctx.spec
    pol = spec.execution_policy
    instance = ctx.state.instance
    qubo = ctx.state.qubo
    plan = ctx.state.hardware_plan
    if instance is None or qubo is None or plan is None:
        return "Execution skipped: no instance/QUBO/plan available."

    domain = get_domain(spec.problem)
    assert isinstance(domain, ProblemDomain)
    warm_start = domain.instance_warm_start(instance, qubo)
    config = QuboRunConfig(
        seed=pol.seed,
        shots=pol.shots,
        reps=pol.qaoa_reps,
        warm_start=tuple(warm_start) if warm_start is not None else None,
        noisy=pol.noisy_simulation,
        noise_two_qubit_error=pol.noise_two_qubit_error,
        readout_error=pol.readout_error,
    )
    ran: list[str] = []

    # 1. Classical optimum on the SAME instance (all constraints) — fair comparator.
    ctx.state.instance_milp = domain.solve_instance_classical(instance)
    ran.append("instance_milp")

    # 2. QUBO exact ground state — reference for the quantum-contribution audit.
    if 0 < qubo.n <= ctx.settings.bruteforce_qubit_limit:
        solver = get_solver("qubo_exact_optimum")
        t0 = time.perf_counter()
        sol = solver.solve(qubo, config)
        dt = time.perf_counter() - t0
        ctx.state.instance_qubo_exact = domain.evaluate_bits(
            instance,
            sol.best_bits,
            method=solver.name,
            kind=solver.kind,
            backend="bruteforce",
            runtime_s=dt,
            metadata=sol.metadata,
        )
        ctx.state.qubo_exact_energy = sol.energy
        ran.append("qubo_exact_optimum")

    # 3. Simulated annealing on the QUBO — strong classical heuristic.
    if qubo.n > 0:
        solver = get_solver("simulated_annealing")
        t0 = time.perf_counter()
        sol = solver.solve(qubo, config)
        dt = time.perf_counter() - t0
        ctx.state.instance_sa = domain.evaluate_bits(
            instance,
            sol.best_bits,
            method=solver.name,
            kind=solver.kind,
            backend="cpu",
            runtime_s=dt,
            metadata=sol.metadata,
        )
        ran.append("simulated_annealing")

    # 4. QAOA — only if the planner chose a backend and policy authorises. A
    #    credentialed (paid) backend requires RUN_PAID_QPU (L3 + approval + budget);
    #    a simulator requires only RUN_SIMULATOR (L2).
    backend_name = _TARGET_TO_BACKEND.get(plan.target or "")
    if backend_name is not None:
        solver = get_solver(backend_name)
        action = Action.RUN_PAID_QPU if solver.requires_credentials else Action.RUN_SIMULATOR
        auth = ctx.policy.authorize(action, cost_usd=plan.estimated_cost_usd)
        if auth.allowed:
            t0 = time.perf_counter()
            sol = solver.solve(qubo, config)
            dt = time.perf_counter() - t0
            raw = sol.metadata
            tp = raw.get("transpile") if isinstance(raw, dict) else None
            ctx.state.qaoa_raw = QaoaResult(
                degenerate=bool(raw.get("degenerate", False)),
                best_bits=sol.best_bits,
                best_energy=sol.energy,
                n_qubits=int(raw.get("n_qubits", qubo.n)),
                reps=int(raw.get("reps", pol.qaoa_reps)),
                expectation_ising=raw.get("expectation_ising"),
                num_parameters=raw.get("num_parameters"),
                optimizer=raw.get("optimizer"),
                optimizer_evals=raw.get("optimizer_evals"),
                restarts=raw.get("restarts"),
                shots=raw.get("shots"),
                counts=raw.get("counts", {}),
                sample_mean_energy=raw.get("sample_mean_energy"),
                transpile=TranspileMetrics(**tp) if isinstance(tp, dict) else None,
            )
            slim = {
                k: raw.get(k)
                for k in (
                    "expectation_ising",
                    "num_parameters",
                    "optimizer",
                    "optimizer_evals",
                    "restarts",
                    "shots",
                    "sample_mean_energy",
                    "transpile",
                    "reps",
                )
            }
            slim["qubo_energy"] = sol.energy
            ctx.state.instance_qaoa = domain.evaluate_bits(
                instance,
                sol.best_bits,
                method=solver.name,
                kind=solver.kind,
                backend="gate_model_statevector_sim",
                runtime_s=dt,
                qpu_time_s=sol.qpu_time_s,
                cost_usd=sol.cost_usd,
                metadata=slim,
            )
            ran.append("qaoa_sim")

            # Optional noisy-simulation pass (present-hardware feasibility). The
            # HONEST degradation signal is the distribution MEAN energy (noise-
            # sensitive), not the best-of-shots decoded solution.
            if isinstance(raw, dict) and "noisy_best_bits" in raw:
                ideal_mean = raw.get("sample_mean_energy")
                noisy_mean = raw.get("noisy_mean_energy")
                degradation = (
                    noisy_mean - ideal_mean
                    if (ideal_mean is not None and noisy_mean is not None)
                    else None
                )
                noisy_meta = {
                    "noise_model": raw.get("noise_model"),
                    "mean_energy_ideal": ideal_mean,
                    "mean_energy_noisy": noisy_mean,
                    "mean_energy_mitigated": raw.get("mitigated_mean_energy"),
                    "mean_energy_degradation": degradation,
                    "decoded_best_energy_ideal": sol.energy,
                    "decoded_best_energy_noisy": raw.get("noisy_best_energy"),
                    "note": "Degradation is measured by the distribution MEAN energy; "
                    "best-of-shots decoding stays feasible but is not a noise-robust metric.",
                }
                ctx.state.instance_qaoa_noisy = domain.evaluate_bits(
                    instance,
                    raw["noisy_best_bits"],
                    method="qaoa_noisy_sim",
                    kind="quantum",
                    backend="noisy_statevector_sim",
                    metadata=noisy_meta,
                )
                ran.append("qaoa_noisy_sim")
        else:
            ctx.warn(f"QAOA not executed: {auth.reason}")
    else:
        ctx.warn("QAOA skipped: planner did not select a gate-model target.")

    milp_res = ctx.state.instance_milp
    milp_txt = (
        f"{milp_res.objective:,.2f}"
        if milp_res and milp_res.objective is not None
        else "infeasible"
    )
    return f"Executed {len(ran)} method(s) on the instance: {', '.join(ran)}. Instance optimum: {milp_txt}."
