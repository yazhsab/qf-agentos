"""Agent 8 — Quantum-Advantage Auditor.

Assigns the outcome to one honest category and renders the FINAL DECISION block.
Its bias is toward classical: a quantum result must be *verified feasible* and
at least match the classical optimum before it earns anything better than
"classical preferred". This honesty is the product's credibility.
"""

from __future__ import annotations

from ..core.artifacts import HardwarePlan
from ..core.policy import Action
from ..core.result import AuditDecision, DecisionCategory, SolveResult, VerificationReport
from ..core.workflow import RunContext

_TOL = 1e-6


def _fmt(x: float | None) -> str:
    return f"{x:,.2f}" if x is not None else "n/a"


def auditor_agent(ctx: RunContext) -> str:
    st = ctx.state
    plan = st.hardware_plan
    reports: dict[str, VerificationReport] = st.verification
    milp_full = st.classical_milp
    inst_milp = st.instance_milp
    qaoa = st.instance_qaoa

    rationale: list[str] = []
    objective_gap_pct: float | None = None
    recommended = "classical_milp"

    # Full-problem feasibility governs whether there is anything to recommend.
    problem_infeasible = milp_full is not None and not milp_full.feasible

    crep = reports.get("instance_milp")
    # The QAOA sub-problem runs on the simulator or on real hardware; read whichever.
    qrep = reports.get("qaoa_sim") or reports.get("qaoa_ibm")
    c_feasible = crep.feasible if crep else (inst_milp.feasible if inst_milp else False)
    c_obj = crep.recomputed_objective if crep else (inst_milp.objective if inst_milp else None)
    q_feasible = qrep.feasible if qrep else (qaoa.feasible if qaoa else False)
    q_obj = qrep.recomputed_objective if qrep else (qaoa.objective if qaoa else None)
    qc = qrep.quantum_contribution if qrep else None
    contributed = bool(qc and qc["contributed"])
    reached_gs = bool(qc and qc.get("reached_ground_state"))

    if problem_infeasible:
        category = DecisionCategory.CLASSICAL_PREFERRED
        recommended = "none (problem is infeasible as stated)"
        rationale.append("The full problem admits NO feasible solution:")
        status = milp_full.metadata.get("status") if milp_full else None
        rationale.append(f"  classical MILP status: {status}.")
        rationale.append(
            "Relax the constraints or add eligible inventory; quantum cannot help here."
        )
    elif plan is None or plan.abstain:
        category = DecisionCategory.QUANTUM_NOT_FEASIBLE
        rationale.append("Hardware planner abstained from quantum execution:")
        rationale.extend(f"  - {r}" for r in (plan.reasons if plan else ["planner did not run"]))
        rationale.append("The classical MILP result stands as the recommendation.")
    elif qaoa is None:
        # A backend WAS selected but produced no result — do NOT call this
        # "abstention" (that would contradict the recorded hardware plan). Report
        # the true cause: a policy block or an execution failure.
        category = DecisionCategory.QUANTUM_NOT_FEASIBLE
        rationale.append(f"Quantum backend '{plan.target}' was selected but produced no result:")
        exec_err = next((e for e in ctx.state.errors if e.step == "execution"), None)
        block = next(
            (w for w in ctx.warnings if "QAOA not executed" in w or "QAOA skipped" in w), None
        )
        if exec_err is not None:
            rationale.append(f"  - execution failed: {exec_err.message}")
        elif block is not None:
            rationale.append(f"  - {block}")
        else:
            rationale.append("  - no QAOA result was recorded")
        rationale.append("The classical MILP result stands as the recommendation.")
    elif not q_feasible:
        category = DecisionCategory.CLASSICAL_PREFERRED
        rationale.append(
            "The decoded quantum solution is INFEASIBLE against the instance constraints."
        )
        if qrep:
            bad = [c.name for c in qrep.checks if not c.satisfied]
            rationale.append(f"Violated: {', '.join(bad)} — these were not encoded in the QUBO.")
        rationale.append(
            "This is exactly the information lost in quantum encoding; classical wins."
        )
    elif c_obj is None:
        category = DecisionCategory.QUANTUM_RESEARCH_CANDIDATE
        rationale.append(
            "Classical instance optimum unavailable; quantum feasible — treat as research candidate."
        )
    elif q_obj is None:
        category = DecisionCategory.QUANTUM_RESEARCH_CANDIDATE
        rationale.append("Quantum feasible but its objective could not be recomputed.")
    else:
        # Compare objectives numerically and None-safely. `c_obj` may legitimately
        # be 0.0 (e.g. zero-fee collateral) — never test it for truthiness.
        tol = _TOL * max(1.0, abs(c_obj))
        denom = abs(c_obj) if abs(c_obj) > _TOL else 1.0
        objective_gap_pct = 100.0 * (q_obj - c_obj) / denom
        slower = bool(qaoa and inst_milp and qaoa.runtime_s > inst_milp.runtime_s)
        if q_obj < c_obj - tol:
            category = DecisionCategory.QUANTUM_IMPROVEMENT_OBSERVED
            rationale.append("Quantum objective beats the classical optimum on the instance.")
            rationale.append(
                "This is surprising (MILP is exact) — flagging for independent reproduction."
            )
            recommended = "qaoa_sim (pending reproduction)"
        elif abs(q_obj - c_obj) <= tol:
            category = DecisionCategory.QUANTUM_PARITY
            rationale.append("Quantum is feasible and matches the classical objective exactly —")
            rationale.append(
                f"but it is {'slower' if slower else 'not faster'} and carries no cost or quality "
                "advantage on present hardware."
            )
            rationale.append(
                f"Quantum contribution detected: {contributed}; reached QUBO ground state: {reached_gs}."
            )
        else:
            category = DecisionCategory.CLASSICAL_PREFERRED
            rationale.append(
                f"Quantum is feasible but {objective_gap_pct:+.2f}% worse in cost than the "
                "classical optimum."
            )
            rationale.append(f"Quantum contribution detected: {contributed}.")

    # Production sign-off gate: recommending a production decision requires L4 +
    # explicit human approval. Record the gate status in the evidence.
    prod_auth = ctx.policy.authorize(Action.RECOMMEND_PRODUCTION)
    if not prod_auth.allowed:
        rationale.append(f"Production sign-off: BLOCKED — {prod_auth.reason}")
    else:
        rationale.append("Production sign-off: authorised (L4 + human approval).")

    rendered = _render(
        category,
        recommended,
        milp_full,
        inst_milp,
        qaoa,
        plan,
        objective_gap_pct,
        c_feasible,
        q_feasible,
        reached_gs,
        contributed,
        problem_infeasible,
        rationale,
    )

    st.audit = AuditDecision(
        category=category,
        recommended_method=recommended,
        rationale=rationale,
        classical=inst_milp,
        quantum=qaoa,
        objective_gap_pct=objective_gap_pct,
        problem_infeasible=problem_infeasible,
        rendered=rendered,
    )
    return f"FINAL DECISION: {category.value} (recommend: {recommended})."


def _render(
    category: DecisionCategory,
    recommended: str,
    milp_full: SolveResult | None,
    inst_milp: SolveResult | None,
    qaoa: SolveResult | None,
    plan: HardwarePlan | None,
    gap: float | None,
    c_feasible: bool,
    q_feasible: bool,
    reached_gs: bool,
    contributed: bool,
    problem_infeasible: bool,
    rationale: list[str],
) -> str:
    lines = [f"FINAL DECISION: {category.value}", ""]
    lines.append(f"Recommended method : {recommended}")
    lines.append("")
    if milp_full is not None:
        v = 0 if milp_full.feasible else "n/a"
        lines += [
            "Full-problem classical optimum (scipy/HiGHS MILP):",
            f"  - Objective (posting cost) : {_fmt(milp_full.objective)}",
            f"  - Feasible                 : {milp_full.feasible}",
            f"  - Constraint violations    : {v}",
            f"  - Runtime                  : {milp_full.runtime_s * 1000:.1f} ms",
            "",
        ]
    if not problem_infeasible:
        n = plan.n_qubits if plan else 0
        lines.append(f"Research-instance comparison ({n} qubits):")
        if inst_milp is not None:
            lines += [
                "  Classical instance optimum (MILP):",
                f"    - Objective : {_fmt(inst_milp.objective)}",
                f"    - Feasible  : {c_feasible}",
                f"    - Runtime   : {inst_milp.runtime_s * 1000:.1f} ms",
            ]
        if qaoa is not None:
            is_sim = (qaoa.backend or "").endswith("statevector_sim")
            where = "statevector simulation" if is_sim else f"real hardware · {qaoa.backend}"
            qpu_note = "(simulated)" if is_sim else "(real device)"
            lines += [
                f"  Quantum (QAOA, {where}):",
                f"    - Objective              : {_fmt(qaoa.objective)}",
                f"    - Feasible               : {q_feasible}",
                f"    - Runtime                : {qaoa.runtime_s * 1000:.1f} ms",
                f"    - QPU access time        : {qaoa.qpu_time_s:.3f} s {qpu_note}",
                f"    - Estimated cost         : ${qaoa.cost_usd:.2f}",
                f"    - Reached QUBO optimum   : {reached_gs}",
                f"    - Quantum contribution   : {contributed}",
            ]
        else:
            lines.append("  Quantum: not executed (see reasons below).")
        if gap is not None:
            lines += ["", f"Objective gap (quantum vs classical instance optimum): {gap:+.2f}%"]
    lines += ["", "Reason:"]
    lines += [f"  {r}" for r in rationale]
    return "\n".join(lines)
