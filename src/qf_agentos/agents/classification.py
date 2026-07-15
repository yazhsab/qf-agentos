"""Classification-task agents (fraud detection via quantum kernels).

A parallel middle-pipeline to the optimization agents. Reuses Requirements,
Formulation, and Governance; adds classical-baseline / quantum-planner /
execution / verification / auditor steps specialised for supervised learning.
The auditor's honesty bar is high: a quantum kernel must be *statistically
significantly* better than the classical kernel to earn more than parity.
"""

from __future__ import annotations

from typing import Any

from ..backends import quantum_available
from ..core.artifacts import ReproducibilityInfo
from ..core.domain import ClassificationDomain
from ..core.policy import Action
from ..core.result import AuditDecision, DecisionCategory, SolveResult, VerificationReport
from ..core.workflow import RunContext
from ..finance import get_domain

_TOL = 1e-9


def _domain(ctx: RunContext) -> ClassificationDomain:
    domain = get_domain(ctx.spec.problem)
    assert isinstance(domain, ClassificationDomain)
    return domain


def classification_baseline_agent(ctx: RunContext) -> str:
    domain = _domain(ctx)
    dataset = domain.load_dataset(ctx.spec)
    split = domain.split(ctx.spec, dataset)
    ctx.state.dataset = dataset
    ctx.state.split = split
    models = domain.classical_baselines(ctx.spec, dataset, split)
    ctx.state.class_models.update(models)
    best = min(models.values(), key=lambda m: m.error)
    return (
        f"Classical baselines: {', '.join(models)}; best = {best.name} "
        f"({best.metric_name}={best.metric:.3f})."
    )


def classification_planner_agent(ctx: RunContext) -> str:
    domain = _domain(ctx)
    plan = domain.plan_quantum(
        ctx.spec,
        ctx.state.dataset,
        ctx.spec.execution_policy.max_effective_qubits,
        quantum_available(),
    )
    ctx.state.feature_plan = plan
    if plan["abstain"]:
        return f"Quantum plan: ABSTAIN — {'; '.join(plan['reasons'])}. Use classical result."
    return (
        f"Quantum plan: {plan['n_qubits']}-qubit fidelity kernel over features "
        f"{plan['selected_feature_names']}; {plan['n_quantum_train']} training samples."
    )


def classification_execution_agent(ctx: RunContext) -> str:
    plan = ctx.state.feature_plan
    if plan is None or plan.get("abstain"):
        ctx.warn("Quantum kernel skipped: planner abstained.")
        return "Execution: quantum kernel skipped (abstained)."
    auth = ctx.policy.authorize(Action.RUN_SIMULATOR)
    if not auth.allowed:
        ctx.warn(f"Quantum kernel not executed: {auth.reason}")
        return f"Execution: quantum kernel not authorised ({auth.reason})."
    domain = _domain(ctx)
    model = domain.run_quantum(ctx.spec, ctx.state.dataset, ctx.state.split, plan)
    ctx.state.class_models[model.name] = model
    return f"Execution: quantum kernel trained — {model.metric_name}={model.metric:.3f}."


def classification_verification_agent(ctx: RunContext) -> str:
    domain = _domain(ctx)
    reports = domain.verify(ctx.spec, ctx.state.dataset, ctx.state.split, ctx.state.class_models)
    ctx.state.verification = reports
    ctx.state.reproducibility = ReproducibilityInfo(
        deterministic=True,
        seed=ctx.spec.execution_policy.seed,
        note="Same spec + same seed reproduce this evidence bundle exactly (timestamps aside).",
    )
    q = reports.get("quantum_kernel_ridge")
    q_txt = ""
    if q is not None and q.quantum_contribution:
        q_txt = f" Quantum significant: {q.quantum_contribution['significant']}."
    feas = {m: r.feasible for m, r in reports.items()}
    return f"Verified {len(reports)} model(s); feasibility {feas}.{q_txt}"


def _champion(ctx: RunContext) -> Any:
    reports = ctx.state.verification
    classical = [
        m
        for m in ctx.state.class_models.values()
        if m.kind == "classical" and reports.get(m.name, None) and reports[m.name].feasible
    ]
    return min(classical, key=lambda m: m.error) if classical else None


def classification_auditor_agent(ctx: RunContext) -> str:
    st = ctx.state
    plan = st.feature_plan or {}
    champion = _champion(ctx)
    candidate = st.class_models.get("quantum_kernel_ridge")
    qrep: VerificationReport | None = st.verification.get("quantum_kernel_ridge")
    qc = qrep.quantum_contribution if qrep else None

    rationale: list[str] = []
    gap_pct: float | None = None
    recommended = champion.name if champion else "none"

    if champion is None:
        category = DecisionCategory.CLASSICAL_PREFERRED
        rationale.append("No feasible classical baseline (data validation failed).")
    elif plan.get("abstain") or candidate is None:
        category = DecisionCategory.QUANTUM_NOT_FEASIBLE
        rationale.append("Quantum kernel not run:")
        rationale.extend(f"  - {r}" for r in plan.get("reasons", ["not executed"]))
        rationale.append(f"Classical champion '{champion.name}' stands.")
    elif not (qrep and qrep.feasible):
        category = DecisionCategory.CLASSICAL_PREFERRED
        rationale.append("Quantum model failed validation (leakage/temporal); classical preferred.")
    else:
        assert qc is not None
        # Gap is measured against the SAME-learner comparator the significance test
        # used (the RBF kernel), so the reported gap and the verdict agree.
        comparator = qc["compared_to"]
        gap_pct = (
            100.0
            * (qc["quantum_metric"] - qc["classical_metric"])
            / max(abs(qc["classical_metric"]), _TOL)
        )
        if qc["contributed"]:
            category = DecisionCategory.QUANTUM_IMPROVEMENT_OBSERVED
            recommended = f"{candidate.name} (pending independent reproduction)"
            rationale.append(
                f"Quantum kernel is STATISTICALLY significantly better than the "
                f"same-learner classical kernel ({comparator})."
            )
            rationale.append("Surprising — flagging for independent reproduction.")
        elif qc["significant"] and qc["mean_diff"] < 0:
            category = DecisionCategory.CLASSICAL_PREFERRED
            rationale.append(
                f"Quantum kernel is significantly WORSE than the classical kernel ({comparator})."
            )
        else:
            category = DecisionCategory.QUANTUM_PARITY
            rationale.append(f"Quantum and classical ({comparator}) kernels are within noise —")
            rationale.append(
                "no statistically significant difference (95% bootstrap CI includes 0)."
            )

    prod = ctx.policy.authorize(Action.RECOMMEND_PRODUCTION)
    rationale.append(
        "Production sign-off: authorised (L4 + human approval)."
        if prod.allowed
        else f"Production sign-off: BLOCKED — {prod.reason}"
    )

    rendered = _render(category, recommended, champion, candidate, qc, gap_pct, plan, rationale)
    classical_res = _to_result(champion, "full_dataset") if champion else None
    quantum_res = _to_result(candidate, "research_instance") if candidate else None
    st.audit = AuditDecision(
        category=category,
        recommended_method=recommended,
        rationale=rationale,
        classical=classical_res,
        quantum=quantum_res,
        objective_gap_pct=gap_pct,
        rendered=rendered,
    )
    return f"FINAL DECISION: {category.value} (recommend: {recommended})."


def _to_result(model: Any, scope: str) -> SolveResult:
    from ..finance.qml import model_to_result

    backend = "gate_model_statevector_sim" if model.kind == "quantum" else "cpu"
    return model_to_result(model, scope=scope, backend=backend)


def _render(
    category: DecisionCategory,
    recommended: str,
    champion: Any,
    candidate: Any,
    qc: dict[str, Any] | None,
    gap_pct: float | None,
    plan: dict[str, Any],
    rationale: list[str],
) -> str:
    lines = [f"FINAL DECISION: {category.value}", "", f"Recommended method : {recommended}", ""]
    if champion is not None:
        lines += [
            "Classical champion:",
            f"  - Model    : {champion.name}",
            f"  - {champion.metric_name.upper():<8} : {champion.metric:.4f}",
            "  - metrics  : " + ", ".join(f"{k}={v:.3f}" for k, v in champion.metrics.items()),
            "",
        ]
    if candidate is not None:
        lines += [
            f"Quantum kernel (fidelity, {plan.get('n_qubits')} qubits):",
            f"  - {candidate.metric_name.upper():<8} : {candidate.metric:.4f}",
            "  - metrics  : " + ", ".join(f"{k}={v:.3f}" for k, v in candidate.metrics.items()),
        ]
        if qc is not None:
            lines += [
                f"  - vs classical: mean Δ{candidate.metric_name} = {qc['mean_diff']:+.4f}, "
                f"95% CI [{qc['ci95'][0]:+.4f}, {qc['ci95'][1]:+.4f}]",
                f"  - statistically significant: {qc['significant']}",
            ]
        lines.append("")
    if gap_pct is not None:
        lines += [f"Metric gap (quantum vs classical): {gap_pct:+.2f}%", ""]
    lines += ["Reason:"] + [f"  {r}" for r in rationale]
    return "\n".join(lines)
