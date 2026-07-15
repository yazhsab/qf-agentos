"""Build the evidence bundle: experiment manifest, technical report, model card.

For a financial organisation this is not optional. Every run emits a
machine-readable manifest (for audit trails and reproduction) and human-readable
Markdown (for technical and executive review). A deterministic ``evidence_digest``
lets two runs be compared for reproducibility independently of wall-clock fields.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as _md
import json
import platform
from dataclasses import asdict
from typing import Any

from ..core.artifacts import EvidenceBundle
from ..core.result import SolveResult
from ..core.workflow import RunContext

_RESULT_FIELDS = [
    "classical_lp",
    "classical_milp",
    "instance_milp",
    "instance_qubo_exact",
    "instance_sa",
    "instance_qaoa",
]


def _pkg_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in ["qf-agentos", "scipy", "numpy", "pydantic", "qiskit", "qiskit-aer"]:
        try:
            out[p] = _md.version(p)
        except Exception:
            out[p] = "not installed"
    return out


def _collect_results(ctx: RunContext) -> dict[str, SolveResult]:
    results: dict[str, SolveResult] = {}
    for field in _RESULT_FIELDS:
        r = getattr(ctx.state, field)
        if r is not None:
            results[r.method] = r
    return results


def _evidence_digest(ctx: RunContext, results: dict[str, SolveResult]) -> str:
    """Deterministic hash over decision-relevant content (excludes timestamps/env)."""
    payload = {
        "spec": ctx.spec.model_dump(mode="json"),
        "seed": ctx.seed,
        "results": {
            m: {
                "feasible": r.feasible,
                "objective": round(r.objective, 6) if r.objective is not None else None,
            }
            for m, r in results.items()
        },
        "audit": ctx.state.audit.category.value if ctx.state.audit else None,
        "gap": (
            round(ctx.state.audit.objective_gap_pct, 6)
            if ctx.state.audit and ctx.state.audit.objective_gap_pct is not None
            else None
        ),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def build_bundle(ctx: RunContext) -> EvidenceBundle:
    st = ctx.state
    results = _collect_results(ctx)
    digest = _evidence_digest(ctx, results)
    if st.reproducibility is not None:
        st.reproducibility.evidence_digest = digest

    manifest: dict[str, Any] = {
        "run_id": ctx.run_id,
        "seed": ctx.seed,
        "evidence_digest": digest,
        "environment": _pkg_versions(),
        "spec": ctx.spec.model_dump(mode="json"),
        "requirements": st.requirements.model_dump(mode="json") if st.requirements else None,
        "formulations": st.formulations.model_dump(mode="json") if st.formulations else None,
        "hardware_plan": st.hardware_plan.model_dump(mode="json") if st.hardware_plan else None,
        "quantum_selection": st.quantum_selection.model_dump(mode="json")
        if st.quantum_selection
        else None,
        "results": {m: r.model_dump(mode="json") for m, r in results.items()},
        "verification": {m: v.model_dump(mode="json") for m, v in st.verification.items()},
        "reproducibility": st.reproducibility.model_dump(mode="json")
        if st.reproducibility
        else None,
        "audit": st.audit.model_dump(mode="json") if st.audit else None,
        "warnings": ctx.warnings,
        "errors": [e.model_dump(mode="json") for e in st.errors],
        "trace": [asdict(e) for e in ctx.trace],
    }

    return EvidenceBundle(
        manifest=manifest,
        report_md=_render_report(ctx, manifest),
        model_card_md=_render_model_card(ctx),
    )


def _render_report(ctx: RunContext, manifest: dict[str, Any]) -> str:
    spec = ctx.spec
    st = ctx.state
    audit = st.audit
    plan = st.hardware_plan
    req = st.requirements
    L: list[str] = []
    L.append(f"# QF-AgentOS Evidence Report — `{ctx.run_id}`\n")
    L.append(
        f"**Problem:** {spec.problem}  |  **Objective:** {spec.objective.type.value}  "
        f"|  **Autonomy:** {spec.execution_policy.autonomy_level.value}\n"
    )
    L.append(
        f"**Evidence digest:** `{manifest['evidence_digest'][:16]}…` "
        "(deterministic for a given spec + seed)\n"
    )

    L.append("## Executive summary\n")
    if audit:
        L.append(
            f"**FINAL DECISION: {audit.category.value}** — recommended method: "
            f"`{audit.recommended_method}`.\n"
        )
        L.append("> " + "\n> ".join(audit.rationale) + "\n")

    if req:
        L.append("## 1. Requirements (Agent 1)\n")
        L.append(f"- Eligible securities: **{req.n_eligible}** of {req.n_securities}")
        L.append(f"- Required collateral: **{req.required_collateral:,.0f}**")
        L.append(
            f"- Available coverage: **{req.available_coverage:,.0f}** "
            f"(headroom {req.coverage_headroom:,.0f})"
        )
        if req.discovered_gaps:
            L.append("- Discovered constraint gaps:")
            L += [f"  - {g}" for g in req.discovered_gaps]
        L.append("")

    L.append("## 2-3. Formulation & Classical baseline (Agents 2-3)\n")
    L.append("| method | scope | feasible | objective (cost) | runtime |")
    L.append("|---|---|---|---|---|")
    for field in _RESULT_FIELDS:
        r = getattr(st, field)
        if r is None:
            continue
        obj = f"{r.objective:,.2f}" if r.objective is not None else "—"
        L.append(
            f"| `{r.method}` | {r.scope} | {r.feasible} | {obj} | {r.runtime_s * 1000:.1f} ms |"
        )
    L.append("")

    if plan:
        L.append("## 4-5. Hardware planning & encoding losses (Agents 4-5)\n")
        L.append(
            f"- Research instance: **{plan.n_qubits} qubits**, QUBO density {plan.qubo_density}"
        )
        L.append(
            f"- Target: **{plan.target or 'ABSTAIN'}**; est. 2-qubit depth "
            f"{plan.estimated_two_qubit_depth}; est. cost ${plan.estimated_cost_usd:.2f}"
        )
        L.append(f"- Real QPU: {plan.real_qpu}")
        L.append("- Backends discovered:")
        L += [
            f"  - {c.name}: {'available' if c.available else 'unavailable'} — {c.detail}"
            for c in plan.capabilities
        ]
        L.append("- **Information lost in quantum encoding:**")
        L += [f"  - {loss}" for loss in plan.encoding_losses]
        L.append("")

    if st.verification:
        L.append("## 7. Verification (Agent 7)\n")
        for method, rep in st.verification.items():
            L.append(f"### `{method}` — feasible: **{rep.feasible}**")
            if rep.checks:
                L.append("| constraint | satisfied | value | limit | slack |")
                L.append("|---|---|---|---|---|")
                for c in rep.checks:
                    L.append(
                        f"| {c.name} | {c.satisfied} | {c.value:,.2f} | {c.limit:,.2f} | {c.slack:,.2f} |"
                    )
            if not rep.objective_matches_solver:
                L.append("- ⚠️ solver objective did NOT match independent recomputation")
            if rep.quantum_contribution:
                qc = rep.quantum_contribution
                L.append(f"- **Quantum-contribution accounting:** {qc['verdict']}")
                L.append(
                    f"  - QAOA mean energy {qc['qaoa_mean_energy']:.4f} vs random "
                    f"{qc['random_mean_energy']:.4f}"
                )
                L.append(
                    f"  - P(optimal) QAOA {qc['p_optimal_qaoa']:.3f} vs random {qc['p_optimal_random']:.3f}; "
                    f"reached ground state: {qc['reached_ground_state']}"
                )
            L.append("")

    L.append("## 8. Quantum-Advantage Auditor (Agent 8)\n")
    if audit:
        L.append("```\n" + audit.rendered + "\n```\n")

    L.append("## 9. Governance (Agent 9)\n")
    if st.reproducibility:
        L.append(
            f"- Reproducibility: deterministic, seed={st.reproducibility.seed}. "
            f"{st.reproducibility.note}"
        )
    if st.errors:
        L.append("- ⚠️ Non-fatal step errors (run continued):")
        L += [f"  - {e.step}: {e.error_type}: {e.message}" for e in st.errors]
    if ctx.warnings:
        L.append("- Warnings raised during the run:")
        L += [f"  - {w}" for w in ctx.warnings]
    env = manifest["environment"]
    L.append(
        f"- Environment: qf-agentos {env.get('qf-agentos')} on Python "
        f"{env['python']}, qiskit {env.get('qiskit')}."
    )
    L.append("")
    L.append(
        "_Generated by QF-AgentOS. This is an experimental research artifact, not "
        "investment advice or a production trading decision._"
    )
    return "\n".join(L)


def _render_model_card(ctx: RunContext) -> str:
    spec = ctx.spec
    audit = ctx.state.audit
    return "\n".join(
        [
            f"# Model Card — QF-AgentOS collateral-allocation run `{ctx.run_id}`\n",
            "## Intended use",
            "Decision-support for collateral optimisation, comparing classical and quantum methods. "
            "Not for autonomous execution of trades, transfers, or limit changes.\n",
            "## Method",
            "- Classical: SciPy/HiGHS LP relaxation + binary MILP (exact).",
            "- Quantum: QUBO → Ising → QAOA on a statevector simulator (reduced instance).",
            "- All quantum outputs re-verified against the full constraint set.\n",
            "## Result",
            f"- Decision category: **{audit.category.value if audit else 'n/a'}**",
            f"- Recommended method: `{audit.recommended_method if audit else 'n/a'}`\n",
            "## Limitations",
            "- Single counterparty posting model; concentration handled as a generic group cap.",
            "- QUBO relaxation drops concentration/HQLA (re-checked, not encoded).",
            "- No real-QPU results (gated behind L3 + credentials + budget).",
            f"- Autonomy level for this run: {spec.execution_policy.autonomy_level.value}.\n",
            "## Reproducibility",
            f"- Seed: {spec.execution_policy.seed}. Deterministic given identical inputs.",
        ]
    )
