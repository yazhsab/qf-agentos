"""QF-AgentOS REST API (FastAPI).

Exposes the solve pipeline, backend capabilities, skills, and the run registry
as a service. Run with ``qf-agent serve`` or
``uvicorn qf_agentos.api:app``. Requires the ``server`` extra.

Solving is CPU-bound; endpoints are declared ``def`` (not ``async def``) so
FastAPI runs them in a worker threadpool and the event loop stays responsive.
For heavy production loads, put a task queue behind ``POST /solve``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .backends.registry import discover_capabilities
from .core.config import get_settings
from .core.errors import QFAgentOSError
from .core.ir import ProblemSpec
from .governance.store import EvidenceStore
from .pipeline import solve as solve_spec
from .skills import load_skills

app = FastAPI(
    title="QF-AgentOS",
    version=__version__,
    description="Agentic quantum-finance: formulate, solve, verify, and audit.",
)


class SolveRequest(BaseModel):
    spec: ProblemSpec
    approve: bool = Field(default=False, description="Approve paid/irreversible steps (L3+).")
    include_report: bool = Field(default=True, description="Include the Markdown report.")
    persist: bool = Field(default=True, description="Save the evidence bundle to the store.")


class SolveResponse(BaseModel):
    run_id: str
    decision: str
    recommended_method: str
    problem_infeasible: bool
    objective_gap_pct: float | None
    evidence_digest: str
    warnings: list[str]
    manifest: dict[str, Any]
    report_md: str | None = None


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/backends")
def backends() -> list[dict[str, Any]]:
    return [c.model_dump() for c in discover_capabilities()]


@app.get("/skills")
def skills() -> list[dict[str, Any]]:
    return load_skills()


@app.get("/runs")
def runs() -> list[dict[str, Any]]:
    store = EvidenceStore(get_settings().evidence_dir)
    return [r.__dict__ for r in store.list_runs()]


@app.get("/runs/{run_id}")
def run_manifest(run_id: str) -> dict[str, Any]:
    store = EvidenceStore(get_settings().evidence_dir)
    manifest = store.load_manifest(run_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    return manifest


@app.post("/solve", response_model=SolveResponse)
def solve(request: SolveRequest) -> SolveResponse:
    settings = get_settings()
    n = len(request.spec.inventory)
    if n > settings.api_max_inventory:
        raise HTTPException(
            status_code=413,
            detail=(
                f"inventory has {n} securities; the API limit is "
                f"{settings.api_max_inventory}. Use the CLI/SDK for larger problems."
            ),
        )
    try:
        ctx = solve_spec(request.spec, human_approved=request.approve)
    except QFAgentOSError as exc:
        raise HTTPException(status_code=422, detail=exc.message) from exc

    audit = ctx.state.audit
    bundle = ctx.state.bundle
    if audit is None or bundle is None:
        raise HTTPException(status_code=500, detail="pipeline produced no decision")

    if request.persist:
        EvidenceStore(ctx.settings.evidence_dir).save(ctx.run_id, bundle)

    return SolveResponse(
        run_id=ctx.run_id,
        decision=audit.category.value,
        recommended_method=audit.recommended_method,
        problem_infeasible=audit.problem_infeasible,
        objective_gap_pct=audit.objective_gap_pct,
        evidence_digest=bundle.manifest.get("evidence_digest", ""),
        warnings=ctx.warnings,
        manifest=bundle.manifest,
        report_md=bundle.report_md if request.include_report else None,
    )
