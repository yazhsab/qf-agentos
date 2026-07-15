"""QF-AgentOS REST API (FastAPI).

Exposes the solve pipeline, backend capabilities, skills, and the run registry
as a service. Run with ``qf-agent serve`` or ``uvicorn qf_agentos.api:app``.
Requires the ``server`` extra.

Solving is CPU-bound; endpoints are declared ``def`` (not ``async def``) so
FastAPI runs them in a worker threadpool and the event loop stays responsive.
For heavy production loads, put a task queue behind ``POST /solve``.

Security: ``POST /solve`` and the run registry require an ``X-API-Key`` header
when ``QF_API_KEYS`` is set (otherwise the API is open, for development, and a
startup warning is logged). ``POST /solve`` is additionally rate-limited per
client. Discovery endpoints (``/healthz``, ``/backends``, ``/skills``) are open.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from . import __version__
from .backends.registry import discover_capabilities
from .core.config import get_settings
from .core.errors import QFAgentOSError
from .core.ir import ProblemSpec
from .core.observability import get_logger
from .governance.store import EvidenceStore
from .pipeline import solve as solve_spec
from .skills import load_skills

_logger = get_logger("api")


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    if not get_settings().auth_required():
        _logger.warning("API authentication is OPEN (set QF_API_KEYS to require an X-API-Key).")
    yield


app = FastAPI(
    title="QF-AgentOS",
    version=__version__,
    description="Agentic quantum-finance: formulate, solve, verify, and audit.",
    lifespan=_lifespan,
)

# ---------------------------------------------------------------------------
# Auth + rate limiting
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_rate_state: dict[str, deque[float]] = defaultdict(deque)


def _reset_rate_limiter() -> None:
    """Clear the in-memory rate-limit state (used in tests)."""
    _rate_state.clear()


def authenticate(api_key: str | None = Security(_api_key_header)) -> str:
    """Return the caller identity. Open (dev) mode when no keys are configured."""
    keys = get_settings().api_key_set()
    if not keys:
        return "anonymous"
    if api_key is not None and api_key in keys:
        return api_key
    raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


def rate_limit(request: Request, identity: str = Depends(authenticate)) -> str:
    """Sliding-window per-client rate limit (in-memory; single instance)."""
    settings = get_settings()
    limit = settings.api_rate_limit_per_minute
    client = request.client.host if request.client else "unknown"
    key = identity if identity != "anonymous" else client
    now = time.monotonic()
    window = _rate_state[key]
    while window and now - window[0] > 60.0:
        window.popleft()
    if len(window) >= limit:
        raise HTTPException(
            status_code=429, detail="rate limit exceeded", headers={"Retry-After": "60"}
        )
    window.append(now)
    return identity


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


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
def runs(_identity: str = Depends(authenticate)) -> list[dict[str, Any]]:
    store = EvidenceStore(get_settings().evidence_dir)
    return [r.__dict__ for r in store.list_runs()]


@app.get("/runs/{run_id}")
def run_manifest(run_id: str, _identity: str = Depends(authenticate)) -> dict[str, Any]:
    store = EvidenceStore(get_settings().evidence_dir)
    manifest = store.load_manifest(run_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    return manifest


@app.post("/solve", response_model=SolveResponse)
def solve(request: SolveRequest, _identity: str = Depends(rate_limit)) -> SolveResponse:
    settings = get_settings()
    # Guard every problem family: bound the largest driver of solve cost
    # (securities for collateral, transactions for routing) over the sync API.
    n = max(len(request.spec.inventory), len(request.spec.transactions))
    if n > settings.api_max_inventory:
        raise HTTPException(
            status_code=413,
            detail=(
                f"problem size {n} exceeds the API limit of "
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
