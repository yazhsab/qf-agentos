"""QF-AgentOS REST API (FastAPI).

Exposes the solve pipeline, backend capabilities, skills, and the run registry
as a service. Run with ``qf-agent serve`` or ``uvicorn qf_agentos.api:app``.
Requires the ``server`` extra.

Solving is CPU-bound; endpoints are declared ``def`` (not ``async def``) so
FastAPI runs them in a worker threadpool and the event loop stays responsive.
``POST /solve`` runs synchronously (for quick specs); ``POST /jobs`` submits the
same work to a bounded async queue and returns a job id to poll at
``GET /jobs/{id}`` — the production path for long QAOA runs.

Security: ``POST /solve``, ``POST /jobs`` and the run registry require an ``X-API-Key`` header
when ``QF_API_KEYS`` is set (otherwise the API is open, for development, and a
startup warning is logged). ``POST /solve`` is additionally rate-limited per
client. Discovery endpoints (``/healthz``, ``/backends``, ``/skills``) are open.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import yaml
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from . import __version__
from .api_jobs import JobManager, JobStatus
from .backends.registry import discover_capabilities
from .core.config import get_settings
from .core.errors import QFAgentOSError, SpecError
from .core.ir import ProblemSpec, parse_spec
from .core.observability import get_logger
from .governance.store import get_evidence_store
from .pipeline import solve as solve_spec
from .skills import load_skills
from .studio import list_example_specs, read_index_html

_logger = get_logger("api")

# Async job queue + a lock serialising evidence-store writes across worker threads.
_jobs_manager: JobManager | None = None
_jobs_lock = threading.Lock()
_persist_lock = threading.Lock()


def _jobs() -> JobManager:
    global _jobs_manager
    with _jobs_lock:
        if _jobs_manager is None:
            s = get_settings()
            _jobs_manager = JobManager(workers=s.api_job_workers, max_jobs=s.api_max_jobs)
        return _jobs_manager


def _reset_jobs() -> None:
    """Shut down and drop the job manager (used in tests)."""
    global _jobs_manager
    with _jobs_lock:
        if _jobs_manager is not None:
            _jobs_manager.shutdown()
            _jobs_manager = None


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    if not get_settings().auth_required():
        _logger.warning("API authentication is OPEN (set QF_API_KEYS to require an X-API-Key).")
    yield
    _reset_jobs()


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


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str


class StudioRunRequest(BaseModel):
    """A Studio submission: a raw YAML/JSON spec the server validates + enqueues."""

    spec_yaml: str
    approve: bool = Field(default=False)


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    problem: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: SolveResponse | None = None


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
    return [r.__dict__ for r in get_evidence_store().list_runs()]


@app.get("/runs/{run_id}")
def run_manifest(run_id: str, _identity: str = Depends(authenticate)) -> dict[str, Any]:
    manifest = get_evidence_store().load_manifest(run_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    return manifest


def _check_size(spec: ProblemSpec) -> None:
    """Bound the largest driver of solve cost over the API (413 if too big)."""
    settings = get_settings()
    n = max(len(spec.inventory), len(spec.transactions), len(spec.obligations))
    if n > settings.api_max_inventory:
        raise HTTPException(
            status_code=413,
            detail=(
                f"problem size {n} exceeds the API limit of "
                f"{settings.api_max_inventory}. Use the CLI/SDK for larger problems."
            ),
        )


def _execute_solve(request: SolveRequest) -> SolveResponse:
    """Run the pipeline and build the response (shared by /solve and /jobs)."""
    try:
        ctx = solve_spec(request.spec, human_approved=request.approve)
    except QFAgentOSError as exc:
        raise HTTPException(status_code=422, detail=exc.message) from exc

    audit = ctx.state.audit
    bundle = ctx.state.bundle
    if audit is None or bundle is None:
        raise HTTPException(status_code=500, detail="pipeline produced no decision")

    if request.persist:
        with _persist_lock:  # serialise index.jsonl appends across worker threads
            get_evidence_store(ctx.settings).save(ctx.run_id, bundle)

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


@app.post("/solve", response_model=SolveResponse)
def solve(request: SolveRequest, _identity: str = Depends(rate_limit)) -> SolveResponse:
    _check_size(request.spec)
    return _execute_solve(request)


@app.post("/jobs", response_model=JobSubmitResponse, status_code=202)
def submit_job(request: SolveRequest, _identity: str = Depends(rate_limit)) -> JobSubmitResponse:
    """Submit a solve to the async queue; poll GET /jobs/{id} for the result."""
    _check_size(request.spec)

    def _work() -> dict[str, Any]:
        try:
            return _execute_solve(request).model_dump()
        except HTTPException as exc:
            # Surface validation/pipeline errors as a failed job, not a 500.
            raise RuntimeError(str(exc.detail)) from exc

    job = _jobs().submit(_work, problem=request.spec.problem)
    return JobSubmitResponse(job_id=job.id, status=job.status.value)


@app.get("/jobs", response_model=list[JobStatusResponse])
def list_jobs(_identity: str = Depends(authenticate)) -> list[JobStatusResponse]:
    return [JobStatusResponse(**j.summary()) for j in _jobs().list()]


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str, _identity: str = Depends(authenticate)) -> JobStatusResponse:
    job = _jobs().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")
    result = (
        SolveResponse(**job.result) if job.status == JobStatus.SUCCEEDED and job.result else None
    )
    return JobStatusResponse(**job.summary(), result=result)


# ---------------------------------------------------------------------------
# QF-Studio (web UI)
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def studio_home() -> HTMLResponse:
    return HTMLResponse(read_index_html())


@app.get("/examples")
def examples() -> list[dict[str, Any]]:
    return list_example_specs()


@app.post("/studio/run", response_model=JobSubmitResponse, status_code=202)
def studio_run(req: StudioRunRequest, _identity: str = Depends(rate_limit)) -> JobSubmitResponse:
    """Validate a raw YAML/JSON spec (real validator) and enqueue an async solve."""
    try:
        data = yaml.safe_load(req.spec_yaml)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="Spec must be a mapping.")
    try:
        spec = parse_spec(data)
    except SpecError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _check_size(spec)
    request = SolveRequest(spec=spec, approve=req.approve)

    def _work() -> dict[str, Any]:
        try:
            return _execute_solve(request).model_dump()
        except HTTPException as exc:
            raise RuntimeError(str(exc.detail)) from exc

    job = _jobs().submit(_work, problem=spec.problem)
    return JobSubmitResponse(job_id=job.id, status=job.status.value)
