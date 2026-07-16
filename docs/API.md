# REST API reference

```bash
pip install -e ".[server]"
qf-agent serve --host 0.0.0.0 --port 8000     # or: uvicorn qf_agentos.api:app
```

Interactive docs at `/docs` (OpenAPI at `/openapi.json`). The bundled Studio is at `/`.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/healthz` | open | Liveness + version |
| `GET` | `/backends` | open | Truthful backend capability discovery |
| `GET` | `/skills` | open | Installed Quantum Skills |
| `GET` | `/examples` | open | Bundled example specs (for the Studio) |
| `GET` | `/` | open | **QF-Studio** web UI |
| `POST` | `/solve` | key + rate-limit | Run the pipeline **synchronously** |
| `POST` | `/jobs` | key + rate-limit | Submit an **async** solve → `job_id` |
| `GET` | `/jobs` | key | List job records |
| `GET` | `/jobs/{job_id}` | key | Poll a job; returns the result when done |
| `POST` | `/studio/run` | key + rate-limit | Validate raw YAML + enqueue (Studio's entry point) |
| `GET` | `/runs` | key | The experiment registry |
| `GET` | `/runs/{run_id}` | key | A run's manifest |

## Authentication

Set `QF_API_KEYS` (comma-separated) to require an `X-API-Key` header on the solve
endpoints and the registries:

```bash
export QF_API_KEYS='key-one,key-two'
curl -H 'X-API-Key: key-one' http://localhost:8000/runs
```

With no keys set the API is **open** (development mode) and logs a startup warning.
Discovery endpoints and the Studio page stay open either way.

The three solve entry points are additionally **rate-limited** per client
(`QF_API_RATE_LIMIT_PER_MINUTE`, default 60 → `429`) and **size-guarded**
(`QF_API_MAX_INVENTORY`, default 2000 → `413`), across every problem family.

## Synchronous solve

```bash
curl -X POST http://localhost:8000/solve \
  -H 'Content-Type: application/json' \
  -d '{"spec": { ... }, "approve": false, "include_report": true, "persist": true}'
```

```json
{
  "run_id": "run-20260716T091948-18a20b9a",
  "decision": "CLASSICAL PREFERRED",
  "recommended_method": "classical_milp",
  "problem_infeasible": false,
  "objective_gap_pct": null,
  "evidence_digest": "646d5d51a249e6bd…",
  "warnings": ["No concentration limits given — a single issuer could dominate the pool."],
  "manifest": { "…": "…" },
  "report_md": "# QF-AgentOS Evidence Report …"
}
```

| Field | Meaning |
|---|---|
| `approve` | Approve paid/irreversible steps (**required for real QPU**) |
| `include_report` | Include the Markdown evidence report |
| `persist` | Save the evidence bundle to the configured registry |

## Asynchronous solve (the production path)

`/solve` blocks. For long QAOA/QPU runs use the job queue:

```bash
# 1. submit → 202
curl -X POST http://localhost:8000/jobs -H 'Content-Type: application/json' \
     -d '{"spec": { ... }}'
# {"job_id": "d6b60f4cb67c…", "status": "queued"}

# 2. poll
curl http://localhost:8000/jobs/d6b60f4cb67c…
```

```json
{
  "job_id": "d6b60f4cb67c…",
  "status": "succeeded",
  "problem": "collateral_allocation",
  "created_at": 1.0, "started_at": 1.1, "finished_at": 6.2,
  "error": null,
  "result": { "decision": "CLASSICAL PREFERRED", "…": "…" }
}
```

Status: `queued` → `running` → `succeeded` | `failed` (with `error`). Jobs run in a
bounded thread pool (`QF_API_JOB_WORKERS`, default 2) with an LRU-capped registry
(`QF_API_MAX_JOBS`, default 256). In-memory and single-instance by design — for
multi-replica deployments put a real broker behind the same interface.

## Studio entry point

`POST /studio/run` takes **raw YAML** (what humans write), validates it with the real
validator, and enqueues an async job:

```bash
curl -X POST http://localhost:8000/studio/run -H 'Content-Type: application/json' \
     -d '{"spec_yaml": "problem: collateral_allocation\nconstraints: {…}", "approve": false}'
# → 202 {"job_id": "…", "status": "queued"}   then poll GET /jobs/{id}
```

Invalid YAML, a non-mapping document, or a spec that fails validation returns **422**
with an actionable message. Malformed or too-deeply-nested requests return **400** —
never a 500.

## Error codes

| Code | Meaning |
|---|---|
| `400` | Malformed / too deeply nested request |
| `401` | Missing or invalid `X-API-Key` |
| `413` | Problem exceeds `QF_API_MAX_INVENTORY` — use the CLI/SDK |
| `422` | Invalid spec (validation message included) |
| `429` | Rate limit exceeded (`Retry-After: 60`) |
| `500` | Pipeline produced no decision (should not happen) |

## QF-Studio

Two frontends, same API:

- **Bundled** (`GET /`) — a self-contained vanilla-JS page, ships in the wheel, no build
  step, no external requests. Pick or paste a spec → solve → colour-coded verdict badge,
  metrics, warnings, rendered evidence report, recent-runs table.
- **React** ([`studio-react/`](../studio-react/)) — a richer Vite + TypeScript SPA kept
  as a separate deployable so the Python package stays Node-free.

```bash
cd studio-react && npm install && npm run dev    # proxies the API on :8000
```

## Registries

Runs persist through one pluggable interface (`QF_REGISTRY_BACKEND`):

| Backend | Config | Use |
|---|---|---|
| `file` (default) | `QF_EVIDENCE_DIR` | Dependency-free, single host |
| `mlflow` | `QF_MLFLOW_TRACKING_URI`, `QF_MLFLOW_EXPERIMENT` | Shared experiment tracking |
| `postgres` | `QF_POSTGRES_DSN` | Multi-instance SQL persistence |

See [CONFIGURATION.md](CONFIGURATION.md).

## Docker

```bash
docker compose up --build      # API + Studio on :8000, persists evidence/
```
