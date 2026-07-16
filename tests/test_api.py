"""REST API endpoints via FastAPI's TestClient."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from qf_agentos.api import app
from qf_test_utils import make_spec

client = TestClient(app)


def _spec_payload(**kwargs) -> dict:
    return make_spec(**kwargs).model_dump(mode="json")


def _poll_job(job_id: str, *, timeout: float = 30.0, headers: dict | None = None) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/jobs/{job_id}", headers=headers)
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_backends_listing():
    r = client.get("/backends")
    assert r.status_code == 200
    names = {b["name"] for b in r.json()}
    assert "qaoa_sim" in names


def test_skills_listing():
    r = client.get("/skills")
    assert r.status_code == 200
    assert any(s.get("name") == "collateral-optimizer" for s in r.json())


def test_solve_endpoint():
    payload = {"spec": _spec_payload(required=4_000_000, allow_gate_model=False), "persist": False}
    r = client.post("/solve", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "decision" in body
    assert body["evidence_digest"]
    assert body["run_id"].startswith("run-")


def test_solve_invalid_spec_returns_422():
    bad = _spec_payload()
    bad["constraints"]["concentration"] = {"issuer": 2.0}  # out of range
    r = client.post("/solve", json={"spec": bad})
    assert r.status_code == 422


def test_solve_rejects_oversized_inventory(monkeypatch):
    # Defence against a single unauthenticated request driving a huge classical solve.
    from qf_agentos.core.config import reset_settings_cache

    monkeypatch.setenv("QF_API_MAX_INVENTORY", "3")
    reset_settings_cache()
    try:
        r = client.post("/solve", json={"spec": _spec_payload(allow_gate_model=False)})
        assert r.status_code == 413
    finally:
        reset_settings_cache()


def test_solve_rejects_oversized_routing(monkeypatch):
    from qf_agentos.core.config import reset_settings_cache
    from qf_test_utils import make_routing_spec

    monkeypatch.setenv("QF_API_MAX_INVENTORY", "3")
    reset_settings_cache()
    try:
        payload = {"spec": make_routing_spec(allow_gate_model=False).model_dump(mode="json")}
        assert client.post("/solve", json=payload).status_code == 413  # 6 transactions > 3
    finally:
        reset_settings_cache()


def test_async_job_lifecycle():
    payload = {"spec": _spec_payload(required=4_000_000, allow_gate_model=False), "persist": False}
    r = client.post("/jobs", json=payload)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert r.json()["status"] in ("queued", "running", "succeeded")

    body = _poll_job(job_id)
    assert body["status"] == "succeeded"
    assert body["error"] is None
    result = body["result"]
    assert result["decision"]
    assert result["evidence_digest"]
    assert result["run_id"].startswith("run-")
    assert body["finished_at"] >= body["started_at"] >= body["created_at"]


def test_job_status_not_found():
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_jobs_listing_returns_list():
    r = client.get("/jobs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_jobs_reject_oversized_spec(monkeypatch):
    from qf_agentos.core.config import reset_settings_cache

    monkeypatch.setenv("QF_API_MAX_INVENTORY", "3")
    reset_settings_cache()
    try:
        r = client.post("/jobs", json={"spec": _spec_payload(allow_gate_model=False)})
        assert r.status_code == 413
    finally:
        reset_settings_cache()


def test_jobs_require_api_key_when_configured(monkeypatch):
    from qf_agentos.api import _reset_rate_limiter
    from qf_agentos.core.config import reset_settings_cache

    monkeypatch.setenv("QF_API_KEYS", "k")
    reset_settings_cache()
    _reset_rate_limiter()
    try:
        payload = {"spec": _spec_payload(allow_gate_model=False), "persist": False}
        assert client.post("/jobs", json=payload).status_code == 401
        assert client.get("/jobs").status_code == 401
        r = client.post("/jobs", json=payload, headers={"X-API-Key": "k"})
        assert r.status_code == 202
    finally:
        reset_settings_cache()
        _reset_rate_limiter()


def test_runs_endpoint_returns_list():
    r = client.get("/runs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_run_not_found():
    r = client.get("/runs/does-not-exist")
    assert r.status_code == 404


def test_healthz_open_without_key():
    assert client.get("/healthz").status_code == 200


def test_solve_requires_api_key_when_configured(monkeypatch):
    from qf_agentos.api import _reset_rate_limiter
    from qf_agentos.core.config import reset_settings_cache

    monkeypatch.setenv("QF_API_KEYS", "secret-key-1,secret-key-2")
    reset_settings_cache()
    _reset_rate_limiter()
    try:
        payload = {
            "spec": _spec_payload(required=4_000_000, allow_gate_model=False),
            "persist": False,
        }
        assert client.post("/solve", json=payload).status_code == 401  # no key
        assert (
            client.post("/solve", json=payload, headers={"X-API-Key": "nope"}).status_code == 401
        )  # wrong key
        r = client.post("/solve", json=payload, headers={"X-API-Key": "secret-key-1"})
        assert r.status_code == 200  # valid key
    finally:
        reset_settings_cache()
        _reset_rate_limiter()


def test_runs_requires_api_key_when_configured(monkeypatch):
    from qf_agentos.core.config import reset_settings_cache

    monkeypatch.setenv("QF_API_KEYS", "k")
    reset_settings_cache()
    try:
        assert client.get("/runs").status_code == 401
        assert client.get("/runs", headers={"X-API-Key": "k"}).status_code == 200
    finally:
        reset_settings_cache()


def test_rate_limit_exceeded(monkeypatch):
    from qf_agentos.api import _reset_rate_limiter
    from qf_agentos.core.config import reset_settings_cache

    monkeypatch.setenv("QF_API_RATE_LIMIT_PER_MINUTE", "2")
    reset_settings_cache()
    _reset_rate_limiter()
    try:
        payload = {
            "spec": _spec_payload(required=4_000_000, allow_gate_model=False),
            "persist": False,
        }
        assert client.post("/solve", json=payload).status_code == 200
        assert client.post("/solve", json=payload).status_code == 200
        assert client.post("/solve", json=payload).status_code == 429  # 3rd exceeds limit=2
    finally:
        reset_settings_cache()
        _reset_rate_limiter()
