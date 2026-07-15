"""REST API endpoints via FastAPI's TestClient."""

from __future__ import annotations

from fastapi.testclient import TestClient

from qf_agentos.api import app
from qf_test_utils import make_spec

client = TestClient(app)


def _spec_payload(**kwargs) -> dict:
    return make_spec(**kwargs).model_dump(mode="json")


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


def test_runs_endpoint_returns_list():
    r = client.get("/runs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_run_not_found():
    r = client.get("/runs/does-not-exist")
    assert r.status_code == 404
