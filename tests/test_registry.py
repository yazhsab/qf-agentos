"""Pluggable evidence registry: the file store and the optional MLflow backend."""

from __future__ import annotations

import pytest

from qf_agentos.core.artifacts import EvidenceBundle
from qf_agentos.core.config import Settings, reset_settings_cache
from qf_agentos.governance.store import (
    EvidenceStore,
    EvidenceStoreProtocol,
    get_evidence_store,
)


def _bundle(run_problem: str = "collateral_allocation") -> EvidenceBundle:
    return EvidenceBundle(
        manifest={
            "problem": run_problem,
            "evidence_digest": "abc123",
            "audit": {
                "category": "CLASSICAL PREFERRED",
                "recommended_method": "classical_milp",
                "problem_infeasible": False,
                "objective_gap_pct": 1.5,
            },
        },
        report_md="# Report\nhello",
        model_card_md="# Card",
    )


def test_file_store_satisfies_protocol():
    assert isinstance(EvidenceStore("evidence"), EvidenceStoreProtocol)


def test_factory_returns_file_store_by_default(tmp_path):
    store = get_evidence_store(Settings(evidence_dir=tmp_path))
    assert isinstance(store, EvidenceStore)


def test_factory_returns_mlflow_store_when_configured(tmp_path):
    pytest.importorskip("mlflow")
    from qf_agentos.governance.mlflow_store import MLflowEvidenceStore

    store = get_evidence_store(
        Settings(
            registry_backend="mlflow",
            mlflow_tracking_uri=f"file://{tmp_path}/mlruns",
            mlflow_experiment="qf-test-factory",
        )
    )
    assert isinstance(store, MLflowEvidenceStore)
    assert isinstance(store, EvidenceStoreProtocol)


def test_mlflow_round_trip(tmp_path):
    pytest.importorskip("mlflow")
    from qf_agentos.governance.mlflow_store import MLflowEvidenceStore

    store = MLflowEvidenceStore(
        tracking_uri=f"file://{tmp_path}/mlruns", experiment="qf-test-roundtrip"
    )
    run_id = "run-20260716-deadbeef"
    store.save(run_id, _bundle())

    records = store.list_runs()
    assert any(r.run_id == run_id for r in records)
    rec = next(r for r in records if r.run_id == run_id)
    assert rec.decision == "CLASSICAL PREFERRED"
    assert rec.recommended_method == "classical_milp"
    assert rec.problem_infeasible is False

    manifest = store.load_manifest(run_id)
    assert manifest is not None
    assert manifest["evidence_digest"] == "abc123"
    assert store.load_manifest("does-not-exist") is None


def test_env_selects_mlflow_backend(monkeypatch, tmp_path):
    pytest.importorskip("mlflow")
    from qf_agentos.governance.mlflow_store import MLflowEvidenceStore

    monkeypatch.setenv("QF_REGISTRY_BACKEND", "mlflow")
    monkeypatch.setenv("QF_MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    monkeypatch.setenv("QF_MLFLOW_EXPERIMENT", "qf-test-env")
    reset_settings_cache()
    try:
        assert isinstance(get_evidence_store(), MLflowEvidenceStore)
    finally:
        reset_settings_cache()
