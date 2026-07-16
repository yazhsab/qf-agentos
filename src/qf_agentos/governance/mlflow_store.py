"""MLflow-backed experiment registry (optional).

Implements the same :class:`EvidenceStoreProtocol` as the file store, logging each
solved run as an MLflow run — params + a metric for the objective gap, tags for the
decision / digest, and the manifest, report, and model card as text artifacts — so
runs are browsable in the MLflow UI and shareable across a team. Selected via
``QF_REGISTRY_BACKEND=mlflow``; requires the ``mlflow`` extra.

Uses ``MlflowClient`` directly (no global tracking-URI state), so a fresh store can
be created per request without leaking configuration between callers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from ..core.artifacts import EvidenceBundle
from .store import RunRecord

_RUN_ID_TAG = "qf.run_id"


class MLflowEvidenceStore:
    """Persists evidence bundles to an MLflow tracking server / directory."""

    def __init__(self, tracking_uri: str | None = None, experiment: str = "qf-agentos") -> None:
        try:
            from mlflow.tracking import MlflowClient
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "The MLflow registry requires the 'mlflow' extra: pip install 'qf-agentos[mlflow]'."
            ) from exc
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._experiment_id = self._ensure_experiment(experiment)

    def _ensure_experiment(self, name: str) -> str:
        exp = self._client.get_experiment_by_name(name)
        if exp is not None:
            return str(exp.experiment_id)
        return str(self._client.create_experiment(name))

    def save(self, run_id: str, bundle: EvidenceBundle) -> str:
        audit = bundle.manifest.get("audit") or {}
        problem = str(bundle.manifest.get("problem", ""))
        run = self._client.create_run(
            self._experiment_id,
            run_name=run_id,
            tags={
                _RUN_ID_TAG: run_id,
                "qf.decision": audit.get("category", "n/a"),
                "qf.recommended_method": audit.get("recommended_method", "n/a"),
                "qf.evidence_digest": bundle.manifest.get("evidence_digest", ""),
                "qf.problem_infeasible": str(bool(audit.get("problem_infeasible", False))),
                "qf.problem": problem,
            },
        )
        rid = run.info.run_id
        self._client.log_param(rid, "problem", problem)
        self._client.log_param(rid, "recommended_method", audit.get("recommended_method", "n/a"))
        gap = audit.get("objective_gap_pct")
        if gap is not None:
            self._client.log_metric(rid, "objective_gap_pct", float(gap))
        self._client.log_text(
            rid, json.dumps(bundle.manifest, indent=2, default=str), "manifest.json"
        )
        self._client.log_text(rid, bundle.report_md, "report.md")
        self._client.log_text(rid, bundle.model_card_md, "model_card.md")
        self._client.set_terminated(rid, "FINISHED")
        return str(rid)

    def list_runs(self) -> list[RunRecord]:
        runs = self._client.search_runs(
            [self._experiment_id], order_by=["attributes.start_time ASC"]
        )
        out: list[RunRecord] = []
        for r in runs:
            tags = r.data.tags
            if _RUN_ID_TAG not in tags:
                continue
            created = datetime.fromtimestamp((r.info.start_time or 0) / 1000, UTC).isoformat()
            out.append(
                RunRecord(
                    run_id=tags[_RUN_ID_TAG],
                    created_at=created,
                    decision=tags.get("qf.decision", "n/a"),
                    recommended_method=tags.get("qf.recommended_method", "n/a"),
                    evidence_digest=tags.get("qf.evidence_digest", ""),
                    problem_infeasible=tags.get("qf.problem_infeasible", "False") == "True",
                )
            )
        return out

    def load_manifest(self, run_id: str) -> dict[str, Any] | None:
        runs = self._client.search_runs(
            [self._experiment_id],
            filter_string=f"tags.`{_RUN_ID_TAG}` = '{run_id}'",
            max_results=1,
        )
        if not runs:
            return None
        import mlflow.artifacts

        text = mlflow.artifacts.load_text(f"{runs[0].info.artifact_uri}/manifest.json")
        data: dict[str, Any] = json.loads(text)
        return data


__all__ = ["MLflowEvidenceStore"]
