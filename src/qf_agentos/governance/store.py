"""Evidence store — a minimal, file-based experiment registry.

Persists each run's evidence bundle under ``<root>/<run_id>/`` and maintains an
append-only ``index.jsonl`` so runs can be listed and looked up. This is the
persistence layer the governance design calls for; it is deliberately
dependency-free (a database can be swapped in behind the same interface).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..core.artifacts import EvidenceBundle

if TYPE_CHECKING:
    from ..core.config import Settings


@dataclass
class RunRecord:
    run_id: str
    created_at: str
    decision: str
    recommended_method: str
    evidence_digest: str
    problem_infeasible: bool


@runtime_checkable
class EvidenceStoreProtocol(Protocol):
    """The persistence interface the pipeline/API depend on. Both the file store
    and the MLflow registry implement it, so they are drop-in interchangeable."""

    def save(self, run_id: str, bundle: EvidenceBundle) -> Any: ...

    def list_runs(self) -> list[RunRecord]: ...

    def load_manifest(self, run_id: str) -> dict[str, Any] | None: ...


class EvidenceStore:
    """Reads and writes evidence bundles and the run index."""

    def __init__(self, root: Path | str = "evidence") -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.jsonl"

    def save(self, run_id: str, bundle: EvidenceBundle) -> Path:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps(bundle.manifest, indent=2, default=str))
        (run_dir / "report.md").write_text(bundle.report_md)
        (run_dir / "model_card.md").write_text(bundle.model_card_md)

        audit = bundle.manifest.get("audit") or {}
        record = RunRecord(
            run_id=run_id,
            created_at=datetime.now(UTC).isoformat(),
            decision=audit.get("category", "n/a"),
            recommended_method=audit.get("recommended_method", "n/a"),
            evidence_digest=bundle.manifest.get("evidence_digest", ""),
            problem_infeasible=bool(audit.get("problem_infeasible", False)),
        )
        self.root.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a") as fh:
            fh.write(json.dumps(record.__dict__) + "\n")
        return run_dir

    def list_runs(self) -> list[RunRecord]:
        if not self.index_path.exists():
            return []
        records: list[RunRecord] = []
        for line in self.index_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(RunRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
        return records

    def load_manifest(self, run_id: str) -> dict[str, Any] | None:
        path = self.root / run_id / "manifest.json"
        if not path.exists():
            return None
        data: dict[str, Any] = json.loads(path.read_text())
        return data


def get_evidence_store(settings: Settings | None = None) -> EvidenceStoreProtocol:
    """Return the configured evidence store: the file store (default) or the
    MLflow registry when ``QF_REGISTRY_BACKEND=mlflow``."""
    from ..core.config import get_settings

    settings = settings or get_settings()
    if settings.registry_backend == "mlflow":
        from .mlflow_store import MLflowEvidenceStore

        return MLflowEvidenceStore(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment=settings.mlflow_experiment,
        )
    if settings.registry_backend == "postgres":
        from .postgres_store import PostgresEvidenceStore

        if not settings.postgres_dsn:
            raise RuntimeError("QF_REGISTRY_BACKEND=postgres requires QF_POSTGRES_DSN.")
        return PostgresEvidenceStore(settings.postgres_dsn)
    return EvidenceStore(settings.evidence_dir)


__all__ = ["EvidenceStore", "EvidenceStoreProtocol", "RunRecord", "get_evidence_store"]
