"""SQL-backed evidence registry (optional).

Implements the same :class:`EvidenceStoreProtocol` as the file store on a SQL
database via SQLAlchemy Core — Postgres in production, SQLite in tests, identical
code. Selected via ``QF_REGISTRY_BACKEND=postgres`` + ``QF_POSTGRES_DSN``; requires
the ``postgres`` extra (SQLAlchemy + psycopg). The manifest, report, and model card
are stored inline so a run is fully recoverable from one row.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from ..core.artifacts import EvidenceBundle
from .store import RunRecord


class PostgresEvidenceStore:
    """Persists evidence bundles to a SQL database (Postgres / SQLite)."""

    def __init__(self, dsn: str) -> None:
        try:
            from sqlalchemy import (
                Boolean,
                Column,
                MetaData,
                String,
                Table,
                Text,
                create_engine,
            )
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "The Postgres registry requires the 'postgres' extra: "
                "pip install 'qf-agentos[postgres]'."
            ) from exc
        self._engine = create_engine(dsn)
        self._metadata = MetaData()
        self._runs = Table(
            "qf_runs",
            self._metadata,
            Column("run_id", String, primary_key=True),
            Column("created_at", String, nullable=False),
            Column("decision", String, nullable=False),
            Column("recommended_method", String, nullable=False),
            Column("evidence_digest", String, nullable=False),
            Column("problem_infeasible", Boolean, nullable=False),
            Column("manifest", Text, nullable=False),
            Column("report_md", Text, nullable=False),
            Column("model_card_md", Text, nullable=False),
        )
        self._metadata.create_all(self._engine)

    def save(self, run_id: str, bundle: EvidenceBundle) -> str:
        audit = bundle.manifest.get("audit") or {}
        values = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "decision": audit.get("category", "n/a"),
            "recommended_method": audit.get("recommended_method", "n/a"),
            "evidence_digest": bundle.manifest.get("evidence_digest", ""),
            "problem_infeasible": bool(audit.get("problem_infeasible", False)),
            "manifest": json.dumps(bundle.manifest, default=str),
            "report_md": bundle.report_md,
            "model_card_md": bundle.model_card_md,
        }
        # Idempotent upsert that works on every dialect: delete-then-insert.
        with self._engine.begin() as conn:
            conn.execute(self._runs.delete().where(self._runs.c.run_id == run_id))
            conn.execute(self._runs.insert().values(**values))
        return run_id

    def list_runs(self) -> list[RunRecord]:
        from sqlalchemy import select

        with self._engine.connect() as conn:
            rows = (
                conn.execute(select(self._runs).order_by(self._runs.c.created_at)).mappings().all()
            )
        return [
            RunRecord(
                run_id=r["run_id"],
                created_at=r["created_at"],
                decision=r["decision"],
                recommended_method=r["recommended_method"],
                evidence_digest=r["evidence_digest"],
                problem_infeasible=bool(r["problem_infeasible"]),
            )
            for r in rows
        ]

    def load_manifest(self, run_id: str) -> dict[str, Any] | None:
        from sqlalchemy import select

        with self._engine.connect() as conn:
            val = conn.execute(
                select(self._runs.c.manifest).where(self._runs.c.run_id == run_id)
            ).scalar_one_or_none()
        if val is None:
            return None
        data: dict[str, Any] = json.loads(val)
        return data


__all__ = ["PostgresEvidenceStore"]
