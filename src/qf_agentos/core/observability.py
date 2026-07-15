"""Structured logging and (optional) tracing for QF-AgentOS.

Production runs need machine-parseable logs correlated by ``run_id``. This module
provides a ``get_logger`` factory and a ``configure_logging`` entry point that
supports plain-text (human) and JSON (ingestion) formats. OpenTelemetry tracing
is optional and activated only when installed and enabled in settings.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

_run_id_var: ContextVar[str | None] = ContextVar("qf_run_id", default=None)
_CONFIGURED = False


class _RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id_var.get() or "-"
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", "-"),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "text", *, force: bool = False) -> None:
    """Configure the root ``qf_agentos`` logger. Idempotent unless ``force``."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    logger = logging.getLogger("qf_agentos")
    logger.handlers.clear()
    logger.setLevel(level.upper())
    logger.propagate = False

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.addFilter(_RunIdFilter())
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s [%(run_id)s] %(name)s: %(message)s")
        )
    logger.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``qf_agentos`` namespace."""
    if not name.startswith("qf_agentos"):
        name = f"qf_agentos.{name}"
    return logging.getLogger(name)


def set_run_id(run_id: str | None) -> None:
    """Bind a ``run_id`` to the current context so all logs carry it."""
    _run_id_var.set(run_id)


__all__ = ["configure_logging", "get_logger", "set_run_id"]
