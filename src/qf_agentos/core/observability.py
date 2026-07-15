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
from collections.abc import Iterator
from contextlib import contextmanager
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


# ---------------------------------------------------------------------------
# Optional OpenTelemetry tracing (per-agent-step spans)
# ---------------------------------------------------------------------------

_tracing_enabled = False
_tracer: Any = None
_provider_set = False
_sdk_provider: Any = None


def configure_tracing(
    enabled: bool, *, service_name: str = "qf-agentos", exporter: Any = None
) -> None:
    """Enable OpenTelemetry tracing if requested and the ``otel`` extra is present.

    A no-op when disabled; logs a warning (and stays disabled) if enabled without
    OpenTelemetry installed. Idempotent — the provider is installed once.
    """
    global _tracing_enabled, _tracer, _provider_set, _sdk_provider
    if not enabled:
        _tracing_enabled = False
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except Exception:
        get_logger("observability").warning(
            "QF_TRACING_ENABLED is set but OpenTelemetry is not installed "
            "(pip install 'qf-agentos[otel]'); tracing stays off."
        )
        _tracing_enabled = False
        return

    if not _provider_set:
        _sdk_provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        _sdk_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(_sdk_provider)
        _provider_set = True
    if exporter is not None and _sdk_provider is not None:  # e.g. in-memory exporter for tests
        _sdk_provider.add_span_processor(BatchSpanProcessor(exporter))
    _tracer = trace.get_tracer("qf_agentos")
    _tracing_enabled = True


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Start a tracing span if tracing is active; otherwise a no-op."""
    if not _tracing_enabled or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            current.set_attribute(key, value)
        yield current


__all__ = ["configure_logging", "configure_tracing", "get_logger", "set_run_id", "span"]
