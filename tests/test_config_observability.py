"""Settings and structured logging/observability."""

from __future__ import annotations

import json
import logging

import pytest

from qf_agentos.core.config import Settings, get_settings, reset_settings_cache
from qf_agentos.core.observability import (
    configure_logging,
    get_logger,
    set_run_id,
)


def test_settings_defaults():
    s = get_settings()
    assert s.default_seed == 7
    assert s.statevector_qubit_limit > 0
    assert not s.has_ibm_credentials()


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("QF_DEFAULT_SEED", "99")
    monkeypatch.setenv("QF_LOG_FORMAT", "json")
    reset_settings_cache()
    try:
        s = get_settings()
        assert s.default_seed == 99
        assert s.log_format == "json"
    finally:
        reset_settings_cache()


def test_secret_not_leaked_in_repr(monkeypatch):
    monkeypatch.setenv("QF_IBM_TOKEN", "super-secret-token")
    s = Settings()
    assert s.has_ibm_credentials()
    assert "super-secret-token" not in repr(s)
    assert s.ibm_token.get_secret_value() == "super-secret-token"


def test_get_logger_namespaced():
    assert get_logger("foo").name == "qf_agentos.foo"
    assert get_logger("qf_agentos.bar").name == "qf_agentos.bar"


def test_json_logging_emits_run_id(capsys):
    configure_logging("INFO", "json", force=True)
    set_run_id("run-xyz")
    get_logger("test").info("hello world")
    err = capsys.readouterr().err.strip().splitlines()[-1]
    payload = json.loads(err)
    assert payload["run_id"] == "run-xyz"
    assert payload["msg"] == "hello world"
    assert payload["level"] == "INFO"
    set_run_id(None)
    # Reset to text for other tests.
    configure_logging("INFO", "text", force=True)


def test_configure_logging_idempotent():
    logging.getLogger("qf_agentos").handlers.clear()
    configure_logging("INFO", "text")
    n = len(logging.getLogger("qf_agentos").handlers)
    configure_logging("INFO", "text")  # no-op without force
    assert len(logging.getLogger("qf_agentos").handlers) == n


def test_span_is_noop_when_tracing_disabled():
    from qf_agentos.core.observability import configure_tracing, span

    configure_tracing(False)
    with span("noop", k="v") as s:
        assert s is None


def test_tracing_captures_spans_with_in_memory_exporter():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from qf_agentos.core.observability import configure_tracing, span

    exporter = InMemorySpanExporter()
    configure_tracing(True, exporter=exporter)
    try:
        with span("agent.test", **{"qf.step": "test"}):
            pass
        from opentelemetry import trace

        trace.get_tracer_provider().force_flush()
        names = {s.name for s in exporter.get_finished_spans()}
        assert "agent.test" in names
    finally:
        configure_tracing(False)
