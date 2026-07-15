"""Workflow engine: per-step exception isolation and tracing."""

from __future__ import annotations

from qf_agentos.core.errors import BackendError
from qf_agentos.core.policy import PolicyEngine
from qf_agentos.core.workflow import RunContext, Workflow
from qf_test_utils import make_spec


def _ctx() -> RunContext:
    spec = make_spec()
    return RunContext(spec=spec, policy=PolicyEngine(spec.execution_policy), run_id="t", seed=7)


def test_failing_step_is_isolated_and_run_continues():
    ctx = _ctx()
    order: list[str] = []

    def ok(ctx):
        order.append("ok")
        return "ok"

    def boom(ctx):
        raise ValueError("boom")

    def after(ctx):
        order.append("after")
        ctx.state.integrality_gap = 1.0
        return "after"

    Workflow([("ok", ok), ("boom", boom), ("after", after)]).run(ctx)

    assert order == ["ok", "after"]  # the step after the failure still ran
    assert len(ctx.state.errors) == 1
    assert ctx.state.errors[0].step == "boom"
    assert ctx.state.errors[0].error_type == "ValueError"
    assert ctx.state.integrality_gap == 1.0
    # Trace records both success and failure.
    assert any(e.ok for e in ctx.trace) and any(not e.ok for e in ctx.trace)


def test_domain_error_message_is_preserved():
    ctx = _ctx()

    def bad(ctx):
        raise BackendError("solver exploded")

    Workflow([("bad", bad)]).run(ctx)
    assert "solver exploded" in ctx.state.errors[0].message


def test_emit_receives_failure_notice():
    ctx = _ctx()
    seen: list[tuple[str, str]] = []

    def bad(ctx):
        raise RuntimeError("nope")

    Workflow([("bad", bad)]).run(ctx, emit=lambda n, s: seen.append((n, s)))
    assert seen and "failed" in seen[0][1].lower()
