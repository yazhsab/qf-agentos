"""Deterministic workflow graph.

Rather than delegate control flow to an LLM, QF-AgentOS runs a fixed, ordered
sequence of typed steps. Determinism is a feature, not a limitation: the same
spec + the same seeds always produce the same evidence bundle, which is what a
regulated environment requires.

Each step is an ``Agent``: it reads and writes the shared, *typed* ``RunContext``
(``ctx.state``), appends a ``TraceEvent``, and never mutates the ProblemSpec.
Steps are isolated: if one raises, the failure is recorded and the run continues
so that partial evidence is still produced.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .artifacts import StepError
from .config import Settings, get_settings
from .errors import QFAgentOSError
from .ir import ProblemSpec
from .observability import get_logger, set_run_id, span
from .policy import PolicyEngine
from .state import PipelineState

_logger = get_logger("workflow")


@dataclass
class TraceEvent:
    step: str
    started_at: str
    duration_s: float
    summary: str
    ok: bool = True
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    """Shared, typed state threaded through the workflow."""

    spec: ProblemSpec
    policy: PolicyEngine
    run_id: str
    seed: int
    settings: Settings = field(default_factory=get_settings)
    state: PipelineState = field(default_factory=PipelineState)
    trace: list[TraceEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        _logger.warning(msg)


Agent = Callable[[RunContext], str]  # returns a short human summary of what it did


class Workflow:
    """An ordered list of named agent steps with per-step isolation."""

    def __init__(self, steps: list[tuple[str, Agent]]):
        self.steps = steps

    def run(self, ctx: RunContext, *, emit: Callable[[str, str], None] | None = None) -> RunContext:
        set_run_id(ctx.run_id)
        _logger.info("run started: problem=%s seed=%s", ctx.spec.problem, ctx.seed)
        for name, step in self.steps:
            started = datetime.now(UTC).isoformat()
            t0 = time.perf_counter()
            with span(f"agent.{name}", **{"qf.step": name, "qf.run_id": ctx.run_id}) as sp:
                try:
                    summary = step(ctx)
                    dt = time.perf_counter() - t0
                    ctx.trace.append(TraceEvent(name, started, dt, summary, ok=True))
                    _logger.info("step %-20s ok  (%.1f ms) — %s", name, dt * 1000, summary)
                    if sp is not None:
                        sp.set_attribute("qf.ok", True)
                        sp.set_attribute("qf.duration_ms", dt * 1000)
                    if emit is not None:
                        emit(name, summary)
                except Exception as exc:
                    dt = time.perf_counter() - t0
                    etype = type(exc).__name__
                    msg = str(exc) if isinstance(exc, QFAgentOSError) else f"{etype}: {exc}"
                    ctx.state.errors.append(StepError(step=name, error_type=etype, message=msg))
                    ctx.trace.append(
                        TraceEvent(name, started, dt, f"FAILED: {msg}", ok=False, error=msg)
                    )
                    _logger.error(
                        "step %-20s FAILED: %s",
                        name,
                        msg,
                        exc_info=not isinstance(exc, QFAgentOSError),
                    )
                    if sp is not None:
                        sp.set_attribute("qf.ok", False)
                        sp.set_attribute("qf.error", msg)
                    if emit is not None:
                        emit(name, f"[failed] {msg}")
        _logger.info(
            "run finished: %d step(s), %d error(s)", len(self.steps), len(ctx.state.errors)
        )
        return ctx


__all__ = ["Agent", "RunContext", "TraceEvent", "Workflow"]
