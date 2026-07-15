"""Agent 1 — Financial Requirements.

Delegates problem-specific understanding to the domain, then surfaces discovered
gaps as warnings. In a hosted deployment this is where an LLM would elicit
clarifications; here the logic is deterministic and auditable.
"""

from __future__ import annotations

from ..core.observability import get_logger
from ..core.workflow import RunContext
from ..finance import get_domain

_logger = get_logger("agents.requirements")


def requirements_agent(ctx: RunContext) -> str:
    domain = get_domain(ctx.spec.problem)
    report = domain.requirements(ctx.spec)
    ctx.state.requirements = report
    for gap in report.discovered_gaps:
        ctx.warn(gap)
    _logger.debug("requirements: %s", report.summary)
    return f"Understood {ctx.spec.problem}: {report.summary}; {len(report.discovered_gaps)} gap(s) flagged."
