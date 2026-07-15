"""The default agent pipeline and the one-call ``solve`` entry point."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime

from .agents import (
    auditor_agent,
    classical_baseline_agent,
    execution_agent,
    formulation_agent,
    governance_agent,
    hardware_planner_agent,
    quantum_algorithm_agent,
    requirements_agent,
    verification_agent,
)
from .core.config import Settings, get_settings
from .core.ir import ProblemSpec
from .core.observability import configure_logging
from .core.policy import PolicyEngine
from .core.workflow import RunContext, Workflow


def build_default_pipeline() -> Workflow:
    """The reference 8-agent pipeline (Governance is the 9th, terminal, step)."""
    return Workflow(
        [
            ("requirements", requirements_agent),
            ("formulation", formulation_agent),
            ("classical_baseline", classical_baseline_agent),
            ("hardware_planner", hardware_planner_agent),
            ("quantum_algorithm", quantum_algorithm_agent),
            ("execution", execution_agent),
            ("verification", verification_agent),
            ("auditor", auditor_agent),
            ("governance", governance_agent),
        ]
    )


def make_run_id(spec: ProblemSpec) -> str:
    """A unique run id. The deterministic part is the content hash; the timestamp
    only guarantees uniqueness (reproducibility is checked via evidence_digest)."""
    digest = hashlib.sha256(spec.model_dump_json().encode()).hexdigest()[:8]
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"run-{stamp}-{digest}"


def solve(
    spec: ProblemSpec,
    *,
    human_approved: bool = False,
    emit: Callable[[str, str], None] | None = None,
    settings: Settings | None = None,
) -> RunContext:
    """Run the full pipeline on a spec and return the completed RunContext."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)
    policy = PolicyEngine(spec.execution_policy, human_approved=human_approved)
    ctx = RunContext(
        spec=spec,
        policy=policy,
        run_id=make_run_id(spec),
        seed=spec.execution_policy.seed,
        settings=settings,
    )
    build_default_pipeline().run(ctx, emit=emit)
    return ctx
