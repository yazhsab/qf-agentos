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
from .agents.classification import (
    classification_auditor_agent,
    classification_baseline_agent,
    classification_execution_agent,
    classification_planner_agent,
    classification_verification_agent,
)
from .core.config import Settings, get_settings
from .core.domain import TaskType
from .core.ir import ProblemSpec
from .core.observability import configure_logging, configure_tracing
from .core.policy import PolicyEngine
from .core.workflow import RunContext, Workflow
from .finance import get_domain


def build_default_pipeline() -> Workflow:
    """The reference optimization pipeline (Governance is the terminal step)."""
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


def build_classification_pipeline() -> Workflow:
    """The classification pipeline (quantum kernels). Reuses Requirements,
    Formulation, and Governance; specialises the middle agents."""
    return Workflow(
        [
            ("requirements", requirements_agent),
            ("formulation", formulation_agent),
            ("classical_baseline", classification_baseline_agent),
            ("quantum_planner", classification_planner_agent),
            ("execution", classification_execution_agent),
            ("verification", classification_verification_agent),
            ("auditor", classification_auditor_agent),
            ("governance", governance_agent),
        ]
    )


def pipeline_for(spec: ProblemSpec) -> Workflow:
    """Select the pipeline by the problem's task type."""
    if get_domain(spec.problem).task_type == TaskType.CLASSIFICATION:
        return build_classification_pipeline()
    return build_default_pipeline()


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
    configure_tracing(settings.tracing_enabled)
    policy = PolicyEngine(spec.execution_policy, human_approved=human_approved)
    ctx = RunContext(
        spec=spec,
        policy=policy,
        run_id=make_run_id(spec),
        seed=spec.execution_policy.seed,
        settings=settings,
    )
    pipeline_for(spec).run(ctx, emit=emit)
    return ctx
