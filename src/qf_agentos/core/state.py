"""The typed pipeline blackboard.

Agents read and write ``ctx.state`` (a :class:`PipelineState`) instead of a
loose ``dict[str, Any]``. Fields are optional because the pipeline is resilient:
a step may be skipped or fail, and downstream agents must tolerate missing
predecessors gracefully rather than raising ``KeyError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .artifacts import (
    EvidenceBundle,
    FormulationCatalogue,
    HardwarePlan,
    QaoaResult,
    QuantumSelection,
    ReproducibilityInfo,
    RequirementsReport,
    StepError,
)
from .domain import ProblemInstance
from .result import AuditDecision, SolveResult, VerificationReport

if TYPE_CHECKING:  # avoid a runtime core -> finance dependency
    from ..finance.collateral import Qubo


@dataclass
class PipelineState:
    """Typed artifacts produced across the pipeline."""

    # Agent 1-2
    requirements: RequirementsReport | None = None
    formulations: FormulationCatalogue | None = None

    # Agent 3 — full-problem classical baseline
    classical_lp: SolveResult | None = None
    classical_milp: SolveResult | None = None
    integrality_gap: float | None = None

    # Agents 4-5 — reduction + planning
    instance: ProblemInstance | None = None
    qubo: Qubo | None = None
    hardware_plan: HardwarePlan | None = None
    quantum_selection: QuantumSelection | None = None

    # Agent 6 — instance-level execution
    instance_milp: SolveResult | None = None
    instance_qubo_exact: SolveResult | None = None
    instance_sa: SolveResult | None = None
    instance_qaoa: SolveResult | None = None
    instance_qaoa_noisy: SolveResult | None = None
    qaoa_raw: QaoaResult | None = None
    qubo_exact_energy: float | None = None

    # Classification task (fraud_detection): dataset, split, models, feature plan.
    # Typed loosely to keep core free of a finance dependency.
    dataset: Any = None
    split: Any = None
    feature_plan: dict[str, Any] | None = None
    class_models: dict[str, Any] = field(default_factory=dict)

    # Agents 7-9 — verification, audit, governance
    verification: dict[str, VerificationReport] = field(default_factory=dict)
    reproducibility: ReproducibilityInfo | None = None
    audit: AuditDecision | None = None
    bundle: EvidenceBundle | None = None

    # Non-fatal step failures (the run continues and still emits evidence)
    errors: list[StepError] = field(default_factory=list)


__all__ = ["PipelineState"]
