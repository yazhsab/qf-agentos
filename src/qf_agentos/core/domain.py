"""Problem-domain abstraction.

The agent pipeline is problem-agnostic: it delegates every problem-specific
operation to a domain. Two task types are supported:

* ``OPTIMIZATION`` — formulate → classical baseline → reduce to a qubit instance →
  build a QUBO → QAOA (collateral, payment routing). Implement ``ProblemDomain``.
* ``CLASSIFICATION`` — dataset → classical baselines → quantum-kernel model →
  temporal/leakage/significance verification (fraud detection). Implement
  ``ClassificationDomain``.

Shared infrastructure (workflow engine, policy, evidence layer, result types) is
reused by both; only the middle agents differ.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from numpy.typing import NDArray

from .artifacts import FormulationCatalogue, RequirementsReport
from .ir import ProblemSpec
from .result import SolveResult, VerificationReport

if TYPE_CHECKING:  # avoid a runtime core -> finance dependency
    from ..finance.collateral import Qubo


class TaskType(str, Enum):
    OPTIMIZATION = "optimization"
    CLASSIFICATION = "classification"


class DomainBase(ABC):
    """Common surface every domain exposes (understanding + formulation)."""

    problem: str
    task_type: TaskType

    @abstractmethod
    def requirements(self, spec: ProblemSpec) -> RequirementsReport: ...

    @abstractmethod
    def formulations(self, spec: ProblemSpec) -> FormulationCatalogue: ...


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


@runtime_checkable
class ProblemInstance(Protocol):
    """The common surface a reduced research instance exposes to the pipeline."""

    degenerate: bool
    provenance: dict[str, Any]

    @property
    def n_qubits(self) -> int: ...

    @property
    def target(self) -> float: ...


@dataclass
class ClassicalBaseline:
    """Full-problem classical results produced by an optimization domain."""

    milp: SolveResult
    lp: SolveResult | None = None
    integrality_gap: float | None = None


class ProblemDomain(DomainBase):
    """Everything the pipeline needs to solve one *optimization* problem family."""

    task_type = TaskType.OPTIMIZATION

    @abstractmethod
    def solve_classical_full(self, spec: ProblemSpec) -> ClassicalBaseline: ...

    @abstractmethod
    def reduce_to_instance(self, spec: ProblemSpec, max_qubits: int) -> ProblemInstance: ...

    @abstractmethod
    def build_qubo(self, instance: ProblemInstance, *, slack_bits: int) -> Qubo: ...

    @abstractmethod
    def solve_instance_classical(self, instance: ProblemInstance) -> SolveResult: ...

    @abstractmethod
    def evaluate_bits(
        self,
        instance: ProblemInstance,
        bits: NDArray[Any] | list[int],
        *,
        method: str,
        kind: str,
        backend: str,
        runtime_s: float = 0.0,
        qpu_time_s: float = 0.0,
        cost_usd: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> SolveResult: ...

    @abstractmethod
    def verify_full(self, spec: ProblemSpec, result: SolveResult) -> VerificationReport: ...

    @abstractmethod
    def verify_instance(
        self, instance: ProblemInstance, result: SolveResult
    ) -> VerificationReport: ...

    def instance_warm_start(self, instance: ProblemInstance, qubo: Qubo) -> list[float] | None:
        """Per-qubit biases in [0,1] (length qubo.n) from a classical relaxation,
        for warm-start QAOA. Default: no warm start."""
        return None


# ---------------------------------------------------------------------------
# Classification (quantum kernels)
# ---------------------------------------------------------------------------


class ClassificationDomain(DomainBase):
    """Everything the pipeline needs for a *classification* problem family.

    Dataset / split / model objects are domain-defined (typed ``Any`` here to keep
    the core free of a finance dependency); the concrete domain uses real types.
    """

    task_type = TaskType.CLASSIFICATION

    @abstractmethod
    def load_dataset(self, spec: ProblemSpec) -> Any: ...

    @abstractmethod
    def split(self, spec: ProblemSpec, dataset: Any) -> tuple[Any, Any]: ...

    @abstractmethod
    def classical_baselines(
        self, spec: ProblemSpec, dataset: Any, split: tuple[Any, Any]
    ) -> dict[str, Any]: ...

    @abstractmethod
    def plan_quantum(
        self, spec: ProblemSpec, dataset: Any, max_qubits: int, sim_available: bool
    ) -> dict[str, Any]: ...

    @abstractmethod
    def run_quantum(
        self, spec: ProblemSpec, dataset: Any, split: tuple[Any, Any], feature_plan: dict[str, Any]
    ) -> Any: ...

    @abstractmethod
    def verify(
        self,
        spec: ProblemSpec,
        dataset: Any,
        split: tuple[Any, Any],
        models: dict[str, Any],
    ) -> dict[str, VerificationReport]: ...


__all__ = [
    "ClassicalBaseline",
    "ClassificationDomain",
    "DomainBase",
    "ProblemDomain",
    "ProblemInstance",
    "TaskType",
]
