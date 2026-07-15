"""Problem-domain abstraction.

The agent pipeline is problem-agnostic: it delegates every problem-specific
operation (formulate, build baselines, reduce to a qubit-sized instance, build
the QUBO, decode + verify) to a :class:`ProblemDomain`. This is how a new problem
family (payment routing, RFQ, fraud, …) plugs in without touching the agents.

The QUBO, the QUBO solvers, the quantum-contribution accounting, and the auditor
are already generic and are shared across all domains.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from numpy.typing import NDArray

from .artifacts import FormulationCatalogue, RequirementsReport
from .ir import ProblemSpec
from .result import SolveResult, VerificationReport

if TYPE_CHECKING:  # avoid a runtime core -> finance dependency
    from ..finance.collateral import Qubo


@runtime_checkable
class ProblemInstance(Protocol):
    """The common surface a reduced research instance exposes to the pipeline."""

    degenerate: bool
    provenance: dict[str, Any]

    @property
    def n_qubits(self) -> int: ...

    @property
    def target(self) -> float:
        """A representative target magnitude for reporting (domain-defined)."""
        ...


@dataclass
class ClassicalBaseline:
    """Full-problem classical results produced by a domain."""

    milp: SolveResult
    lp: SolveResult | None = None
    integrality_gap: float | None = None


class ProblemDomain(ABC):
    """Everything the pipeline needs to solve one problem family."""

    problem: str

    # --- Understanding & formulation --------------------------------------
    @abstractmethod
    def requirements(self, spec: ProblemSpec) -> RequirementsReport: ...

    @abstractmethod
    def formulations(self, spec: ProblemSpec) -> FormulationCatalogue: ...

    # --- Full-problem classical baseline ----------------------------------
    @abstractmethod
    def solve_classical_full(self, spec: ProblemSpec) -> ClassicalBaseline: ...

    # --- Reduction to a quantum-sized instance ----------------------------
    @abstractmethod
    def reduce_to_instance(self, spec: ProblemSpec, max_qubits: int) -> ProblemInstance: ...

    @abstractmethod
    def build_qubo(self, instance: ProblemInstance, *, slack_bits: int) -> Qubo: ...

    # --- Instance-level solving & decoding --------------------------------
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

    # --- Deterministic verification ---------------------------------------
    @abstractmethod
    def verify_full(self, spec: ProblemSpec, result: SolveResult) -> VerificationReport: ...

    @abstractmethod
    def verify_instance(
        self, instance: ProblemInstance, result: SolveResult
    ) -> VerificationReport: ...


__all__ = ["ClassicalBaseline", "ProblemDomain", "ProblemInstance"]
