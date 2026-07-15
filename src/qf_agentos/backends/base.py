"""Backend abstraction.

Every QUBO solver — exact, heuristic, gate-model simulator, or real QPU — exposes
the same small surface so the Execution agent can select among them without
knowing vendor specifics. Real-hardware backends declare that they require
credentials and are gated behind autonomy L3 + approval by the policy engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..finance.collateral import Qubo


@dataclass(frozen=True)
class QuboRunConfig:
    """Run parameters common to QUBO solvers."""

    seed: int = 7
    shots: int = 4096
    reps: int = 2


@dataclass
class QuboSolution:
    """Uniform result of solving a QUBO."""

    best_bits: list[int]
    energy: float
    metadata: dict[str, Any] = field(default_factory=dict)
    qpu_time_s: float = 0.0
    cost_usd: float = 0.0


@runtime_checkable
class QuboSolver(Protocol):
    """The contract every backend implements."""

    name: str
    kind: str  # "classical" | "heuristic" | "quantum"
    requires_credentials: bool

    def is_available(self) -> tuple[bool, str]:
        """Return (available, human-readable detail)."""
        ...

    def solve(self, qubo: Qubo, config: QuboRunConfig) -> QuboSolution:
        """Solve the QUBO or raise :class:`~qf_agentos.core.errors.BackendError`."""
        ...


__all__ = ["QuboRunConfig", "QuboSolution", "QuboSolver"]
