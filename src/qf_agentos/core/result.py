"""Typed results and evidence artifacts shared across agents.

Every solver returns a `SolveResult`; every constraint check is a
`ConstraintCheck`; the auditor emits an `AuditDecision`. Keeping these as
Pydantic models means the whole run serialises cleanly into the evidence
bundle with no bespoke JSON glue.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Allocation(BaseModel):
    """A collateral decision: fraction of each security's market value to post."""

    model_config = ConfigDict(extra="forbid")

    x: dict[str, float] = Field(default_factory=dict)  # security id -> fraction in [0,1]

    def posted(self, tol: float = 1e-6) -> dict[str, float]:
        return {k: v for k, v in self.x.items() if v > tol}


class SolveResult(BaseModel):
    """The outcome of running one method on one (sub-)problem."""

    model_config = ConfigDict(extra="forbid")

    method: str  # e.g. "classical_milp", "qubo_bruteforce", "simulated_annealing", "qaoa_sim"
    kind: str  # "classical" | "heuristic" | "quantum"
    backend: str  # human-readable backend id
    scope: str  # "full_problem" | "research_instance"
    feasible: bool = False
    objective: float | None = None  # posting cost (currency); lower is better
    allocation: Allocation | None = None
    runtime_s: float = 0.0
    qpu_time_s: float = 0.0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConstraintCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    satisfied: bool
    value: float
    limit: float
    slack: float  # signed room to spare; negative means violated
    detail: str = ""


class VerificationReport(BaseModel):
    """Deterministic verification of a candidate solution. Never trusts a solver."""

    model_config = ConfigDict(extra="forbid")

    method: str
    scope: str
    feasible: bool
    recomputed_objective: float | None = None
    objective_matches_solver: bool = True
    checks: list[ConstraintCheck] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    # Quantum-contribution accounting (populated only for quantum methods).
    quantum_contribution: dict[str, Any] | None = None


class DecisionCategory(str, Enum):
    CLASSICAL_PREFERRED = "CLASSICAL PREFERRED"
    QUANTUM_NOT_FEASIBLE = "QUANTUM NOT FEASIBLE ON PRESENT HARDWARE"
    QUANTUM_PARITY = "QUANTUM PARITY"
    QUANTUM_RESEARCH_CANDIDATE = "QUANTUM RESEARCH CANDIDATE"
    QUANTUM_IMPROVEMENT_OBSERVED = "QUANTUM IMPROVEMENT OBSERVED"
    INDEPENDENT_REPRODUCTION_REQUIRED = "INDEPENDENT REPRODUCTION REQUIRED"
    POTENTIAL_OPERATIONAL_ADVANTAGE = "POTENTIAL OPERATIONAL ADVANTAGE"


class AuditDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: DecisionCategory
    recommended_method: str
    rationale: list[str] = Field(default_factory=list)
    classical: SolveResult | None = None
    quantum: SolveResult | None = None
    objective_gap_pct: float | None = None  # (quantum - classical) / |classical|
    problem_infeasible: bool = False  # the full problem admits no feasible solution
    rendered: str = ""  # the human-facing FINAL DECISION block
