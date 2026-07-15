"""Finance Intermediate Representation (Finance IR).

The IR is the contract every agent reads and writes. It turns a natural-language
or YAML financial requirement into a strongly-typed, validated specification so
that formulation, solving, verification and audit are all reproducible from the
same object.

This first release models the *collateral-allocation* problem. The schema is
intentionally small and explicit; new problem families extend it by adding
fields, never by loosening validation.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import SpecError

# Categorical Security attributes that may carry a concentration cap.
ALLOWED_CONCENTRATION_ATTRS: frozenset[str] = frozenset({"issuer", "counterparty"})

# Guard-rail against resource-exhaustion via absurdly large inputs.
MAX_INVENTORY = 100_000


class AutonomyLevel(str, Enum):
    """Human-control levels. Higher levels never bypass the gates below them."""

    L0 = "L0"  # Explain and recommend only.
    L1 = "L1"  # Generate an experiment plan.
    L2 = "L2"  # Execute simulators automatically.
    L3 = "L3"  # Execute paid QPU jobs after explicit approval.
    L4 = "L4"  # Recommend production decisions; human approval mandatory.

    @property
    def rank(self) -> int:
        return {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}[self.value]


class ObjectiveType(str, Enum):
    minimise_collateral_cost = "minimise_collateral_cost"


class Security(BaseModel):
    """One postable line item in the collateral inventory."""

    model_config = ConfigDict(extra="forbid")

    id: str
    issuer: str
    counterparty: str = "DEFAULT"
    market_value: float = Field(gt=0, description="Market value available to post (currency).")
    haircut: float = Field(ge=0, lt=1, description="Regulatory/valuation haircut in [0,1).")
    cost_bps: float = Field(ge=0, description="Cost of posting, in basis points of market value.")
    hqla: bool = Field(default=False, description="High-Quality Liquid Asset flag.")
    liquidity_score: float = Field(default=1.0, ge=0)
    eligible: bool = Field(default=True, description="Eligible to post to this counterparty.")

    @property
    def coverage(self) -> float:
        """Post-haircut collateral value if the whole line is posted."""
        return (1.0 - self.haircut) * self.market_value

    @property
    def cost(self) -> float:
        """Cost (currency) of posting the whole line."""
        return (self.cost_bps / 10_000.0) * self.market_value


class Objective(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: ObjectiveType = ObjectiveType.minimise_collateral_cost


class Constraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_collateral: float = Field(
        gt=0, description="Minimum post-haircut collateral value to deliver (currency)."
    )
    minimum_hqla: float = Field(
        default=0.0, ge=0, description="Minimum post-haircut HQLA value within the posted pool."
    )
    concentration: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-attribute concentration caps, e.g. {issuer: 0.10, counterparty: 0.15}. "
            "Each group's post-haircut value must be <= cap * total post-haircut value."
        ),
    )

    @field_validator("concentration")
    @classmethod
    def _validate_concentration(cls, v: dict[str, float]) -> dict[str, float]:
        for attr, frac in v.items():
            if attr not in ALLOWED_CONCENTRATION_ATTRS:
                allowed = ", ".join(sorted(ALLOWED_CONCENTRATION_ATTRS))
                raise ValueError(
                    f"concentration['{attr}'] is not a supported grouping attribute; "
                    f"use one of: {allowed}"
                )
            if not 0.0 < frac <= 1.0:
                raise ValueError(f"concentration['{attr}'] must be in (0, 1], got {frac}")
        return v


class ExecutionPolicy(BaseModel):
    """Policy the agents must obey. Governs backends, budget, and autonomy."""

    model_config = ConfigDict(extra="forbid")

    compare_classical: bool = True
    allow_quantum_annealing: bool = True
    allow_gate_model: bool = True
    max_qpu_budget_usd: float = Field(default=0.0, ge=0)
    # Upper bound guards against building an O(n^2) QUBO before the planner's
    # budget check; no statevector simulator handles anywhere near this many.
    max_effective_qubits: int = Field(default=20, gt=0, le=64)
    autonomy_level: AutonomyLevel = AutonomyLevel.L2

    # Gate-model hyper-parameters for the quantum sub-problem.
    qaoa_reps: int = Field(default=1, ge=1, le=8)
    shots: int = Field(default=4096, ge=128)
    seed: int = 7


class ProblemSpec(BaseModel):
    """Top-level, validated financial problem specification."""

    model_config = ConfigDict(extra="forbid")

    problem: str = "collateral_allocation"
    objective: Objective = Field(default_factory=Objective)
    constraints: Constraints
    execution_policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    inventory: list[Security] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> ProblemSpec:
        if self.problem != "collateral_allocation":
            raise ValueError(
                f"This release only implements 'collateral_allocation', got '{self.problem}'."
            )
        if len(self.inventory) > MAX_INVENTORY:
            raise ValueError(
                f"Inventory has {len(self.inventory)} securities; the maximum is {MAX_INVENTORY}."
            )
        ids = [s.id for s in self.inventory]
        if len(ids) != len(set(ids)):
            raise ValueError("Inventory security ids must be unique.")
        return self

    # ---- Derived, cached-free convenience views -------------------------------

    @property
    def eligible_inventory(self) -> list[Security]:
        return [s for s in self.inventory if s.eligible]

    @property
    def total_available_coverage(self) -> float:
        return sum(s.coverage for s in self.eligible_inventory)


def _format_validation_error(source: str, exc: ValidationError) -> str:
    """Render a Pydantic ValidationError as short, actionable lines.

    Deliberately does NOT dump the full input (which may contain sensitive
    position data) — only the field locations and messages.
    """
    lines = [f"Invalid problem specification ({source}): {exc.error_count()} error(s)."]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def parse_spec(data: dict[str, Any], *, source: str = "<dict>") -> ProblemSpec:
    """Validate a mapping into a ProblemSpec, raising :class:`SpecError` on failure."""
    if not isinstance(data, dict):
        raise SpecError(f"Spec must be a mapping, got {type(data).__name__} ({source}).")
    try:
        return ProblemSpec.model_validate(data)
    except ValidationError as exc:
        raise SpecError(_format_validation_error(source, exc)) from exc


def load_spec(path: str | Path) -> ProblemSpec:
    """Load and validate a ProblemSpec from a YAML file.

    Raises :class:`SpecError` with an actionable message (never a raw traceback)
    for missing files, malformed YAML, or validation failures.
    """
    p = Path(path)
    try:
        text = p.read_text()
    except FileNotFoundError as exc:
        raise SpecError(f"Spec file not found: {p}") from exc
    except OSError as exc:
        raise SpecError(f"Could not read spec file {p}: {exc}") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"Invalid YAML in {p}: {exc}") from exc

    if raw is None:
        raise SpecError(f"Spec file {p} is empty.")
    if not isinstance(raw, dict):
        raise SpecError(f"Spec file {p} did not parse to a mapping (got {type(raw).__name__}).")
    return parse_spec(raw, source=str(p))
