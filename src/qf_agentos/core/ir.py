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
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import SpecError

# Categorical Security attributes that may carry a concentration cap.
ALLOWED_CONCENTRATION_ATTRS: frozenset[str] = frozenset({"issuer", "counterparty"})

# Guard-rails against resource-exhaustion via absurdly large inputs.
MAX_INVENTORY = 100_000
MAX_TRANSACTIONS = 100_000
MAX_ROUTES = 1_000
MAX_SAMPLES = 2_000  # classification: RBF/quantum kernels are O(n^2)+
MAX_OBLIGATIONS = 100_000
MAX_PARTICIPANTS = 10_000


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
    minimise_routing_cost = "minimise_routing_cost"
    maximise_detection_performance = "maximise_detection_performance"
    maximise_settled_value = "maximise_settled_value"


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


# ---------------------------------------------------------------------------
# Payment-routing problem family
# ---------------------------------------------------------------------------


class PaymentRoute(BaseModel):
    """A candidate route (acquirer / processor / network path) for payments."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""
    cost_bps: float = Field(default=0.0, ge=0, description="Processing cost, bps of amount.")
    fixed_fee: float = Field(default=0.0, ge=0, description="Per-transaction fixed fee (currency).")
    approval_rate: float = Field(gt=0, le=1, description="Probability of approval on this route.")
    fraud_bps: float = Field(default=0.0, ge=0, description="Expected fraud loss, bps of amount.")
    latency_ms: float = Field(default=0.0, ge=0)
    capacity: int = Field(
        default=1_000_000, ge=0, description="Max transactions this route accepts."
    )
    network: str = Field(default="DEFAULT", description="Network/rail, for diversification limits.")


class Transaction(BaseModel):
    """One payment transaction to route."""

    model_config = ConfigDict(extra="forbid")

    id: str
    amount: float = Field(gt=0)
    eligible_routes: list[str] = Field(
        default_factory=list,
        description="Route ids this transaction may use; empty means all routes.",
    )


class RoutingConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decline_penalty_bps: float = Field(
        default=0.0, ge=0, description="Cost of a decline (lost margin/retry), bps of amount."
    )
    latency_weight: float = Field(
        default=0.0, ge=0, description="Currency cost per millisecond of latency."
    )
    network_concentration: float | None = Field(
        default=None, description="Max share of transactions routed to any one network, in (0,1]."
    )
    min_overall_approval: float | None = Field(
        default=None, description="Floor on the portfolio expected approval rate, in (0,1]."
    )

    @field_validator("network_concentration", "min_overall_approval")
    @classmethod
    def _in_unit_interval(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 < v <= 1.0:
            raise ValueError(f"must be in (0, 1], got {v}")
        return v


# ---------------------------------------------------------------------------
# Settlement / liquidity-saving problem family (RTGS gridlock resolution)
# ---------------------------------------------------------------------------


class Participant(BaseModel):
    """A settlement-system participant with available intraday liquidity."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""
    balance: float = Field(
        default=0.0,
        ge=0,
        description="Available liquidity to fund net outgoing settlements (currency).",
    )


class Obligation(BaseModel):
    """One queued payment obligation from a payer to a payee."""

    model_config = ConfigDict(extra="forbid")

    id: str
    payer: str = Field(description="Participant id that owes the payment.")
    payee: str = Field(description="Participant id that receives the payment.")
    amount: float = Field(gt=0, description="Payment amount (currency).")
    priority: int = Field(
        default=0, ge=0, description="Higher settles first when the objective ties."
    )

    @model_validator(mode="after")
    def _distinct_parties(self) -> Obligation:
        if self.payer == self.payee:
            raise ValueError(
                f"Obligation '{self.id}' has the same payer and payee ('{self.payer}')."
            )
        return self


class SettlementConfig(BaseModel):
    """Configuration for a liquidity-saving settlement batch."""

    model_config = ConfigDict(extra="forbid")

    penalty_scale: float = Field(
        default=8.0, gt=0, description="Base QUBO penalty weight for the liquidity constraints."
    )
    min_settled_ratio: float | None = Field(
        default=None,
        description="Optional floor on settled value / total queued value, in (0, 1].",
    )

    @field_validator("min_settled_ratio")
    @classmethod
    def _in_unit_interval(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 < v <= 1.0:
            raise ValueError(f"must be in (0, 1], got {v}")
        return v


# ---------------------------------------------------------------------------
# Classification problem family (quantum kernels)
# ---------------------------------------------------------------------------


class SyntheticConfig(BaseModel):
    """Deterministic synthetic-data generator for a classification demo."""

    model_config = ConfigDict(extra="forbid")

    n_samples: int = Field(default=240, ge=20, le=MAX_SAMPLES)
    n_features: int = Field(default=6, ge=1, le=32)
    n_informative: int = Field(default=3, ge=1)
    class_balance: float = Field(default=0.2, gt=0, lt=1, description="Positive-class rate.")
    separability: float = Field(default=0.9, ge=0)


class ClassificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_metric: Literal["auc", "accuracy", "f1"] = "auc"
    test_fraction: float = Field(default=0.3, gt=0, lt=1, description="Temporal holdout fraction.")
    feature_budget: int = Field(
        default=4, ge=1, le=16, description="Qubits = features for the kernel."
    )
    rbf_gamma: float = Field(default=0.3, gt=0)
    ridge_lambda: float = Field(default=0.01, gt=0)
    bootstrap: int = Field(default=500, ge=50, le=5000)
    synthetic: SyntheticConfig | None = None


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

    # Where the gate-model QAOA runs. "sim" = local statevector (default, free,
    # L2). "ibm" routes the FINAL optimised circuit to real IBM hardware via
    # Qiskit Runtime — requires credentials, autonomy L3, human approval, and
    # falls back to the simulator (with a note) if IBM is unavailable.
    qpu_backend: Literal["sim", "ibm"] = "sim"

    # Noisy-simulation pass (completes the execution ladder). Off by default so
    # the ideal-vs-classical comparison and evidence digest are unchanged; when
    # on, the report shows how the QAOA result degrades on present-day hardware.
    noisy_simulation: bool = False
    noise_two_qubit_error: float = Field(default=0.02, ge=0, le=0.5)
    readout_error: float = Field(default=0.03, ge=0, lt=0.5)


class ProblemSpec(BaseModel):
    """Top-level, validated financial problem specification.

    Problem-family fields are optional; the ``problem`` discriminator selects
    which block is required and validated. New families add their own block.
    """

    model_config = ConfigDict(extra="forbid")

    problem: str = "collateral_allocation"
    objective: Objective = Field(default_factory=Objective)
    execution_policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)

    # collateral_allocation
    constraints: Constraints | None = None
    inventory: list[Security] = Field(default_factory=list)

    # payment_routing
    routing: RoutingConstraints | None = None
    transactions: list[Transaction] = Field(default_factory=list)
    routes: list[PaymentRoute] = Field(default_factory=list)

    # settlement_netting
    settlement: SettlementConfig | None = None
    participants: list[Participant] = Field(default_factory=list)
    obligations: list[Obligation] = Field(default_factory=list)

    # fraud_detection (classification)
    classification: ClassificationConfig | None = None
    features: list[list[float]] = Field(default_factory=list)
    labels: list[int] = Field(default_factory=list)
    timestamps: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> ProblemSpec:
        if self.problem == "collateral_allocation":
            if self.constraints is None:
                raise ValueError("collateral_allocation requires a 'constraints' block.")
            if len(self.inventory) > MAX_INVENTORY:
                raise ValueError(
                    f"Inventory has {len(self.inventory)} securities; the maximum is {MAX_INVENTORY}."
                )
            ids = [s.id for s in self.inventory]
            if len(ids) != len(set(ids)):
                raise ValueError("Inventory security ids must be unique.")
        elif self.problem == "payment_routing":
            if self.routing is None:
                raise ValueError("payment_routing requires a 'routing' block.")
            if not self.routes:
                raise ValueError("payment_routing requires at least one route.")
            if len(self.routes) > MAX_ROUTES:
                raise ValueError(
                    f"Too many routes ({len(self.routes)}); the maximum is {MAX_ROUTES}."
                )
            if len(self.transactions) > MAX_TRANSACTIONS:
                raise ValueError(
                    f"Too many transactions ({len(self.transactions)}); the maximum is {MAX_TRANSACTIONS}."
                )
            rids = [r.id for r in self.routes]
            if len(rids) != len(set(rids)):
                raise ValueError("Route ids must be unique.")
            tids = [t.id for t in self.transactions]
            if len(tids) != len(set(tids)):
                raise ValueError("Transaction ids must be unique.")
            route_set = set(rids)
            for t in self.transactions:
                for rid in t.eligible_routes:
                    if rid not in route_set:
                        raise ValueError(
                            f"Transaction '{t.id}' lists unknown eligible route '{rid}'."
                        )
        elif self.problem == "settlement_netting":
            if self.settlement is None:
                raise ValueError("settlement_netting requires a 'settlement' block.")
            if not self.participants:
                raise ValueError("settlement_netting requires at least one participant.")
            if not self.obligations:
                raise ValueError("settlement_netting requires at least one obligation.")
            if len(self.participants) > MAX_PARTICIPANTS:
                raise ValueError(
                    f"Too many participants ({len(self.participants)}); "
                    f"the maximum is {MAX_PARTICIPANTS}."
                )
            if len(self.obligations) > MAX_OBLIGATIONS:
                raise ValueError(
                    f"Too many obligations ({len(self.obligations)}); "
                    f"the maximum is {MAX_OBLIGATIONS}."
                )
            pids = [p.id for p in self.participants]
            if len(pids) != len(set(pids)):
                raise ValueError("Participant ids must be unique.")
            oids = [o.id for o in self.obligations]
            if len(oids) != len(set(oids)):
                raise ValueError("Obligation ids must be unique.")
            pset = set(pids)
            for o in self.obligations:
                if o.payer not in pset:
                    raise ValueError(f"Obligation '{o.id}' has unknown payer '{o.payer}'.")
                if o.payee not in pset:
                    raise ValueError(f"Obligation '{o.id}' has unknown payee '{o.payee}'.")
        elif self.problem in ("fraud_detection", "rfq_fill"):
            cfg = self.classification
            if cfg is None:
                raise ValueError(f"{self.problem} requires a 'classification' block.")
            has_inline = bool(self.features) and bool(self.labels)
            if not has_inline and cfg.synthetic is None:
                raise ValueError(
                    f"{self.problem} needs inline 'features'+'labels' or a "
                    "'classification.synthetic' block."
                )
            if has_inline:
                if len(self.features) != len(self.labels):
                    raise ValueError("features and labels must have the same length.")
                if len(self.features) > MAX_SAMPLES:
                    raise ValueError(f"Too many samples; the maximum is {MAX_SAMPLES}.")
                widths = {len(row) for row in self.features}
                if len(widths) > 1:
                    raise ValueError("All feature rows must have the same width.")
                if set(self.labels) - {0, 1}:
                    raise ValueError("labels must be binary (0/1).")
                n_features = widths.pop() if widths else 0
                if self.timestamps and len(self.timestamps) != len(self.labels):
                    raise ValueError("timestamps must match the number of samples.")
            else:
                assert cfg.synthetic is not None
                n_features = cfg.synthetic.n_features
            if cfg.feature_budget > n_features:
                raise ValueError(
                    f"feature_budget ({cfg.feature_budget}) exceeds available features "
                    f"({n_features})."
                )
        else:
            raise ValueError(
                f"Unknown problem '{self.problem}'. Known: collateral_allocation, "
                "payment_routing, settlement_netting, fraud_detection, rfq_fill."
            )
        return self

    # ---- Derived, cached-free convenience views (collateral) ------------------

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
