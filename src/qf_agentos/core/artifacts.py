"""Typed artifacts exchanged between agents.

These replace the previous stringly-typed ``ctx.store`` dictionary. Every value
an agent produces has a schema, so the inter-agent contract is explicit,
mypy-checkable, and serialises cleanly into the evidence bundle.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RequirementsReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_securities: int
    n_eligible: int
    required_collateral: float
    available_coverage: float
    coverage_headroom: float
    trivially_feasible_upper_bound: bool
    concentration_attrs: list[str] = Field(default_factory=list)
    minimum_hqla: float = 0.0
    discovered_gaps: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    autonomy_level: str = "L2"


class Formulation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str
    variables: str
    represents: str
    note: str


class FormulationCatalogue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    catalogue: list[Formulation] = Field(default_factory=list)
    selected_classical: str
    selected_quantum_path: str
    encoding_loss_note: str


class BackendCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    available: bool
    detail: str


class HardwarePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_qubits: int
    qubo_density: float
    target: str | None
    abstain: bool
    reasons: list[str] = Field(default_factory=list)
    estimated_two_qubit_depth: int
    estimated_cost_usd: float
    real_qpu: str
    capabilities: list[BackendCapability] = Field(default_factory=list)
    encoding_losses: list[str] = Field(default_factory=list)
    instance_provenance: dict[str, Any] = Field(default_factory=dict)
    instance_target_collateral: float


class QuantumSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: str | None
    reps: int | None = None
    mixer: str | None = None
    optimizer: str | None = None
    warm_start: str | None = None
    alternatives_considered: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    reason: str | None = None


class TranspileMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_depth: int
    two_qubit_depth: int
    cx_count: int
    basis_gates: list[str] = Field(default_factory=list)


class QaoaResult(BaseModel):
    """Structured return of the QAOA backend (numpy converted to plain types)."""

    model_config = ConfigDict(extra="forbid")

    degenerate: bool = False
    best_bits: list[int] = Field(default_factory=list)
    best_energy: float = 0.0
    n_qubits: int = 0
    reps: int = 1
    expectation_ising: float | None = None
    num_parameters: int | None = None
    optimizer: str | None = None
    optimizer_evals: int | None = None
    restarts: int | None = None
    shots: int | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    sample_mean_energy: float | None = None
    transpile: TranspileMetrics | None = None
    backend: str = "gate_model_statevector_sim"


class ReproducibilityInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deterministic: bool
    seed: int
    note: str
    evidence_digest: str | None = None


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: dict[str, Any]
    report_md: str
    model_card_md: str


class StepError(BaseModel):
    """Records a non-fatal failure of a single agent step (run continues)."""

    model_config = ConfigDict(extra="forbid")

    step: str
    error_type: str
    message: str


__all__ = [
    "BackendCapability",
    "EvidenceBundle",
    "Formulation",
    "FormulationCatalogue",
    "HardwarePlan",
    "QaoaResult",
    "QuantumSelection",
    "ReproducibilityInfo",
    "RequirementsReport",
    "StepError",
    "TranspileMetrics",
]
