"""Shared test helpers (uniquely named to avoid a stray site-packages ``tests``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qf_agentos.core.ir import ProblemSpec

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "collateral-allocation.yaml"


def default_inventory() -> list[dict[str, Any]]:
    return [
        {
            "id": "G1",
            "issuer": "UST",
            "counterparty": "CP_A",
            "market_value": 3_000_000,
            "haircut": 0.01,
            "cost_bps": 6,
            "hqla": True,
        },
        {
            "id": "G2",
            "issuer": "GILT",
            "counterparty": "CP_B",
            "market_value": 2_000_000,
            "haircut": 0.02,
            "cost_bps": 7,
            "hqla": True,
        },
        {
            "id": "C1",
            "issuer": "CORP_A",
            "counterparty": "CP_A",
            "market_value": 1_800_000,
            "haircut": 0.08,
            "cost_bps": 15,
            "hqla": False,
        },
        {
            "id": "C2",
            "issuer": "CORP_A",
            "counterparty": "CP_B",
            "market_value": 1_500_000,
            "haircut": 0.09,
            "cost_bps": 16,
            "hqla": False,
        },
        {
            "id": "C3",
            "issuer": "CORP_B",
            "counterparty": "CP_A",
            "market_value": 1_600_000,
            "haircut": 0.10,
            "cost_bps": 18,
            "hqla": False,
        },
        {
            "id": "C4",
            "issuer": "CORP_B",
            "counterparty": "CP_B",
            "market_value": 1_200_000,
            "haircut": 0.12,
            "cost_bps": 20,
            "hqla": False,
        },
    ]


def make_spec(
    *,
    required: float = 4_000_000,
    min_hqla: float = 0.0,
    concentration: dict[str, float] | None = None,
    max_qubits: int = 8,
    autonomy: str = "L2",
    allow_gate_model: bool = True,
    inventory: list[dict[str, Any]] | None = None,
    seed: int = 7,
) -> ProblemSpec:
    return ProblemSpec.model_validate(
        {
            "problem": "collateral_allocation",
            "objective": {"type": "minimise_collateral_cost"},
            "constraints": {
                "required_collateral": required,
                "minimum_hqla": min_hqla,
                "concentration": concentration or {},
            },
            "execution_policy": {
                "max_effective_qubits": max_qubits,
                "autonomy_level": autonomy,
                "allow_gate_model": allow_gate_model,
                "qaoa_reps": 1,
                "shots": 1024,
                "seed": seed,
            },
            "inventory": default_inventory() if inventory is None else inventory,
        }
    )


def default_routes() -> list[dict[str, Any]]:
    return [
        {
            "id": "R_VISA",
            "cost_bps": 18,
            "fixed_fee": 0.05,
            "approval_rate": 0.94,
            "fraud_bps": 4,
            "latency_ms": 120,
            "capacity": 3,
            "network": "VISA",
        },
        {
            "id": "R_MC",
            "cost_bps": 20,
            "fixed_fee": 0.04,
            "approval_rate": 0.93,
            "fraud_bps": 5,
            "latency_ms": 140,
            "capacity": 3,
            "network": "MC",
        },
        {
            "id": "R_ACQ",
            "cost_bps": 12,
            "fixed_fee": 0.10,
            "approval_rate": 0.88,
            "fraud_bps": 9,
            "latency_ms": 90,
            "capacity": 2,
            "network": "ACQ",
        },
        {
            "id": "R_ALT",
            "cost_bps": 9,
            "fixed_fee": 0.02,
            "approval_rate": 0.82,
            "fraud_bps": 14,
            "latency_ms": 300,
            "capacity": 5,
            "network": "ALT",
        },
    ]


def default_transactions() -> list[dict[str, Any]]:
    return [
        {"id": "T1", "amount": 250_000},
        {"id": "T2", "amount": 180_000},
        {"id": "T3", "amount": 90_000, "eligible_routes": ["R_VISA", "R_MC"]},
        {"id": "T4", "amount": 60_000},
        {"id": "T5", "amount": 30_000},
        {"id": "T6", "amount": 15_000},
    ]


def make_routing_spec(
    *,
    routing: dict[str, Any] | None = None,
    max_qubits: int = 12,
    autonomy: str = "L2",
    allow_gate_model: bool = True,
    routes: list[dict[str, Any]] | None = None,
    transactions: list[dict[str, Any]] | None = None,
    seed: int = 7,
) -> ProblemSpec:
    return ProblemSpec.model_validate(
        {
            "problem": "payment_routing",
            "objective": {"type": "minimise_routing_cost"},
            "routing": routing
            if routing is not None
            else {
                "decline_penalty_bps": 120,
                "network_concentration": 0.6,
                "min_overall_approval": 0.9,
            },
            "execution_policy": {
                "max_effective_qubits": max_qubits,
                "autonomy_level": autonomy,
                "allow_gate_model": allow_gate_model,
                "qaoa_reps": 1,
                "shots": 1024,
                "seed": seed,
            },
            "routes": default_routes() if routes is None else routes,
            "transactions": default_transactions() if transactions is None else transactions,
        }
    )


def default_participants() -> list[dict[str, Any]]:
    return [
        {"id": "BANK_A", "balance": 10},
        {"id": "BANK_B", "balance": 10},
        {"id": "BANK_C", "balance": 10},
        {"id": "BANK_D", "balance": 60},
    ]


def default_obligations() -> list[dict[str, Any]]:
    # A gridlock cycle A->B->C->A (100 each) + a liquidity-funded D->A.
    return [
        {"id": "O_AB", "payer": "BANK_A", "payee": "BANK_B", "amount": 100},
        {"id": "O_BC", "payer": "BANK_B", "payee": "BANK_C", "amount": 100},
        {"id": "O_CA", "payer": "BANK_C", "payee": "BANK_A", "amount": 100},
        {"id": "O_DA", "payer": "BANK_D", "payee": "BANK_A", "amount": 50},
    ]


def make_settlement_spec(
    *,
    max_qubits: int = 12,
    autonomy: str = "L2",
    allow_gate_model: bool = True,
    participants: list[dict[str, Any]] | None = None,
    obligations: list[dict[str, Any]] | None = None,
    settlement: dict[str, Any] | None = None,
    qaoa_reps: int = 1,
    seed: int = 7,
) -> ProblemSpec:
    return ProblemSpec.model_validate(
        {
            "problem": "settlement_netting",
            "objective": {"type": "maximise_settled_value"},
            "settlement": settlement if settlement is not None else {"penalty_scale": 8.0},
            "execution_policy": {
                "max_effective_qubits": max_qubits,
                "autonomy_level": autonomy,
                "allow_gate_model": allow_gate_model,
                "qaoa_reps": qaoa_reps,
                "shots": 1024,
                "seed": seed,
            },
            "participants": default_participants() if participants is None else participants,
            "obligations": default_obligations() if obligations is None else obligations,
        }
    )


def make_fraud_spec(
    *,
    target_metric: str = "auc",
    feature_budget: int = 4,
    allow_gate_model: bool = True,
    synthetic: dict[str, Any] | None = None,
    features: list[list[float]] | None = None,
    labels: list[int] | None = None,
    max_qubits: int = 4,
    autonomy: str = "L2",
    seed: int = 7,
) -> ProblemSpec:
    cfg: dict[str, Any] = {
        "target_metric": target_metric,
        "test_fraction": 0.3,
        "feature_budget": feature_budget,
        "bootstrap": 200,
    }
    if features is None:
        cfg["synthetic"] = synthetic or {
            "n_samples": 160,
            "n_features": 6,
            "n_informative": 3,
            "class_balance": 0.25,
            "separability": 0.9,
        }
    body: dict[str, Any] = {
        "problem": "fraud_detection",
        "objective": {"type": "maximise_detection_performance"},
        "classification": cfg,
        "execution_policy": {
            "max_effective_qubits": max_qubits,
            "autonomy_level": autonomy,
            "allow_gate_model": allow_gate_model,
            "seed": seed,
        },
    }
    if features is not None:
        body["features"] = features
        body["labels"] = labels
    return ProblemSpec.model_validate(body)
