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
