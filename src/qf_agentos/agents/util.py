"""Shared helpers for agents: turn a raw solution into a verified SolveResult."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.result import Allocation, SolveResult
from ..finance.collateral import (
    ResearchInstance,
    bits_to_allocation,
    check_constraints,
)


def evaluate_instance_bits(
    method: str,
    kind: str,
    backend: str,
    instance: ResearchInstance,
    bits: np.ndarray | list[int],
    *,
    runtime_s: float = 0.0,
    qpu_time_s: float = 0.0,
    cost_usd: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> SolveResult:
    """Decode a bitstring on the research instance and check the FULL instance
    constraint set (including constraints the QUBO never encoded)."""
    ids = [s.id for s in instance.securities]
    alloc = bits_to_allocation(ids, bits)
    feasible, obj, _checks = check_constraints(
        instance.securities,
        alloc,
        instance.required_collateral,
        instance.minimum_hqla,
        instance.concentration,
    )
    return SolveResult(
        method=method,
        kind=kind,
        backend=backend,
        scope="research_instance",
        feasible=feasible,
        objective=obj,
        allocation=alloc,
        runtime_s=runtime_s,
        qpu_time_s=qpu_time_s,
        cost_usd=cost_usd,
        metadata=metadata or {},
    )


def evaluate_instance_alloc(
    method: str,
    kind: str,
    backend: str,
    instance: ResearchInstance,
    alloc: Allocation,
    *,
    runtime_s: float = 0.0,
) -> SolveResult:
    feasible, obj, _ = check_constraints(
        instance.securities,
        alloc,
        instance.required_collateral,
        instance.minimum_hqla,
        instance.concentration,
    )
    return SolveResult(
        method=method,
        kind=kind,
        backend=backend,
        scope="research_instance",
        feasible=feasible,
        objective=obj,
        allocation=alloc,
        runtime_s=runtime_s,
    )
