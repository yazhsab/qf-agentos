"""Agent 2 — Formulation.

Delegates to the domain to enumerate the mathematically-distinct ways to express
the problem and what each can and cannot represent. The concrete lossy encoding
(the QUBO's dropped constraints) is produced later by the Hardware Planner.
"""

from __future__ import annotations

from ..core.workflow import RunContext
from ..finance import get_domain


def formulation_agent(ctx: RunContext) -> str:
    domain = get_domain(ctx.spec.problem)
    catalogue = domain.formulations(ctx.spec)
    ctx.state.formulations = catalogue
    return (
        f"Enumerated {len(catalogue.catalogue)} formulations; "
        f"classical={catalogue.selected_classical}, quantum={catalogue.selected_quantum_path}."
    )
