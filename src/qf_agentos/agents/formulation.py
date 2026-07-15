"""Agent 2 — Formulation.

Enumerates the mathematically-distinct ways to express the problem and states,
for each, what it can represent and what it cannot. The concrete lossy encoding
(the QUBO's dropped constraints) is produced later by the Hardware Planner once
the qubit budget fixes the instance size; this agent records the catalogue and
the mapping rationale.
"""

from __future__ import annotations

from ..core.artifacts import Formulation, FormulationCatalogue
from ..core.workflow import RunContext


def formulation_agent(ctx: RunContext) -> str:
    n = len(ctx.spec.eligible_inventory)

    catalogue = [
        Formulation(
            name="continuous_lp",
            kind="Linear Program",
            variables=f"{n} continuous x_i in [0,1] (fraction posted)",
            represents="cost, coverage, HQLA floor, concentration — all exactly",
            note="Theoretical best; a lower bound on achievable cost.",
        ),
        Formulation(
            name="binary_milp",
            kind="Mixed-Integer Linear Program",
            variables=f"{n} binary x_i (post whole lot or nothing)",
            represents="cost, coverage, HQLA floor, concentration — all exactly",
            note="The FAIR classical comparator for any QUBO/QAOA result.",
        ),
        Formulation(
            name="qubo",
            kind="Quadratic Unconstrained Binary Optimisation",
            variables="binary x_i + slack bits on a reduced research instance",
            represents="cost + coverage (inequality via slack bits)",
            note="Concentration and HQLA cannot be expressed unconstrained; they "
            "become verification-only constraints.",
        ),
        Formulation(
            name="ising",
            kind="Ising Hamiltonian",
            variables="spins z_i in {+1,-1}, x_i = (1 - z_i)/2",
            represents="same content as the QUBO",
            note="Direct input to gate-model QAOA and to annealers.",
        ),
    ]

    ctx.state.formulations = FormulationCatalogue(
        catalogue=catalogue,
        selected_classical="binary_milp",
        selected_quantum_path="qubo -> ising -> QAOA",
        encoding_loss_note=(
            "The quantum path is a relaxation of a reduced instance. Any decoded "
            "quantum solution is re-checked against the full instance constraints."
        ),
    )
    return (
        f"Enumerated {len(catalogue)} formulations; classical=binary_milp, quantum=qubo→ising→QAOA."
    )
