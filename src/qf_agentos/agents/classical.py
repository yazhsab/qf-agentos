"""Agent 3 — Classical Baseline.

Delegates to the domain, which runs *serious* classical optimisation on the full
problem (an LP relaxation lower bound + an exact binary MILP). The system never
compares a quantum method against a deliberately weak baseline.
"""

from __future__ import annotations

from ..core.domain import ProblemDomain
from ..core.workflow import RunContext
from ..finance import get_domain


def classical_baseline_agent(ctx: RunContext) -> str:
    domain = get_domain(ctx.spec.problem)
    assert isinstance(domain, ProblemDomain)
    baseline = domain.solve_classical_full(ctx.spec)
    ctx.state.classical_lp = baseline.lp
    ctx.state.classical_milp = baseline.milp
    ctx.state.integrality_gap = baseline.integrality_gap

    milp = baseline.milp
    if milp.feasible and milp.objective is not None:
        lp_txt = (
            f"{baseline.lp.objective:,.2f}"
            if baseline.lp is not None and baseline.lp.objective is not None
            else "n/a"
        )
        n_posted = milp.metadata.get("n_posted", "?")
        return (
            f"MILP optimal: cost {milp.objective:,.2f} in {milp.runtime_s * 1000:.0f} ms "
            f"(LP lower bound {lp_txt}); {n_posted} decision(s) selected."
        )
    return f"MILP infeasible ({milp.metadata.get('status')})."
