"""Agent 3 — Classical Baseline.

Runs *serious* classical optimisation on the full problem: a continuous LP
relaxation (lower bound) and a binary MILP (the fair comparator). Both use the
HiGHS solver shipped inside SciPy. The system never compares a quantum method
against a deliberately weak baseline.
"""

from __future__ import annotations

import time

from ..core.result import SolveResult
from ..core.workflow import RunContext
from ..finance.collateral import solve_binary_milp, solve_lp_relaxation


def classical_baseline_agent(ctx: RunContext) -> str:
    spec = ctx.spec

    t0 = time.perf_counter()
    lp = solve_lp_relaxation(spec)
    lp_dt = time.perf_counter() - t0

    t0 = time.perf_counter()
    milp = solve_binary_milp(spec)
    milp_dt = time.perf_counter() - t0

    n_posted = len(milp.allocation.posted()) if milp.allocation is not None else 0

    ctx.state.classical_lp = SolveResult(
        method="classical_lp_relaxation",
        kind="classical",
        backend="scipy/HiGHS",
        scope="full_problem",
        feasible=lp.feasible,
        objective=lp.objective,
        allocation=lp.allocation,
        runtime_s=lp_dt,
        metadata={"role": "lower bound on cost", "status": lp.status},
    )
    ctx.state.classical_milp = SolveResult(
        method="classical_milp",
        kind="classical",
        backend="scipy/HiGHS",
        scope="full_problem",
        feasible=milp.feasible,
        objective=milp.objective,
        allocation=milp.allocation,
        runtime_s=milp_dt,
        metadata={
            "role": "fair binary comparator / production recommendation",
            "status": milp.status,
            "n_posted": n_posted,
        },
    )

    if milp.feasible and lp.feasible and lp.objective is not None and milp.objective is not None:
        ctx.state.integrality_gap = milp.objective - lp.objective

    if milp.feasible and milp.objective is not None:
        lp_txt = f"{lp.objective:,.2f}" if lp.objective is not None else "n/a"
        return (
            f"MILP optimal: cost {milp.objective:,.2f} in {milp_dt * 1000:.0f} ms "
            f"(LP lower bound {lp_txt}); posting {n_posted} of {len(spec.eligible_inventory)} securities."
        )
    return f"MILP infeasible ({milp.status}); LP feasible={lp.feasible}."
