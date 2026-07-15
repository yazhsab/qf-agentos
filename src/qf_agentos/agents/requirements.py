"""Agent 1 — Financial Requirements.

Turns the validated Finance IR into an execution-ready understanding: it surfaces
missing constraints, records assumptions, and runs a cheap feasibility sanity
check before any solver spends effort. In a hosted deployment this is where an
LLM would elicit clarifications; here the logic is deterministic and auditable.
"""

from __future__ import annotations

from ..core.artifacts import RequirementsReport
from ..core.observability import get_logger
from ..core.workflow import RunContext

_logger = get_logger("agents.requirements")


def requirements_agent(ctx: RunContext) -> str:
    spec = ctx.spec
    cn = spec.constraints
    assumptions: list[str] = []
    discovered_gaps: list[str] = []

    if not spec.inventory:
        ctx.warn("Inventory is empty — no securities available to post.")
    if not cn.concentration:
        discovered_gaps.append(
            "No concentration limits given — a single issuer could dominate the pool."
        )
    if cn.minimum_hqla == 0:
        discovered_gaps.append("No minimum HQLA floor given — pool liquidity is unconstrained.")

    n_ineligible = len(spec.inventory) - len(spec.eligible_inventory)
    if n_ineligible:
        assumptions.append(
            f"{n_ineligible} securities marked ineligible are excluded from posting."
        )

    available = spec.total_available_coverage
    required = cn.required_collateral
    feasible_upper_bound = available >= required
    if not feasible_upper_bound:
        ctx.warn(
            f"Inventory post-haircut coverage {available:,.0f} < required {required:,.0f}: "
            "the problem is infeasible as stated."
        )
    for gap in discovered_gaps:
        ctx.warn(gap)

    ctx.state.requirements = RequirementsReport(
        n_securities=len(spec.inventory),
        n_eligible=len(spec.eligible_inventory),
        required_collateral=required,
        available_coverage=available,
        coverage_headroom=available - required,
        trivially_feasible_upper_bound=feasible_upper_bound,
        concentration_attrs=list(cn.concentration.keys()),
        minimum_hqla=cn.minimum_hqla,
        discovered_gaps=discovered_gaps,
        assumptions=assumptions,
        autonomy_level=spec.execution_policy.autonomy_level.value,
    )
    _logger.debug(
        "requirements: eligible=%d headroom=%.0f",
        len(spec.eligible_inventory),
        available - required,
    )

    return (
        f"Understood {spec.objective.type.value}: {len(spec.eligible_inventory)} eligible securities, "
        f"required collateral {required:,.0f}, headroom {available - required:,.0f}; "
        f"{len(discovered_gaps)} constraint gap(s) flagged."
    )
