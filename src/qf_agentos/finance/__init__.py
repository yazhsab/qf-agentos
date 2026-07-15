"""Finance domain models, problem builders, and the domain registry."""

from __future__ import annotations

from ..core.domain import ProblemDomain
from ..core.errors import SpecError

# Problem families known to the platform. Each maps to a ProblemDomain.
KNOWN_PROBLEMS: tuple[str, ...] = ("collateral_allocation", "payment_routing")


def get_domain(problem: str) -> ProblemDomain:
    """Return the :class:`ProblemDomain` for a problem family (lazy import)."""
    if problem == "collateral_allocation":
        from .collateral import CollateralDomain

        return CollateralDomain()
    if problem == "payment_routing":
        from .payment_routing import PaymentRoutingDomain

        return PaymentRoutingDomain()
    raise SpecError(f"Unknown problem '{problem}'. Known problems: {', '.join(KNOWN_PROBLEMS)}.")


__all__ = ["KNOWN_PROBLEMS", "get_domain"]
