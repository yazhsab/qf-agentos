"""Finance domain models, problem builders, and the domain registry."""

from __future__ import annotations

from ..core.domain import DomainBase
from ..core.errors import SpecError

# Problem families known to the platform. Each maps to a DomainBase.
KNOWN_PROBLEMS: tuple[str, ...] = (
    "collateral_allocation",
    "payment_routing",
    "settlement_netting",
    "fraud_detection",
    "rfq_fill",
)


def get_domain(problem: str) -> DomainBase:
    """Return the domain for a problem family (lazy import)."""
    if problem == "collateral_allocation":
        from .collateral import CollateralDomain

        return CollateralDomain()
    if problem == "payment_routing":
        from .payment_routing import PaymentRoutingDomain

        return PaymentRoutingDomain()
    if problem == "settlement_netting":
        from .settlement import SettlementDomain

        return SettlementDomain()
    if problem == "fraud_detection":
        from .fraud import FraudDetectionDomain

        return FraudDetectionDomain()
    if problem == "rfq_fill":
        from .rfq import RFQFillDomain

        return RFQFillDomain()
    raise SpecError(f"Unknown problem '{problem}'. Known problems: {', '.join(KNOWN_PROBLEMS)}.")


__all__ = ["KNOWN_PROBLEMS", "get_domain"]
