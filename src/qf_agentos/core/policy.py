"""Policy engine — enforces autonomy levels and budget gates.

The engine turns the abstract control model (L0..L4) into concrete, auditable
authorisation decisions. It is deliberately conservative: paid or irreversible
actions require both a sufficient autonomy level *and* explicit human approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .ir import AutonomyLevel, ExecutionPolicy


class Action(str, Enum):
    EXPLAIN = "explain"
    PLAN = "plan"
    RUN_SIMULATOR = "run_simulator"
    RUN_PAID_QPU = "run_paid_qpu"
    RECOMMEND_PRODUCTION = "recommend_production"


# Minimum autonomy level required to even consider each action.
_MIN_LEVEL: dict[Action, AutonomyLevel] = {
    Action.EXPLAIN: AutonomyLevel.L0,
    Action.PLAN: AutonomyLevel.L1,
    Action.RUN_SIMULATOR: AutonomyLevel.L2,
    Action.RUN_PAID_QPU: AutonomyLevel.L3,
    Action.RECOMMEND_PRODUCTION: AutonomyLevel.L4,
}


@dataclass(frozen=True)
class Authorization:
    action: Action
    allowed: bool
    needs_human_approval: bool
    reason: str


class PolicyEngine:
    """Authorises agent actions against the ExecutionPolicy."""

    def __init__(self, policy: ExecutionPolicy, human_approved: bool = False):
        self.policy = policy
        self.human_approved = human_approved

    def authorize(self, action: Action, *, cost_usd: float = 0.0) -> Authorization:
        level = self.policy.autonomy_level
        required = _MIN_LEVEL[action]

        if level.rank < required.rank:
            return Authorization(
                action,
                False,
                False,
                f"Autonomy {level.value} is below the {required.value} required for {action.value}.",
            )

        if action is Action.RUN_PAID_QPU:
            if cost_usd > self.policy.max_qpu_budget_usd:
                return Authorization(
                    action,
                    False,
                    True,
                    f"Estimated cost ${cost_usd:.2f} exceeds max_qpu_budget_usd "
                    f"${self.policy.max_qpu_budget_usd:.2f}.",
                )
            if not self.human_approved:
                return Authorization(
                    action,
                    False,
                    True,
                    "Paid QPU execution requires explicit human approval (pass --yes / approve).",
                )

        if action is Action.RECOMMEND_PRODUCTION and not self.human_approved:
            return Authorization(
                action,
                False,
                True,
                "Production recommendations require mandatory human sign-off.",
            )

        return Authorization(action, True, action is Action.RUN_PAID_QPU, "authorised")
