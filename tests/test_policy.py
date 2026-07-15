"""Policy engine — the safety-critical authorization branches."""

from __future__ import annotations

import pytest

from qf_agentos.core.ir import AutonomyLevel, ExecutionPolicy
from qf_agentos.core.policy import Action, PolicyEngine


def _policy(level: str, *, budget: float = 0.0, approved: bool = False) -> PolicyEngine:
    pol = ExecutionPolicy(autonomy_level=AutonomyLevel(level), max_qpu_budget_usd=budget)
    return PolicyEngine(pol, human_approved=approved)


def test_l0_allows_explain_only():
    eng = _policy("L0")
    assert eng.authorize(Action.EXPLAIN).allowed
    assert not eng.authorize(Action.PLAN).allowed
    assert not eng.authorize(Action.RUN_SIMULATOR).allowed


def test_l2_allows_simulator_not_paid_qpu():
    eng = _policy("L2")
    assert eng.authorize(Action.RUN_SIMULATOR).allowed
    auth = eng.authorize(Action.RUN_PAID_QPU, cost_usd=0.0)
    assert not auth.allowed  # below L3


def test_paid_qpu_requires_approval_even_at_l3():
    eng = _policy("L3", budget=100.0, approved=False)
    auth = eng.authorize(Action.RUN_PAID_QPU, cost_usd=10.0)
    assert not auth.allowed and auth.needs_human_approval


def test_paid_qpu_allowed_with_approval_and_budget():
    eng = _policy("L3", budget=100.0, approved=True)
    auth = eng.authorize(Action.RUN_PAID_QPU, cost_usd=10.0)
    assert auth.allowed


def test_paid_qpu_blocked_over_budget():
    eng = _policy("L3", budget=5.0, approved=True)
    auth = eng.authorize(Action.RUN_PAID_QPU, cost_usd=50.0)
    assert not auth.allowed
    assert "budget" in auth.reason.lower()


def test_recommend_production_requires_l4_and_approval():
    assert not _policy("L2").authorize(Action.RECOMMEND_PRODUCTION).allowed
    assert not _policy("L4", approved=False).authorize(Action.RECOMMEND_PRODUCTION).allowed
    assert _policy("L4", approved=True).authorize(Action.RECOMMEND_PRODUCTION).allowed


@pytest.mark.parametrize("level,rank", [("L0", 0), ("L2", 2), ("L4", 4)])
def test_autonomy_rank(level, rank):
    assert AutonomyLevel(level).rank == rank
