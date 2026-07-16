"""D-Wave Leap annealer backend: availability, pipeline routing + L3 gating, and
the adapter decode path (mocked sampler; skipped unless the dwave extra is present)."""

from __future__ import annotations

import pytest

from qf_agentos.backends.base import QuboRunConfig, QuboSolution
from qf_agentos.core.config import reset_settings_cache
from qf_agentos.core.ir import ProblemSpec
from qf_agentos.finance.collateral import build_qubo, qubo_energy, reduce_to_instance
from qf_agentos.pipeline import solve
from qf_test_utils import default_inventory


def _dwave_available() -> bool:
    try:
        import dimod  # noqa: F401
        import dwave.system  # noqa: F401
    except Exception:
        return False
    return True


@pytest.fixture
def qubo():
    spec = ProblemSpec.model_validate(
        {
            "problem": "collateral_allocation",
            "constraints": {"required_collateral": 4_000_000},
            "inventory": default_inventory(),
        }
    )
    return build_qubo(reduce_to_instance(spec, 9, slack_bits=4), slack_bits=4)


def _dwave_spec(*, qpu_backend: str = "dwave", autonomy: str = "L3") -> ProblemSpec:
    return ProblemSpec.model_validate(
        {
            "problem": "collateral_allocation",
            "constraints": {"required_collateral": 4_000_000},
            "inventory": default_inventory(),
            "execution_policy": {
                "max_effective_qubits": 12,
                "autonomy_level": autonomy,
                "qpu_backend": qpu_backend,
                "max_qpu_budget_usd": 5.0,
            },
        }
    )


# --- availability ---------------------------------------------------------


def test_dwave_unavailable_without_sdk_or_creds():
    from qf_agentos.backends.registry import get_solver

    available, detail = get_solver("dwave_hybrid").is_available()
    assert not available
    assert "dwave" in detail.lower() or "QF_DWAVE_TOKEN" in detail


# --- pipeline routing + L3 gating (mocked; no SDK needed) ------------------


class _FakeDwaveSolver:
    name = "dwave_hybrid"
    kind = "quantum"
    requires_credentials = True

    def is_available(self) -> tuple[bool, str]:
        return True, "fake D-Wave"

    def solve(self, qubo, config: QuboRunConfig) -> QuboSolution:
        bits = [0] * qubo.n
        counts = {"0" * qubo.n: 90, "1" + "0" * (qubo.n - 1): 10}
        return QuboSolution(
            best_bits=bits,
            energy=qubo_energy(qubo, bits),
            metadata={
                "backend": "fake_advantage_system",
                "shots": 100,
                "counts": counts,
                "sample_mean_energy": 0.0,
            },
            qpu_time_s=0.5,
        )


def _patch_dwave(monkeypatch):
    # Report the capability as available (real is_available needs the SDK)...
    monkeypatch.setattr(
        "qf_agentos.backends.dwave.DwaveHybridSolver.is_available",
        lambda self: (True, "fake D-Wave"),
    )
    # ...and hand the executor the fake solver instead of the real one.
    from qf_agentos.agents import quantum_agent
    from qf_agentos.backends.registry import get_solver as real_get_solver

    monkeypatch.setattr(
        quantum_agent,
        "get_solver",
        lambda name: _FakeDwaveSolver() if name == "dwave_hybrid" else real_get_solver(name),
    )


def test_dwave_route_requires_l3_approval(monkeypatch):
    _patch_dwave(monkeypatch)
    reset_settings_cache()
    ctx = solve(_dwave_spec(), human_approved=False)
    assert ctx.state.hardware_plan.target == "annealer_dwave_hybrid"
    assert ctx.state.instance_qaoa is None  # RUN_PAID_QPU not authorised
    assert any("QAOA not executed" in w for w in ctx.warnings)


def test_dwave_route_runs_with_l3_approval(monkeypatch):
    _patch_dwave(monkeypatch)
    reset_settings_cache()
    ctx = solve(_dwave_spec(), human_approved=True)
    assert ctx.state.hardware_plan.target == "annealer_dwave_hybrid"
    assert ctx.state.instance_qaoa is not None
    assert ctx.state.instance_qaoa.backend == "fake_advantage_system"
    assert not ctx.state.errors  # verification runs on the annealer histogram
    qrep = ctx.state.verification.get("dwave_hybrid")
    assert qrep is not None and qrep.quantum_contribution is not None


def test_dwave_falls_back_to_sim_when_unavailable():
    reset_settings_cache()
    ctx = solve(_dwave_spec(autonomy="L2"), human_approved=False)
    plan = ctx.state.hardware_plan
    assert plan.target == "gate_model_statevector_sim"
    assert any("qpu_backend='dwave'" in r and "unavailable" in r for r in plan.reasons)


# --- adapter decode path (needs the dwave extra; skipped otherwise) --------


@pytest.mark.skipif(not _dwave_available(), reason="dwave-ocean-sdk not installed")
def test_dwave_adapter_decodes_sampleset(monkeypatch, qubo):
    from types import SimpleNamespace

    import dwave.system as dws

    from qf_agentos.backends.dwave import DwaveHybridSolver

    n = qubo.n
    zero = dict.fromkeys(range(n), 0)
    one = {i: (1 if i == 0 else 0) for i in range(n)}

    class FakeSampleset:
        info = {"qpu_access_time": 500000, "charge_time": 1, "run_time": 2}
        first = SimpleNamespace(sample=zero)

        def data(self, fields=None):
            yield SimpleNamespace(sample=zero, num_occurrences=90)
            yield SimpleNamespace(sample=one, num_occurrences=10)

    class FakeSampler:
        solver = SimpleNamespace(name="fake_advantage_system")

        def __init__(self, token=None):
            pass

        def sample(self, bqm, label=None):
            return FakeSampleset()

    monkeypatch.setattr(dws, "LeapHybridSampler", FakeSampler)
    monkeypatch.setattr(
        "qf_agentos.backends.dwave.DwaveHybridSolver.is_available", lambda self: (True, "fake")
    )
    monkeypatch.setenv("QF_DWAVE_TOKEN", "dummy-token")
    reset_settings_cache()
    try:
        sol = DwaveHybridSolver().solve(qubo, QuboRunConfig(seed=7))
        assert len(sol.best_bits) == n
        assert sol.metadata["backend"] == "fake_advantage_system"
        assert sol.metadata["shots"] == 100
        assert sol.metadata["counts"]
        assert sol.qpu_time_s == pytest.approx(0.5)
        assert "dummy-token" not in str(sol.metadata)
    finally:
        reset_settings_cache()
