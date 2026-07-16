"""Real-IBM (Qiskit Runtime) backend: channel config, availability, pipeline
routing + L3 gating, and the adapter decode path (with a mocked runtime — no
network, no credentials, no real device)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qf_agentos.backends import quantum_available
from qf_agentos.backends.base import QuboRunConfig, QuboSolution
from qf_agentos.core.config import Settings, reset_settings_cache
from qf_agentos.core.ir import ProblemSpec
from qf_agentos.finance.collateral import build_qubo, qubo_energy, reduce_to_instance
from qf_agentos.pipeline import solve
from qf_test_utils import default_inventory


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


def _ibm_spec(*, qpu_backend: str = "ibm", autonomy: str = "L3") -> ProblemSpec:
    return ProblemSpec.model_validate(
        {
            "problem": "collateral_allocation",
            "constraints": {"required_collateral": 4_000_000},
            "inventory": default_inventory(),
            "execution_policy": {
                "max_effective_qubits": 12,
                "autonomy_level": autonomy,
                "qaoa_reps": 1,
                "shots": 256,
                "qpu_backend": qpu_backend,
                "max_qpu_budget_usd": 5.0,
            },
        }
    )


# --- config + availability ------------------------------------------------


def test_ibm_channel_default_is_the_current_platform():
    # The legacy "ibm_quantum" channel was retired; must default to the new one.
    assert Settings().ibm_channel == "ibm_quantum_platform"


def test_ibm_channel_is_configurable(monkeypatch):
    monkeypatch.setenv("QF_IBM_CHANNEL", "ibm_cloud")
    reset_settings_cache()
    try:
        assert Settings().ibm_channel == "ibm_cloud"
    finally:
        reset_settings_cache()


def test_ibm_unavailable_without_credentials():
    from qf_agentos.backends.registry import get_solver

    available, detail = get_solver("qaoa_ibm").is_available()
    # qiskit-ibm-runtime is installed here, so the missing piece is the token.
    assert not available
    assert "QF_IBM_TOKEN" in detail


# --- pipeline routing + L3 gating -----------------------------------------


class _FakeIbmSolver:
    """Stand-in for the IBM solver so routing/gating are tested without a device."""

    name = "qaoa_ibm"
    kind = "quantum"
    requires_credentials = True

    def is_available(self) -> tuple[bool, str]:
        return True, "fake IBM"

    def solve(self, qubo, config: QuboRunConfig) -> QuboSolution:
        bits = [0] * qubo.n
        return QuboSolution(
            best_bits=bits,
            energy=qubo_energy(qubo, bits),
            metadata={"backend": "fake_ibm_device", "shots": 256, "n_qubits": qubo.n},
            qpu_time_s=1.25,
        )


def _patch_ibm_solver(monkeypatch):
    from qf_agentos.agents import quantum_agent
    from qf_agentos.backends.registry import get_solver as real_get_solver

    def fake(name: str):
        return _FakeIbmSolver() if name == "qaoa_ibm" else real_get_solver(name)

    monkeypatch.setattr(quantum_agent, "get_solver", fake)


@pytest.mark.skipif(not quantum_available(), reason="qiskit not installed")
@pytest.mark.slow
def test_ibm_route_requires_l3_approval(monkeypatch):
    monkeypatch.setenv("QF_IBM_TOKEN", "dummy-token")
    reset_settings_cache()
    _patch_ibm_solver(monkeypatch)
    try:
        # Planner routes to IBM (available), but without approval the paid QPU is denied.
        ctx = solve(_ibm_spec(), human_approved=False)
        assert ctx.state.hardware_plan.target == "gate_model_ibm_runtime"
        assert ctx.state.instance_qaoa is None  # RUN_PAID_QPU not authorised
        assert any("QAOA not executed" in w for w in ctx.warnings)
    finally:
        reset_settings_cache()


@pytest.mark.skipif(not quantum_available(), reason="qiskit not installed")
@pytest.mark.slow
def test_ibm_route_runs_with_l3_approval(monkeypatch):
    monkeypatch.setenv("QF_IBM_TOKEN", "dummy-token")
    reset_settings_cache()
    _patch_ibm_solver(monkeypatch)
    try:
        ctx = solve(_ibm_spec(), human_approved=True)
        assert ctx.state.hardware_plan.target == "gate_model_ibm_runtime"
        assert ctx.state.instance_qaoa is not None
        assert ctx.state.instance_qaoa.backend == "fake_ibm_device"
        assert ctx.state.instance_qaoa.qpu_time_s == pytest.approx(1.25)
    finally:
        reset_settings_cache()


@pytest.mark.skipif(not quantum_available(), reason="qiskit not installed")
@pytest.mark.slow
def test_ibm_falls_back_to_sim_when_unavailable():
    # qpu_backend='ibm' but no credentials -> run on the simulator, with a reason.
    reset_settings_cache()
    ctx = solve(_ibm_spec(autonomy="L2"), human_approved=False)
    plan = ctx.state.hardware_plan
    assert plan.target == "gate_model_statevector_sim"
    assert any("qpu_backend='ibm'" in r and "unavailable" in r for r in plan.reasons)


# --- adapter decode path (mocked Qiskit Runtime) --------------------------


@pytest.mark.skipif(not quantum_available(), reason="qiskit not installed")
@pytest.mark.slow
def test_adapter_decodes_counts_with_mocked_runtime(monkeypatch, qubo):
    import qiskit
    import qiskit_ibm_runtime as qir

    n = qubo.n
    counts = {"0" * n: 700, "1" + "0" * (n - 1): 300}
    captured: dict[str, object] = {}

    class FakeService:
        def __init__(self, **kwargs):
            captured.update(kwargs)  # channel/token/instance actually passed

        def backend(self, _name):
            return SimpleNamespace(name="fake")

        def least_busy(self, **_kw):
            return SimpleNamespace(name="fake")

    class FakeJob:
        def result(self):
            item = SimpleNamespace(
                data=SimpleNamespace(meas=SimpleNamespace(get_counts=lambda: counts)),
                metadata={},
            )
            return [item]

    sampler_modes: list[object] = []

    class FakeSampler:
        def __init__(self, mode=None):
            sampler_modes.append(mode)  # job mode passes the backend directly

        def run(self, _circuits, shots=None):
            return FakeJob()

    monkeypatch.setattr(qir, "QiskitRuntimeService", FakeService)
    monkeypatch.setattr(qir, "SamplerV2", FakeSampler)
    monkeypatch.setattr(
        qiskit, "generate_preset_pass_manager", lambda **_kw: SimpleNamespace(run=lambda c: c)
    )
    monkeypatch.setenv("QF_IBM_TOKEN", "dummy-token")
    monkeypatch.setenv("QF_IBM_CHANNEL", "ibm_cloud")
    reset_settings_cache()
    try:
        from qf_agentos.backends.ibm_runtime import IbmRuntimeQaoaSolver

        sol = IbmRuntimeQaoaSolver().solve(qubo, QuboRunConfig(seed=7, reps=1, shots=256))
        assert len(sol.best_bits) == n
        assert sol.metadata["backend"] == "fake"
        assert sol.metadata["shots"] == 1000  # 700 + 300
        # The configured channel + token actually reach the runtime service...
        assert captured["channel"] == "ibm_cloud"
        assert captured["token"] == "dummy-token"
        # ...but the token must NEVER appear in the solution metadata (no leak).
        assert "dummy-token" not in str(sol.metadata)
        # Job execution mode: the sampler is given the backend directly (NOT a
        # Session) so it works on the free Open plan (which forbids sessions).
        assert sampler_modes and sampler_modes[0].name == "fake"
    finally:
        reset_settings_cache()
