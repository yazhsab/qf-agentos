"""D-Wave Leap hybrid backend — real, credential-gated.

Solves the QUBO on D-Wave's Leap hybrid sampler (which handles binary quadratic
models of practical size). Inert without ``dwave-ocean-sdk`` and a Leap token,
and gated behind autonomy L3 + approval. Not exercised by the default pipeline.
"""

from __future__ import annotations

from ..core.config import get_settings
from ..core.errors import BackendError, BackendUnavailableError
from ..finance.collateral import Qubo, qubo_energy
from .base import QuboRunConfig, QuboSolution


class DwaveHybridSolver:
    name = "dwave_hybrid"
    kind = "quantum"
    requires_credentials = True

    def is_available(self) -> tuple[bool, str]:
        try:
            import dimod  # noqa: F401
            import dwave.system  # noqa: F401
        except Exception:
            return False, "install qf-agentos[dwave] (dwave-ocean-sdk)"
        if not get_settings().has_dwave_credentials():
            return False, "set QF_DWAVE_TOKEN"
        return True, "D-Wave Leap hybrid sampler"

    def solve(self, qubo: Qubo, config: QuboRunConfig) -> QuboSolution:
        available, detail = self.is_available()
        if not available:
            raise BackendUnavailableError(self.name, detail)

        import dimod
        from dwave.system import LeapHybridSampler

        settings = get_settings()
        token = settings.dwave_token.get_secret_value() if settings.dwave_token else None

        try:
            bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
            for (i, j), c in qubo.Q.items():
                if i == j:
                    bqm.add_linear(i, c)
                else:
                    bqm.add_quadratic(i, j, c)
            bqm.offset = qubo.offset

            sampler = LeapHybridSampler(token=token)
            sampleset = sampler.sample(bqm, label="qf-agentos collateral QUBO")
            best = sampleset.first
        except Exception as exc:
            raise BackendError(f"D-Wave execution failed: {exc}") from exc

        bits = [int(best.sample[i]) for i in range(qubo.n)]
        qpu_time = float(sampleset.info.get("qpu_access_time", 0.0)) / 1e6  # us -> s
        return QuboSolution(
            best_bits=bits,
            energy=qubo_energy(qubo, bits),
            metadata={
                "charge_time_s": sampleset.info.get("charge_time"),
                "run_time_s": sampleset.info.get("run_time"),
            },
            qpu_time_s=qpu_time,
        )


__all__ = ["DwaveHybridSolver"]
