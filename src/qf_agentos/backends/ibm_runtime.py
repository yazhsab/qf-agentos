"""IBM Quantum (Qiskit Runtime) QAOA backend — real, credential-gated.

Cost-aware pattern: the variational parameters are optimised on the local
statevector simulator (cheap), and only the *final* optimised circuit is sampled
on real hardware. This backend is inert without ``qiskit-ibm-runtime`` and IBM
credentials, and is gated behind autonomy L3 + human approval by the policy
engine. It is not exercised by the default pipeline.
"""

from __future__ import annotations

import numpy as np

from ..core.config import get_settings
from ..core.errors import BackendError, BackendUnavailableError
from ..finance.collateral import Qubo, qubo_energy
from .base import QuboRunConfig, QuboSolution


class IbmRuntimeQaoaSolver:
    name = "qaoa_ibm"
    kind = "quantum"
    requires_credentials = True

    def is_available(self) -> tuple[bool, str]:
        try:
            import qiskit_ibm_runtime  # noqa: F401
        except Exception:
            return False, "install qf-agentos[ibm] (qiskit-ibm-runtime)"
        if not get_settings().has_ibm_credentials():
            return False, "set QF_IBM_TOKEN (and optionally QF_IBM_INSTANCE/QF_IBM_BACKEND)"
        return True, "IBM Quantum via Qiskit Runtime"

    def solve(self, qubo: Qubo, config: QuboRunConfig) -> QuboSolution:
        available, detail = self.is_available()
        if not available:
            raise BackendUnavailableError(self.name, detail)

        from qiskit import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session

        from .quantum import optimize_qaoa

        settings = get_settings()
        token = settings.ibm_token.get_secret_value() if settings.ibm_token else None

        try:
            # 1. Optimise parameters cheaply on the local simulator.
            opt = optimize_qaoa(qubo, reps=config.reps, seed=config.seed)
            isa_sim = opt["isa"]
            circuit = isa_sim.assign_parameters(opt["best_params"]).copy()
            circuit.measure_all()

            # 2. Sample ONLY the final circuit on real hardware. The channel is
            #    configurable (the legacy "ibm_quantum" channel was retired; the
            #    current IBM Quantum Platform default is "ibm_quantum_platform").
            service = QiskitRuntimeService(
                channel=settings.ibm_channel, token=token, instance=settings.ibm_instance
            )
            backend = (
                service.backend(settings.ibm_backend)
                if settings.ibm_backend
                else service.least_busy(operational=True, simulator=False)
            )
            pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
            isa_circuit = pm.run(circuit)

            with Session(backend=backend) as session:
                sampler = SamplerV2(mode=session)
                job = sampler.run([isa_circuit], shots=config.shots)
                result = job.result()
            counts = result[0].data.meas.get_counts()
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"IBM Runtime execution failed: {exc}") from exc

        n = qubo.n
        best_key = min(counts, key=lambda k: qubo_energy(qubo, _bits(k, n)))
        best_bits = _bits(best_key, n)
        qpu_time = _extract_qpu_seconds(result)
        return QuboSolution(
            best_bits=[int(b) for b in best_bits],
            energy=qubo_energy(qubo, best_bits),
            metadata={
                "backend": getattr(backend, "name", "ibm"),
                "shots": int(sum(counts.values())),
            },
            qpu_time_s=qpu_time,
        )


def _bits(key: str, n: int) -> np.ndarray:
    s = key.replace(" ", "")
    return np.array([int(s[n - 1 - i]) for i in range(n)], dtype=int)


def _extract_qpu_seconds(result: object) -> float:
    try:
        meta = result[0].metadata  # type: ignore[index]
        return float(meta.get("execution", {}).get("execution_spans", 0.0) or 0.0)
    except Exception:
        return 0.0


__all__ = ["IbmRuntimeQaoaSolver"]
