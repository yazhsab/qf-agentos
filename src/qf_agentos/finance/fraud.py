"""Fraud-detection classification domain (quantum kernels).

An honest experimentation harness — NOT a claim that quantum beats classical. All
of the quantum-kernel pipeline (baselines, feature/sample reduction, fidelity
kernel, temporal/leakage/significance verification) lives in the shared
:class:`QuantumKernelClassificationDomain`; this family only supplies the
imbalanced, fraud-like synthetic dataset.
"""

from __future__ import annotations

from ..core.ir import ProblemSpec
from . import qml
from .qkernel import QuantumKernelClassificationDomain


class FraudDetectionDomain(QuantumKernelClassificationDomain):
    problem = "fraud_detection"
    positive_label = "positive"

    def _synthetic_dataset(self, spec: ProblemSpec) -> qml.Dataset:
        cfg = spec.classification
        assert cfg is not None and cfg.synthetic is not None
        s = cfg.synthetic
        return qml.make_synthetic(
            n_samples=s.n_samples,
            n_features=s.n_features,
            n_informative=s.n_informative,
            class_balance=s.class_balance,
            separability=s.separability,
            seed=spec.execution_policy.seed,
        )
