"""RFQ fill-prediction classification domain (quantum kernels).

A market maker receives a request-for-quote (RFQ) and quotes a price; the client
either *fills* (trades) or walks. Predicting fill probability from features known
at quote time — quoted spread, order size, volatility, response latency,
counterparty tier — drives pricing and hedging. This is a binary classification
problem, so it reuses the shared :class:`QuantumKernelClassificationDomain`
pipeline verbatim; only the (synthetic) dataset and a little wording are
RFQ-specific.

As everywhere in QF-AgentOS, this is an honest experimentation harness: strong
classical baselines vs a quantum fidelity kernel on the SAME learner and temporal
holdout, with a bootstrap significance test — NOT a claim that quantum wins.
"""

from __future__ import annotations

import numpy as np

from ..core.ir import ProblemSpec
from . import qml
from .qkernel import QuantumKernelClassificationDomain

# Illustrative names for the informative RFQ drivers (the synthetic features are
# standardised signals; the names describe what each driver stands in for).
_RFQ_DRIVERS = [
    "quoted_spread_bps",
    "order_size_ratio",
    "volatility",
    "response_latency_ms",
    "counterparty_tier",
    "time_of_day",
]


def _rfq_feature_names(d: int) -> list[str]:
    names = _RFQ_DRIVERS[:d]
    names += [f"signal_{j}" for j in range(len(names), d)]
    return names


def make_rfq_synthetic(
    *,
    n_samples: int = 240,
    n_features: int = 6,
    n_informative: int = 3,
    class_balance: float = 0.35,
    separability: float = 0.9,
    seed: int = 7,
) -> qml.Dataset:
    """Deterministic RFQ fill dataset with a NON-LINEAR decision boundary.

    Fill is driven by a linear pull from the informative signals (alternating
    signs: tighter spread and better tier raise fill; size, volatility, latency
    lower it) PLUS a spread-x-size interaction term — a nonlinearity a ZZ
    feature-map kernel can, in principle, exploit (whether it *helps* is exactly
    what the pipeline tests honestly). The fill rate is pinned to
    ``class_balance`` by thresholding the top scores, so the label balance is
    exact and controllable. Seeded and reproducible.
    """
    rng = np.random.default_rng(seed)
    n, d = n_samples, n_features
    X = rng.standard_normal((n, d))
    info = min(max(2, n_informative), d)  # need >= 2 signals for the interaction
    signs = np.array([(-1.0) ** j for j in range(info)])  # alternating pull on fill
    linear = X[:, :info] @ signs
    interaction = X[:, 0] * X[:, 1]  # spread-x-size nonlinearity
    score = separability * (linear + 0.6 * interaction) + rng.standard_normal(n) * 0.5
    n_pos = max(1, min(n - 1, round(class_balance * n)))
    threshold = np.sort(score)[-n_pos]  # the top n_pos scores are fills
    y = (score >= threshold).astype(int)
    t = np.arange(n, dtype=int)
    return qml.Dataset(X=X, y=y, t=t, feature_names=_rfq_feature_names(d))


class RFQFillDomain(QuantumKernelClassificationDomain):
    problem = "rfq_fill"
    positive_label = "filled"

    def _assumptions(self) -> list[str]:
        return [
            "Temporal holdout: the latest RFQs are the test set (no look-ahead).",
            "Fill depends only on features known at quote time (no post-trade leakage).",
        ]

    def _synthetic_dataset(self, spec: ProblemSpec) -> qml.Dataset:
        cfg = spec.classification
        assert cfg is not None and cfg.synthetic is not None
        s = cfg.synthetic
        return make_rfq_synthetic(
            n_samples=s.n_samples,
            n_features=s.n_features,
            n_informative=s.n_informative,
            class_balance=s.class_balance,
            separability=s.separability,
            seed=spec.execution_policy.seed,
        )
