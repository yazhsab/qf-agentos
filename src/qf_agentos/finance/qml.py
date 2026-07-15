"""Quantum-kernel classification core (a *second* quantum technique).

Everything numerical for the classification pipeline: deterministic synthetic
data, a temporal train/test split, standardisation, an RBF (classical) and a
fidelity **quantum** kernel, a kernel-ridge classifier used identically for both
(so the comparison isolates the kernel), metrics, a data-leakage check, and a
bootstrap significance test.

The quantum kernel requires the optional ``qiskit`` extra; the classical path
works without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..core.result import SolveResult

FloatArray = NDArray[np.float64]


@dataclass
class TrainedModel:
    """A trained classifier's test-set outcome (higher metric = better)."""

    name: str
    kind: str  # "classical" | "quantum"
    metric_name: str
    metric: float
    error: float  # 1 - metric (lower = better; keeps the auditor's convention)
    metrics: dict[str, float]
    test_scores: FloatArray
    y_test: NDArray[np.int_]
    runtime_s: float = 0.0
    feasible: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class Dataset:
    """Feature matrix, binary labels {0,1}, and a monotonic time index."""

    X: FloatArray  # (n, d)
    y: NDArray[np.int_]  # (n,) in {0,1}
    t: NDArray[np.int_]  # (n,) time order
    feature_names: list[str]

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean()) if self.n else 0.0


def make_synthetic(
    *,
    n_samples: int = 240,
    n_features: int = 6,
    n_informative: int = 3,
    class_balance: float = 0.2,
    separability: float = 0.9,
    seed: int = 7,
) -> Dataset:
    """Deterministic two-class dataset (imbalanced, fraud-like). Seeded."""
    rng = np.random.default_rng(seed)
    n, d = n_samples, n_features
    n_pos = max(1, min(n - 1, round(class_balance * n)))
    y = np.array([1] * n_pos + [0] * (n - n_pos), dtype=int)
    rng.shuffle(y)
    X = rng.standard_normal((n, d))
    info = min(n_informative, d)
    for j in range(info):
        X[y == 1, j] += separability
        X[y == 0, j] -= separability
    t = np.arange(n, dtype=int)
    names = [f"f{j}" for j in range(d)]
    return Dataset(X=X, y=y, t=t, feature_names=names)


def temporal_split(ds: Dataset, test_frac: float) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Hold out the latest ``test_frac`` of samples by time (no look-ahead)."""
    order = np.argsort(ds.t)
    n_test = max(1, round(test_frac * ds.n))
    test_idx = np.sort(order[-n_test:])
    train_idx = np.sort(order[:-n_test])
    return train_idx, test_idx


def standardize(train_X: FloatArray, X: FloatArray) -> FloatArray:
    """Standardise ``X`` using TRAIN statistics only (guards against leakage)."""
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def rbf_kernel(A: FloatArray, B: FloatArray, gamma: float) -> FloatArray:
    sq = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def quantum_fidelity_kernel(
    A: FloatArray, B: FloatArray, *, reps: int = 1, scale: float = 1.0
) -> FloatArray:
    """Fidelity kernel K(a,b) = |<phi(a)|phi(b)>|^2 with a ZZ feature map.

    Precomputes one statevector per row, then the Gram matrix — O(n) circuits,
    not O(n^2). Requires qiskit.
    """
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.quantum_info import Statevector

    d = A.shape[1]
    fmap = ZZFeatureMap(feature_dimension=d, reps=reps)

    def states(M: FloatArray) -> FloatArray:
        out = np.empty((M.shape[0], 2**d), dtype=complex)
        for i, row in enumerate(M):
            out[i] = Statevector(fmap.assign_parameters(scale * row)).data
        return out

    sa = states(A)
    sb = sa if B is A else states(B)
    overlap = sa.conj() @ sb.T
    return np.abs(overlap) ** 2


# ---------------------------------------------------------------------------
# Kernel-ridge classifier (identical for both kernels)
# ---------------------------------------------------------------------------


def krr_fit(k_train: FloatArray, y01: NDArray[np.int_], lam: float) -> FloatArray:
    """Solve (K + lam I) alpha = y, with y in {-1, +1}."""
    y = 2.0 * y01.astype(float) - 1.0
    n = k_train.shape[0]
    return np.linalg.solve(k_train + lam * np.eye(n), y)


def krr_scores(k_test_train: FloatArray, alpha: FloatArray) -> FloatArray:
    return k_test_train @ alpha


def logistic_fit(
    X: FloatArray, y01: NDArray[np.int_], l2: float = 1e-2, iters: int = 25
) -> FloatArray:
    """Logistic regression via IRLS (Newton). Returns weights incl. bias."""
    n, d = X.shape
    xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(d + 1)
    reg = np.eye(d + 1)
    reg[0, 0] = 0.0  # don't regularise the bias
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(xb @ w, -30, 30)))
        wgt = np.clip(p * (1.0 - p), 1e-6, None)
        grad = xb.T @ (p - y01) + l2 * reg @ w
        hess = xb.T @ (xb * wgt[:, None]) + l2 * reg
        try:
            w = w - np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
    return w


def logistic_scores(X: FloatArray, w: FloatArray) -> FloatArray:
    xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return xb @ w


def evaluate_model(
    name: str,
    kind: str,
    metric_name: str,
    scores: FloatArray,
    y_test: NDArray[np.int_],
    *,
    runtime_s: float = 0.0,
) -> TrainedModel:
    metrics = {
        "auc": roc_auc(scores, y_test),
        "accuracy": accuracy(scores, y_test),
        "f1": f1_score(scores, y_test),
    }
    primary = metrics[metric_name]
    return TrainedModel(
        name=name,
        kind=kind,
        metric_name=metric_name,
        metric=primary,
        error=1.0 - primary,
        metrics=metrics,
        test_scores=scores,
        y_test=y_test,
        runtime_s=runtime_s,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def roc_auc(scores: FloatArray, y01: NDArray[np.int_]) -> float:
    pos = scores[y01 == 1]
    neg = scores[y01 == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(len(counts))
    np.add.at(sum_ranks, inv, ranks)
    avg = sum_ranks / counts
    ranks = avg[inv]
    r_pos = ranks[y01 == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def accuracy(scores: FloatArray, y01: NDArray[np.int_], threshold: float = 0.0) -> float:
    pred = (scores >= threshold).astype(int)
    return float((pred == y01).mean())


def f1_score(scores: FloatArray, y01: NDArray[np.int_], threshold: float = 0.0) -> float:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y01 == 1)).sum())
    fp = int(((pred == 1) & (y01 == 0)).sum())
    fn = int(((pred == 0) & (y01 == 1)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec))


def metric_value(name: str, scores: FloatArray, y01: NDArray[np.int_]) -> float:
    if name == "auc":
        return roc_auc(scores, y01)
    if name == "accuracy":
        return accuracy(scores, y01)
    return f1_score(scores, y01)


# ---------------------------------------------------------------------------
# Validation: leakage + significance
# ---------------------------------------------------------------------------


def model_to_result(model: TrainedModel, *, scope: str, backend: str) -> SolveResult:
    """Represent a classifier as a SolveResult (objective = error, lower is better)."""
    from ..core.result import SolveResult

    return SolveResult(
        method=model.name,
        kind=model.kind,
        backend=backend,
        scope=scope,
        feasible=model.feasible,
        objective=model.error,
        allocation=None,
        runtime_s=model.runtime_s,
        metadata={
            "metric_name": model.metric_name,
            "primary_metric": model.metric,
            **{f"metric_{k}": v for k, v in model.metrics.items()},
        },
    )


def leakage_report(train_idx: NDArray[np.int_], test_idx: NDArray[np.int_]) -> dict[str, Any]:
    overlap = set(train_idx.tolist()) & set(test_idx.tolist())
    return {
        "train_test_disjoint": len(overlap) == 0,
        "n_overlap": len(overlap),
        "standardisation": "train-statistics only",
    }


def bootstrap_significance(
    scores_a: FloatArray,
    scores_b: FloatArray,
    y01: NDArray[np.int_],
    *,
    metric: str,
    n_boot: int = 500,
    seed: int = 7,
) -> dict[str, Any]:
    """Bootstrap the test-set metric difference (a - b). Deterministic given seed."""
    rng = np.random.default_rng(seed)
    n = len(y01)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y01[idx]
        if yb.min() == yb.max():  # degenerate resample
            diffs[b] = 0.0
            continue
        diffs[b] = metric_value(metric, scores_a[idx], yb) - metric_value(metric, scores_b[idx], yb)
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {
        "metric": metric,
        "mean_diff": float(diffs.mean()),
        "ci95": [lo, hi],
        "significant": bool(lo > 0 or hi < 0),  # CI excludes zero
        "n_boot": n_boot,
    }
