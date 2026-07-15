"""Quantum-kernel classification numerical core."""

from __future__ import annotations

import numpy as np
import pytest

from qf_agentos.backends import quantum_available
from qf_agentos.finance.qml import (
    bootstrap_significance,
    krr_fit,
    krr_scores,
    leakage_report,
    make_synthetic,
    metric_value,
    rbf_kernel,
    roc_auc,
    standardize,
    temporal_split,
)


def test_synthetic_is_deterministic_and_shaped():
    a = make_synthetic(n_samples=100, n_features=5, seed=7)
    b = make_synthetic(n_samples=100, n_features=5, seed=7)
    assert a.X.shape == (100, 5) and a.n == 100 and a.d == 5
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)
    assert 0.0 < a.positive_rate < 1.0
    assert set(np.unique(a.y).tolist()) <= {0, 1}


def test_temporal_split_disjoint_and_later_is_test():
    ds = make_synthetic(n_samples=100, seed=1)
    tr, te = temporal_split(ds, 0.3)
    assert len(tr) + len(te) == 100
    assert not (set(tr.tolist()) & set(te.tolist()))
    assert ds.t[te].min() > ds.t[tr].max()  # test is strictly later in time


def test_standardize_uses_train_statistics():
    ds = make_synthetic(n_samples=80, seed=2)
    tr, _te = temporal_split(ds, 0.25)
    Z = standardize(ds.X[tr], ds.X[tr])
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-9)


def test_rbf_kernel_properties():
    X = make_synthetic(n_samples=20, n_features=4, seed=3).X
    K = rbf_kernel(X, X, gamma=0.5)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1.0)
    assert K.min() >= 0.0 and K.max() <= 1.0 + 1e-9


def test_krr_separates_a_separable_dataset():
    ds = make_synthetic(n_samples=160, n_features=4, separability=1.5, seed=5)
    tr, te = temporal_split(ds, 0.3)
    Xtr = standardize(ds.X[tr], ds.X[tr])
    Xte = standardize(ds.X[tr], ds.X[te])
    K_tr = rbf_kernel(Xtr, Xtr, gamma=0.3)
    K_te = rbf_kernel(Xte, Xtr, gamma=0.3)
    alpha = krr_fit(K_tr, ds.y[tr], lam=1e-2)
    scores = krr_scores(K_te, alpha)
    assert roc_auc(scores, ds.y[te]) > 0.75  # well above chance


def test_roc_auc_edge_cases():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == pytest.approx(1.0)
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == pytest.approx(0.0)


def test_leakage_report():
    tr = np.array([0, 1, 2, 3])
    assert leakage_report(tr, np.array([4, 5]))["train_test_disjoint"]
    assert not leakage_report(tr, np.array([3, 4]))["train_test_disjoint"]


def test_bootstrap_significance_deterministic():
    y = np.array([0, 1] * 30)
    a = np.linspace(0, 1, 60)
    b = a[::-1].copy()
    r1 = bootstrap_significance(a, b, y, metric="auc", seed=7)
    r2 = bootstrap_significance(a, b, y, metric="auc", seed=7)
    assert r1 == r2
    assert "ci95" in r1 and isinstance(r1["significant"], bool)


def test_metric_value_dispatch():
    y = np.array([0, 0, 1, 1])
    s = np.array([-1.0, -0.5, 0.5, 1.0])
    assert metric_value("accuracy", s, y) == pytest.approx(1.0)
    assert metric_value("f1", s, y) == pytest.approx(1.0)


@pytest.mark.skipif(not quantum_available(), reason="qiskit not installed")
def test_quantum_fidelity_kernel_properties():
    from qf_agentos.finance.qml import quantum_fidelity_kernel

    X = standardize((ds := make_synthetic(n_samples=12, n_features=3, seed=4)).X, ds.X)
    K = quantum_fidelity_kernel(X, X)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, atol=1e-9)
    assert np.allclose(np.diag(K), 1.0, atol=1e-9)  # |<phi|phi>|^2 = 1
    assert K.min() >= -1e-9 and K.max() <= 1.0 + 1e-9
