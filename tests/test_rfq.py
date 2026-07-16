"""RFQ fill-prediction family: shared quantum-kernel base, synthetic data, pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from qf_agentos import solve
from qf_agentos.core.errors import SpecError
from qf_agentos.core.ir import parse_spec
from qf_agentos.core.result import DecisionCategory
from qf_agentos.finance import get_domain
from qf_agentos.finance.qkernel import QuantumKernelClassificationDomain
from qf_agentos.finance.rfq import RFQFillDomain, make_rfq_synthetic
from qf_test_utils import make_rfq_spec

# --- IR validation --------------------------------------------------------


def test_rfq_spec_validates():
    spec = make_rfq_spec()
    assert spec.problem == "rfq_fill"
    assert spec.classification is not None


def test_rfq_needs_data_or_synthetic():
    with pytest.raises(SpecError):
        parse_spec({"problem": "rfq_fill", "classification": {"feature_budget": 2}})


def test_rfq_feature_budget_cannot_exceed_features():
    with pytest.raises(SpecError):
        parse_spec(
            {
                "problem": "rfq_fill",
                "classification": {
                    "feature_budget": 10,
                    "synthetic": {"n_features": 4, "n_samples": 60},
                },
            }
        )


# --- synthetic data -------------------------------------------------------


def test_rfq_synthetic_is_deterministic_and_balanced():
    a = make_rfq_synthetic(n_samples=200, n_features=6, class_balance=0.35, seed=7)
    b = make_rfq_synthetic(n_samples=200, n_features=6, class_balance=0.35, seed=7)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)
    # Fill rate is pinned to class_balance by the top-score threshold.
    assert a.positive_rate == pytest.approx(0.35, abs=0.01)
    assert set(a.y.tolist()) == {0, 1}
    assert a.feature_names[0] == "quoted_spread_bps"


def test_rfq_synthetic_different_seeds_differ():
    a = make_rfq_synthetic(seed=1)
    b = make_rfq_synthetic(seed=2)
    assert not np.array_equal(a.X, b.X)


# --- shared base / domain -------------------------------------------------


def test_rfq_domain_registered_and_is_classification():
    dom = get_domain("rfq_fill")
    assert dom.problem == "rfq_fill"
    assert isinstance(dom, RFQFillDomain)
    assert isinstance(dom, QuantumKernelClassificationDomain)


def test_rfq_summary_uses_filled_wording():
    dom = get_domain("rfq_fill")
    rep = dom.requirements(make_rfq_spec())
    assert "filled" in rep.summary
    assert any("no look-ahead" in a for a in rep.assumptions)


# --- pipeline reuse -------------------------------------------------------


@pytest.mark.slow
def test_pipeline_runs_on_rfq():
    spec = make_rfq_spec()
    ctx = solve(spec)
    st = ctx.state
    assert st.dataset is not None
    assert st.audit is not None
    assert not st.errors
    # Both classical baselines and the quantum kernel were trained.
    assert "rbf_kernel_ridge" in st.class_models
    assert "quantum_kernel_ridge" in st.class_models
    # The honest comparator is the RBF kernel (same learner), not logistic.
    qrep = ctx.state.verification["quantum_kernel_ridge"]
    assert qrep.quantum_contribution["compared_to"] == "rbf_kernel_ridge"


@pytest.mark.slow
def test_rfq_determinism():
    spec = make_rfq_spec()
    a = solve(spec).state.bundle.manifest["evidence_digest"]
    b = solve(spec).state.bundle.manifest["evidence_digest"]
    assert a == b


def test_pipeline_runs_on_rfq_without_quantum():
    spec = make_rfq_spec(allow_gate_model=False)
    ctx = solve(spec)
    assert ctx.state.audit is not None
    assert ctx.state.bundle is not None
    assert not ctx.state.errors
    # No quantum kernel when the gate model is disallowed.
    assert "quantum_kernel_ridge" not in ctx.state.class_models
    assert ctx.state.audit.category in {
        DecisionCategory.QUANTUM_NOT_FEASIBLE,
        DecisionCategory.CLASSICAL_PREFERRED,
    }
