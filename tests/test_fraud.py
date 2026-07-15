"""Fraud-detection classification family: IR, pipeline reuse, honesty."""

from __future__ import annotations

import pytest

from qf_agentos import solve
from qf_agentos.core.errors import SpecError
from qf_agentos.core.ir import parse_spec
from qf_agentos.core.result import DecisionCategory
from qf_agentos.finance import get_domain
from qf_test_utils import make_fraud_spec

# --- IR validation --------------------------------------------------------


def test_fraud_spec_validates():
    spec = make_fraud_spec()
    assert spec.problem == "fraud_detection"
    assert spec.classification is not None


def test_missing_classification_block_rejected():
    with pytest.raises(SpecError):
        parse_spec({"problem": "fraud_detection"})


def test_no_data_and_no_synthetic_rejected():
    with pytest.raises(SpecError):
        parse_spec({"problem": "fraud_detection", "classification": {"feature_budget": 2}})


def test_feature_budget_exceeds_features_rejected():
    with pytest.raises(SpecError):
        parse_spec(
            {
                "problem": "fraud_detection",
                "classification": {
                    "feature_budget": 10,
                    "synthetic": {
                        "n_samples": 100,
                        "n_features": 4,
                        "n_informative": 2,
                        "class_balance": 0.3,
                        "separability": 1.0,
                    },
                },
            }
        )


def test_inline_data_length_mismatch_rejected():
    with pytest.raises(SpecError):
        parse_spec(
            {
                "problem": "fraud_detection",
                "classification": {"feature_budget": 2},
                "features": [[0.1, 0.2], [0.3, 0.4]],
                "labels": [0],  # wrong length
            }
        )


def test_non_binary_labels_rejected():
    with pytest.raises(SpecError):
        parse_spec(
            {
                "problem": "fraud_detection",
                "classification": {"feature_budget": 2},
                "features": [[0.1, 0.2], [0.3, 0.4]],
                "labels": [0, 2],
            }
        )


# --- Pipeline reuse -------------------------------------------------------


def test_domain_registered():
    d = get_domain("fraud_detection")
    assert d.problem == "fraud_detection"
    assert d.task_type.value == "classification"


def test_classification_pipeline_runs_and_is_honest():
    ctx = solve(make_fraud_spec())
    assert ctx.state.audit is not None and ctx.state.audit.rendered
    assert ctx.state.bundle is not None and not ctx.state.errors
    # three models trained (2 classical + quantum)
    assert set(ctx.state.class_models) >= {
        "logistic_regression",
        "rbf_kernel_ridge",
        "quantum_kernel_ridge",
    }
    # every model verified, all with a temporal + leakage check
    for rep in ctx.state.verification.values():
        names = {c.name for c in rep.checks}
        assert {"no_data_leakage", "temporal_validity"} <= names


def test_significance_compares_against_the_rbf_kernel():
    # The kernel-isolation guarantee: quantum is judged vs the SAME-learner RBF
    # kernel, not the min-error classical model.
    ctx = solve(make_fraud_spec())
    qc = ctx.state.verification["quantum_kernel_ridge"].quantum_contribution
    assert qc is not None
    assert qc["compared_to"] == "rbf_kernel_ridge"
    rbf = ctx.state.class_models["rbf_kernel_ridge"]
    assert qc["classical_metric"] == rbf.metric


def test_quantum_not_silently_claimed_better():
    ctx = solve(make_fraud_spec())
    audit = ctx.state.audit
    q = ctx.state.class_models["quantum_kernel_ridge"]
    rbf = ctx.state.class_models["rbf_kernel_ridge"]
    # An IMPROVEMENT verdict must be backed by a significant win over the RBF kernel.
    if audit.category == DecisionCategory.QUANTUM_IMPROVEMENT_OBSERVED:
        qc = ctx.state.verification["quantum_kernel_ridge"].quantum_contribution
        assert qc and qc["significant"] and qc["mean_diff"] > 0
        assert q.metric >= rbf.metric


def test_model_card_describes_the_actual_methodology():
    # Regression: a fraud run must NOT ship a collateral model card.
    bundle = solve(make_fraud_spec()).state.bundle
    card = bundle.model_card_md
    assert "fraud_detection" in card
    assert "kernel" in card.lower()
    assert "collateral" not in card.lower()
    assert "MILP" not in card


def test_abstention_when_gate_model_disabled():
    ctx = solve(make_fraud_spec(allow_gate_model=False))
    assert ctx.state.feature_plan["abstain"]
    assert "quantum_kernel_ridge" not in ctx.state.class_models
    assert ctx.state.audit.category == DecisionCategory.QUANTUM_NOT_FEASIBLE


def test_determinism():
    a = solve(make_fraud_spec()).state.bundle.manifest["evidence_digest"]
    b = solve(make_fraud_spec()).state.bundle.manifest["evidence_digest"]
    assert a == b


def test_inline_data_path():
    # Small linearly-separable inline dataset.
    features = [[float(i % 5), float((i * 2) % 7), float(i % 3)] for i in range(60)]
    labels = [1 if row[0] > 2 else 0 for row in features]
    ctx = solve(make_fraud_spec(features=features, labels=labels, feature_budget=3, max_qubits=3))
    assert ctx.state.bundle is not None
    assert "logistic_regression" in ctx.state.class_models
