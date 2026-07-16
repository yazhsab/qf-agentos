"""Quantum-kernel classification: the shared base for classification families.

Both fraud detection and RFQ fill-prediction are binary classification problems
solved the same honest way: train strong classical baselines (logistic regression
+ RBF kernel-ridge) and a quantum fidelity-kernel classifier on the SAME
kernel-ridge learner, on a temporal holdout, and report whether any quantum
improvement is statistically significant (bootstrap CI). Data-leakage and
temporal-validity checks are deterministic.

Only the dataset (a domain-specific synthetic generator) and a little wording
differ between families; everything else lives here. A concrete family subclasses
:class:`QuantumKernelClassificationDomain`, sets ``problem`` + ``positive_label``,
and implements ``_synthetic_dataset``.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.artifacts import Formulation, FormulationCatalogue, RequirementsReport
from ..core.domain import ClassificationDomain
from ..core.ir import ProblemSpec
from ..core.result import ConstraintCheck, VerificationReport
from . import qml

# Cap on training samples fed to the O(n^2) quantum kernel (a reduced instance).
QUANTUM_SAMPLE_CAP = 160


def _select_features(ds: qml.Dataset, train_idx: NDArray[np.int_], budget: int) -> list[int]:
    """Top-``budget`` features by standardised class-mean separation on TRAIN only."""
    Xtr, ytr = ds.X[train_idx], ds.y[train_idx]
    scores = np.zeros(ds.d)
    for j in range(ds.d):
        col = Xtr[:, j]
        s = col.std() or 1.0
        scores[j] = abs(col[ytr == 1].mean() - col[ytr == 0].mean()) / s
    return sorted(range(ds.d), key=lambda j: scores[j], reverse=True)[:budget]


def _stratified_subsample(
    y: NDArray[np.int_], idx: NDArray[np.int_], cap: int, seed: int
) -> NDArray[np.int_]:
    if len(idx) <= cap:
        return idx
    rng = np.random.default_rng(seed)
    yy = y[idx]
    pos = idx[yy == 1]
    neg = idx[yy == 0]
    n_pos = max(1, min(len(pos), round(cap * len(pos) / len(idx))))
    n_neg = cap - n_pos
    sel = np.concatenate(
        [
            rng.choice(pos, n_pos, replace=False),
            rng.choice(neg, min(n_neg, len(neg)), replace=False),
        ]
    )
    return np.sort(sel)


class QuantumKernelClassificationDomain(ClassificationDomain):
    """Shared quantum-kernel classification pipeline. Subclasses provide the data."""

    problem: str
    positive_label: str = "positive"  # wording in summaries ("positive" / "filled")

    # --- domain hooks (subclasses override) -------------------------------

    @abstractmethod
    def _synthetic_dataset(self, spec: ProblemSpec) -> qml.Dataset:
        """Build the family's synthetic dataset from ``spec.classification.synthetic``."""

    def _assumptions(self) -> list[str]:
        return ["Temporal holdout: the latest samples are the test set (no look-ahead)."]

    def _intended_use(self) -> str:
        return f"a quantum-kernel classifier for the {self.problem} task"

    # --- understanding / formulation --------------------------------------

    def requirements(self, spec: ProblemSpec) -> RequirementsReport:
        cfg = spec.classification
        assert cfg is not None
        if spec.features:
            n = len(spec.features)
            d = len(spec.features[0]) if spec.features else 0
            pos_rate = float(np.mean(spec.labels)) if spec.labels else 0.0
        else:
            assert cfg.synthetic is not None
            n, d, pos_rate = (
                cfg.synthetic.n_samples,
                cfg.synthetic.n_features,
                cfg.synthetic.class_balance,
            )
        gaps: list[str] = []
        if pos_rate < 0.1:
            gaps.append(
                f"Highly imbalanced ({pos_rate:.0%} {self.positive_label}) — accuracy is "
                "misleading; AUC preferred."
            )
        if cfg.feature_budget < d:
            gaps.append(f"Quantum kernel uses {cfg.feature_budget} of {d} features (qubit budget).")
        return RequirementsReport(
            problem=self.problem,
            summary=(
                f"{n} samples x {d} features, {pos_rate:.0%} {self.positive_label}; "
                f"metric={cfg.target_metric}"
            ),
            metrics={
                "n_samples": float(n),
                "n_features": float(d),
                "positive_rate": pos_rate,
                "test_fraction": cfg.test_fraction,
            },
            feasible_upper_bound=True,
            discovered_gaps=gaps,
            assumptions=self._assumptions(),
            autonomy_level=spec.execution_policy.autonomy_level.value,
        )

    def formulations(self, spec: ProblemSpec) -> FormulationCatalogue:
        cfg = spec.classification
        assert cfg is not None
        return FormulationCatalogue(
            catalogue=[
                Formulation(
                    name="logistic_regression",
                    kind="Linear classifier",
                    variables="weights over all features",
                    represents="a strong, fast linear baseline",
                    note="IRLS-fitted; reference point.",
                ),
                Formulation(
                    name="rbf_kernel_ridge",
                    kind="Kernel method (classical)",
                    variables="dual coefficients over all training samples",
                    represents="non-linear classical baseline (RBF kernel)",
                    note="The FAIR comparator — same learner as the quantum model.",
                ),
                Formulation(
                    name="quantum_kernel_ridge",
                    kind="Quantum kernel method",
                    variables=f"fidelity kernel over {cfg.feature_budget} features",
                    represents="a ZZ feature-map fidelity kernel + kernel ridge",
                    note="Reduced instance; compared to RBF-KRR under the SAME learner.",
                ),
            ],
            selected_classical="rbf_kernel_ridge",
            selected_quantum_path="ZZ feature map -> fidelity kernel -> kernel ridge",
            encoding_loss_note=(
                "The quantum kernel sees a reduced feature/sample instance; results are "
                "compared to the classical RBF kernel under the identical learner, on the "
                "same temporal test set, with a bootstrap significance test."
            ),
        )

    # --- data --------------------------------------------------------------

    def load_dataset(self, spec: ProblemSpec) -> qml.Dataset:
        cfg = spec.classification
        assert cfg is not None
        if spec.features:
            X = np.asarray(spec.features, dtype=float)
            y = np.asarray(spec.labels, dtype=int)
            t = (
                np.asarray(spec.timestamps, dtype=float).argsort().argsort()
                if spec.timestamps
                else np.arange(len(y))
            )
            return qml.Dataset(
                X=X, y=y, t=t.astype(int), feature_names=[f"f{j}" for j in range(X.shape[1])]
            )
        return self._synthetic_dataset(spec)

    def split(self, spec: ProblemSpec, dataset: qml.Dataset) -> tuple[Any, Any]:
        cfg = spec.classification
        assert cfg is not None
        return qml.temporal_split(dataset, cfg.test_fraction)

    # --- classical baselines ----------------------------------------------

    def classical_baselines(
        self, spec: ProblemSpec, dataset: qml.Dataset, split: tuple[Any, Any]
    ) -> dict[str, qml.TrainedModel]:
        cfg = spec.classification
        assert cfg is not None
        tr, te = split
        Xtr = qml.standardize(dataset.X[tr], dataset.X[tr])
        Xte = qml.standardize(dataset.X[tr], dataset.X[te])
        ytr, yte = dataset.y[tr], dataset.y[te]
        m = cfg.target_metric
        out: dict[str, qml.TrainedModel] = {}

        t0 = time.perf_counter()
        w = qml.logistic_fit(Xtr, ytr)
        out["logistic_regression"] = qml.evaluate_model(
            "logistic_regression",
            "classical",
            m,
            qml.logistic_scores(Xte, w),
            yte,
            runtime_s=time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        k_tr = qml.rbf_kernel(Xtr, Xtr, cfg.rbf_gamma)
        k_te = qml.rbf_kernel(Xte, Xtr, cfg.rbf_gamma)
        alpha = qml.krr_fit(k_tr, ytr, cfg.ridge_lambda)
        out["rbf_kernel_ridge"] = qml.evaluate_model(
            "rbf_kernel_ridge",
            "classical",
            m,
            qml.krr_scores(k_te, alpha),
            yte,
            runtime_s=time.perf_counter() - t0,
        )
        return out

    # --- quantum planning + execution -------------------------------------

    def plan_quantum(
        self, spec: ProblemSpec, dataset: qml.Dataset, max_qubits: int, sim_available: bool
    ) -> dict[str, Any]:
        cfg = spec.classification
        assert cfg is not None
        pol = spec.execution_policy
        tr, _te = qml.temporal_split(dataset, cfg.test_fraction)
        budget = min(cfg.feature_budget, max_qubits, dataset.d)
        feats = _select_features(dataset, tr, budget)
        q_train = _stratified_subsample(dataset.y, tr, QUANTUM_SAMPLE_CAP, pol.seed)

        reasons: list[str] = []
        abstain = True
        if not pol.allow_gate_model:
            reasons.append("policy disallows the gate model")
        elif not sim_available:
            reasons.append("gate-model simulator unavailable (install qf-agentos[qiskit])")
        elif budget < 1:
            reasons.append("no features available for a quantum kernel")
        else:
            abstain = False

        return {
            "n_qubits": budget,
            "selected_features": feats,
            "selected_feature_names": [dataset.feature_names[j] for j in feats],
            "n_quantum_train": len(q_train),
            "abstain": abstain,
            "reasons": reasons,
            "target": None if abstain else "gate_model_statevector_sim",
        }

    def run_quantum(
        self,
        spec: ProblemSpec,
        dataset: qml.Dataset,
        split: tuple[Any, Any],
        feature_plan: dict[str, Any],
    ) -> qml.TrainedModel:
        cfg = spec.classification
        assert cfg is not None
        pol = spec.execution_policy
        tr, te = split
        feats = feature_plan["selected_features"]
        q_train = _stratified_subsample(dataset.y, tr, QUANTUM_SAMPLE_CAP, pol.seed)

        Xsel = dataset.X[:, feats]
        Xq = qml.standardize(Xsel[q_train], Xsel[q_train])
        Xte = qml.standardize(Xsel[q_train], Xsel[te])
        t0 = time.perf_counter()
        k_tr = qml.quantum_fidelity_kernel(Xq, Xq)
        k_te = qml.quantum_fidelity_kernel(Xte, Xq)
        alpha = qml.krr_fit(k_tr, dataset.y[q_train], cfg.ridge_lambda)
        scores = qml.krr_scores(k_te, alpha)
        return qml.evaluate_model(
            "quantum_kernel_ridge",
            "quantum",
            cfg.target_metric,
            scores,
            dataset.y[te],
            runtime_s=time.perf_counter() - t0,
        )

    # --- verification ------------------------------------------------------

    def verify(
        self,
        spec: ProblemSpec,
        dataset: qml.Dataset,
        split: tuple[Any, Any],
        models: dict[str, qml.TrainedModel],
    ) -> dict[str, VerificationReport]:
        cfg = spec.classification
        assert cfg is not None
        tr, te = split
        leak = qml.leakage_report(tr, te)
        temporal_ok = dataset.t[te].min() > dataset.t[tr].max() if len(tr) and len(te) else True
        # The significance test isolates the KERNEL: quantum-kernel-ridge vs the
        # RBF-kernel-ridge (the SAME learner). Fall back to the best classical
        # only if the RBF comparator is somehow absent.
        comparator = models.get("rbf_kernel_ridge") or min(
            (m for m in models.values() if m.kind == "classical" and m.feasible),
            key=lambda m: m.error,
            default=None,
        )

        reports: dict[str, VerificationReport] = {}
        for name, model in models.items():
            recomputed = qml.metric_value(model.metric_name, model.test_scores, model.y_test)
            matches = abs(recomputed - model.metric) <= 1e-9
            checks = [
                ConstraintCheck(
                    name="no_data_leakage",
                    satisfied=leak["train_test_disjoint"],
                    value=float(leak["n_overlap"]),
                    limit=0.0,
                    slack=-float(leak["n_overlap"]),
                    detail="train/test disjoint; standardisation uses train stats only",
                ),
                ConstraintCheck(
                    name="temporal_validity",
                    satisfied=temporal_ok,
                    value=1.0 if temporal_ok else 0.0,
                    limit=1.0,
                    slack=0.0 if temporal_ok else -1.0,
                    detail="test set is strictly later in time than train",
                ),
            ]
            rep = VerificationReport(
                method=name,
                scope="classification",
                feasible=model.feasible and leak["train_test_disjoint"] and temporal_ok,
                recomputed_objective=1.0 - recomputed,
                objective_matches_solver=matches,
                checks=checks,
            )
            if model.kind == "quantum" and comparator is not None:
                sig = qml.bootstrap_significance(
                    model.test_scores,
                    comparator.test_scores,
                    model.y_test,
                    metric=model.metric_name,
                    n_boot=cfg.bootstrap,
                    seed=spec.execution_policy.seed,
                )
                rep.quantum_contribution = {
                    "compared_to": comparator.name,
                    "quantum_metric": model.metric,
                    "classical_metric": comparator.metric,
                    "mean_diff": sig["mean_diff"],
                    "ci95": sig["ci95"],
                    "significant": sig["significant"],
                    "contributed": bool(sig["significant"] and sig["mean_diff"] > 0),
                    "verdict": (
                        "Quantum kernel is statistically better than the classical RBF kernel."
                        if (sig["significant"] and sig["mean_diff"] > 0)
                        else "No statistically significant quantum improvement over the "
                        "classical kernel."
                    ),
                }
            reports[name] = rep
        return reports
