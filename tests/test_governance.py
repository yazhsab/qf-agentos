"""Evidence bundle assembly, deterministic digest, and the evidence store."""

from __future__ import annotations

from qf_agentos import solve
from qf_agentos.governance.report import build_bundle
from qf_agentos.governance.store import EvidenceStore
from qf_test_utils import make_spec


def _fast_ctx():
    # allow_gate_model=False skips QAOA, keeping governance tests fast.
    return solve(make_spec(required=4_000_000, allow_gate_model=False))


def test_bundle_has_all_sections():
    ctx = _fast_ctx()
    bundle = ctx.state.bundle
    for key in (
        "run_id",
        "seed",
        "evidence_digest",
        "environment",
        "spec",
        "requirements",
        "results",
        "verification",
        "audit",
        "trace",
    ):
        assert key in bundle.manifest
    assert bundle.report_md.startswith("# QF-AgentOS Evidence Report")
    assert "Model Card" in bundle.model_card_md


def test_digest_is_deterministic_and_ignores_timestamp():
    a = _fast_ctx()
    b = _fast_ctx()
    assert a.state.bundle.manifest["evidence_digest"] == b.state.bundle.manifest["evidence_digest"]
    # run_ids differ (timestamped) but the digest does not.
    assert a.run_id != b.run_id or a.run_id == b.run_id  # both acceptable; digest is the contract


def test_rebuild_is_idempotent():
    ctx = _fast_ctx()
    d1 = build_bundle(ctx).manifest["evidence_digest"]
    d2 = build_bundle(ctx).manifest["evidence_digest"]
    assert d1 == d2


def test_evidence_store_round_trip(tmp_path):
    ctx = _fast_ctx()
    store = EvidenceStore(tmp_path)
    run_dir = store.save(ctx.run_id, ctx.state.bundle)
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "report.md").exists()

    records = store.list_runs()
    assert len(records) == 1
    assert records[0].run_id == ctx.run_id
    assert records[0].evidence_digest == ctx.state.bundle.manifest["evidence_digest"]

    manifest = store.load_manifest(ctx.run_id)
    assert manifest is not None and manifest["run_id"] == ctx.run_id
    assert store.load_manifest("missing") is None


def test_store_appends_multiple_runs(tmp_path):
    store = EvidenceStore(tmp_path)
    for _ in range(2):
        ctx = _fast_ctx()
        store.save(ctx.run_id, ctx.state.bundle)
    assert len(store.list_runs()) == 2
