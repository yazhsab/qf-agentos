"""Agent 9 — Governance.

Assembles the evidence bundle (manifest + technical report + model card) into the
RunContext. The CLI/SDK is responsible for persisting it to the evidence store.
"""

from __future__ import annotations

from ..core.workflow import RunContext
from ..governance.report import build_bundle


def governance_agent(ctx: RunContext) -> str:
    bundle = build_bundle(ctx)
    ctx.state.bundle = bundle
    n_results = len(bundle.manifest["results"])
    return (
        f"Evidence bundle assembled: manifest ({n_results} results), technical report, model card."
    )
