"""Run the benchmark: solve each spec and distil an honest leaderboard entry."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import yaml

from ..core.config import Settings
from ..core.ir import ProblemSpec, parse_spec
from ..core.result import DecisionCategory
from ..core.workflow import RunContext
from ..pipeline import solve

# Categories that assert quantum did something better than the fair classical
# comparator (an "advantage", possibly pending reproduction). Parity is NOT one.
_ADVANTAGE = {
    DecisionCategory.QUANTUM_IMPROVEMENT_OBSERVED,
    DecisionCategory.INDEPENDENT_REPRODUCTION_REQUIRED,
    DecisionCategory.POTENTIAL_OPERATIONAL_ADVANTAGE,
}


@dataclass
class ArenaEntry:
    name: str
    problem: str
    task_type: str
    score_kind: str
    classical_method: str
    classical_score: float | None
    quantum_method: str | None
    quantum_score: float | None
    verdict: str
    quantum_helps: bool
    quantum_contribution: bool | None
    runtime_s: float
    evidence_digest: str


@dataclass
class ArenaResult:
    entries: list[ArenaEntry] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.entries)

    @property
    def n_advantage(self) -> int:
        return sum(1 for e in self.entries if e.quantum_helps)

    @property
    def n_parity(self) -> int:
        return sum(1 for e in self.entries if e.verdict == DecisionCategory.QUANTUM_PARITY.value)

    @property
    def n_classical(self) -> int:
        return self.n - self.n_advantage - self.n_parity


def _quantum_contribution(ctx: RunContext) -> bool | None:
    """The 'shot distribution beat random' signal, from whichever quantum ran."""
    reports = ctx.state.verification or {}
    for method in ("qaoa_sim", "qaoa_ibm", "quantum_kernel_ridge"):
        rep = reports.get(method)
        if rep is not None and rep.quantum_contribution:
            return bool(rep.quantum_contribution.get("contributed"))
    return None


def _entry(name: str, ctx: RunContext, runtime_s: float) -> ArenaEntry:
    st = ctx.state
    audit = st.audit
    verdict = audit.category if audit else None
    digest = st.bundle.manifest.get("evidence_digest", "") if st.bundle else ""

    if st.dataset is not None:
        task = "classification"
        models = st.class_models or {}
        classical = max(
            (m for m in models.values() if m.kind == "classical"),
            key=lambda m: m.metric,
            default=None,
        )
        quantum = models.get("quantum_kernel_ridge")
        classical_method = classical.name if classical else "n/a"
        classical_score = classical.metric if classical else None
        quantum_method = "quantum_kernel_ridge" if quantum else None
        quantum_score = quantum.metric if quantum else None
        metric_name = classical.metric_name if classical else "metric"
        score_kind = f"{metric_name} (higher better)"
    else:
        task = "optimization"
        qa = st.instance_qaoa
        classical_method = "classical_milp"
        classical_score = st.classical_milp.objective if st.classical_milp else None
        quantum_method = qa.method if qa else None
        quantum_score = qa.objective if (qa and qa.feasible) else None
        score_kind = "objective (lower better)"

    return ArenaEntry(
        name=name,
        problem=ctx.spec.problem,
        task_type=task,
        score_kind=score_kind,
        classical_method=classical_method,
        classical_score=classical_score,
        quantum_method=quantum_method,
        quantum_score=quantum_score,
        verdict=verdict.value if verdict else "n/a",
        quantum_helps=bool(verdict in _ADVANTAGE),
        quantum_contribution=_quantum_contribution(ctx),
        runtime_s=runtime_s,
        evidence_digest=digest,
    )


def run_arena(specs: dict[str, ProblemSpec], *, settings: Settings | None = None) -> ArenaResult:
    """Solve every spec and collect leaderboard entries (deterministic per seed)."""
    result = ArenaResult()
    for name, spec in specs.items():
        t0 = time.perf_counter()
        ctx = solve(spec, settings=settings)
        result.entries.append(_entry(name, ctx, time.perf_counter() - t0))
    return result


def load_example_specs() -> dict[str, ProblemSpec]:
    """The bundled example specs, as the default benchmark suite."""
    from ..studio import list_example_specs

    out: dict[str, ProblemSpec] = {}
    for ex in list_example_specs():
        data = yaml.safe_load(ex["yaml"])
        if isinstance(data, dict):
            out[ex["name"]] = parse_spec(data)
    return out
