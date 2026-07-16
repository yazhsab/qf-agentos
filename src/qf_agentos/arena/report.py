"""Render a QF-Arena result as a Markdown leaderboard."""

from __future__ import annotations

from .runner import ArenaEntry, ArenaResult


def _fmt(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:,.4g}"


def _helps(e: ArenaEntry) -> str:
    if e.quantum_helps:
        return "**yes**"
    return "no"


def render_leaderboard(result: ArenaResult) -> str:
    lines: list[str] = ["# QF-Arena leaderboard\n"]
    lines.append(
        f"**{result.n} problems** · {result.n_advantage} quantum advantage · "
        f"{result.n_parity} parity · {result.n_classical} classical preferred.\n"
    )
    if result.n and result.n_advantage == 0:
        lines.append(
            "> Across the whole suite, quantum does not beat the exact classical "
            "comparator on any problem — parity at best. That is the honest result the "
            "platform is built to surface.\n"
        )
    lines.append("| problem | family | task | classical | quantum | verdict | quantum helps |")
    lines.append("|---|---|---|---|---|---|---|")
    for e in sorted(result.entries, key=lambda x: x.name):
        lines.append(
            f"| {e.name} | {e.problem} | {e.task_type} | "
            f"{e.classical_method} {_fmt(e.classical_score)} | "
            f"{(e.quantum_method or 'none')} {_fmt(e.quantum_score)} | "
            f"{e.verdict} | {_helps(e)} |"
        )
    kinds = sorted({e.score_kind for e in result.entries})
    lines.append("")
    lines.append(f"_Score convention by task: {', '.join(kinds)}. Deterministic per seed._")
    return "\n".join(lines)
