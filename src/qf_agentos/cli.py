"""``qf-agent`` command-line interface.

qf-agent solve   examples/collateral-allocation.yaml     # full pipeline + evidence bundle
qf-agent explain examples/collateral-allocation.yaml     # L0: understand + formulate, no solving
qf-agent plan    examples/collateral-allocation.yaml     # L1: experiment plan, no execution
qf-agent skills                                          # list installed Quantum Skills
qf-agent serve                                           # run the REST API (needs the server extra)
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .core.errors import QFAgentOSError
from .core.ir import ProblemSpec, load_spec
from .core.policy import PolicyEngine
from .core.workflow import RunContext
from .governance.store import EvidenceStore
from .pipeline import make_run_id
from .pipeline import solve as solve_spec
from .skills import load_skills

app = typer.Typer(add_completion=False, help="QF-AgentOS — agentic quantum finance.")
console = Console()

_DISCLAIMER = (
    "Research artifact — decision-support only, not investment advice, and not a "
    "production trading decision."
)


def _emit(name: str, summary: str) -> None:
    console.print(f"  [bold cyan]{name:<20}[/] {escape(summary)}")


def _load_spec_or_exit(spec_path: Path) -> ProblemSpec:
    try:
        return load_spec(spec_path)
    except QFAgentOSError as exc:
        console.print(Panel(Text(exc.message), title="Specification error", border_style="red"))
        raise typer.Exit(code=exc.exit_code) from exc


@app.command()
def solve(
    spec_path: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(Path("evidence"), "--out", "-o", help="Evidence output directory."),
    approve: bool = typer.Option(
        False, "--yes", "--approve", help="Approve paid/irreversible steps."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress per-step trace."),
) -> None:
    """Run the full agent pipeline and write an evidence bundle."""
    spec = _load_spec_or_exit(spec_path)
    console.rule(
        f"[bold]QF-AgentOS[/] · {spec.problem} · autonomy {spec.execution_policy.autonomy_level.value}"
    )

    ctx = solve_spec(spec, human_approved=approve, emit=None if quiet else _emit)

    audit = ctx.state.audit
    if audit is not None:
        style = (
            "red"
            if audit.problem_infeasible
            else ("green" if "CLASSICAL" not in audit.category.value else "yellow")
        )
        console.print()
        console.print(
            Panel(Text(audit.rendered), title="Quantum-Advantage Auditor", border_style=style)
        )

    if ctx.state.errors:
        console.print(
            Panel(
                Text("\n".join(f"• {e.step}: {e.message}" for e in ctx.state.errors)),
                title="Step errors (run continued)",
                border_style="red",
            )
        )
    if ctx.warnings:
        console.print(
            Panel(
                Text("\n".join(f"• {w}" for w in ctx.warnings)),
                title="Warnings",
                border_style="yellow",
            )
        )

    bundle = ctx.state.bundle
    if bundle is not None:
        run_dir = EvidenceStore(out).save(ctx.run_id, bundle)
        console.print(
            f"\n[bold]Evidence bundle:[/] {run_dir}/  (manifest.json · report.md · model_card.md)"
        )

    console.print(f"[dim]{_DISCLAIMER}[/]")

    if audit is not None and audit.problem_infeasible:
        raise typer.Exit(code=4)


def _partial_run(spec_path: Path, steps: list[str]) -> RunContext:
    from .agents import (
        formulation_agent,
        hardware_planner_agent,
        quantum_algorithm_agent,
        requirements_agent,
    )

    registry = {
        "requirements": requirements_agent,
        "formulation": formulation_agent,
        "hardware_planner": hardware_planner_agent,
        "quantum_algorithm": quantum_algorithm_agent,
    }
    spec = _load_spec_or_exit(spec_path)
    ctx = RunContext(
        spec=spec,
        policy=PolicyEngine(spec.execution_policy),
        run_id=make_run_id(spec),
        seed=spec.execution_policy.seed,
    )
    for name in steps:
        _emit(name, registry[name](ctx))
    return ctx


@app.command()
def explain(spec_path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """L0 — understand and formulate the problem. No solving, no execution."""
    console.rule("[bold]QF-AgentOS explain[/] (L0)")
    ctx = _partial_run(spec_path, ["requirements", "formulation"])
    formulations = ctx.state.formulations
    if formulations is None:
        return
    table = Table(title="Candidate formulations")
    for col in ("name", "class", "represents", "note"):
        table.add_column(col)
    for f in formulations.catalogue:
        table.add_row(f.name, f.kind, f.represents, f.note)
    console.print(table)
    console.print(f"[dim]{_DISCLAIMER}[/]")


@app.command()
def plan(spec_path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """L1 — produce an experiment plan (reduction, backend choice) without executing."""
    console.rule("[bold]QF-AgentOS plan[/] (L1)")
    ctx = _partial_run(
        spec_path, ["requirements", "formulation", "hardware_planner", "quantum_algorithm"]
    )
    hp = ctx.state.hardware_plan
    sel = ctx.state.quantum_selection
    if hp is None or sel is None:
        return
    target = hp.target or ("ABSTAIN — " + "; ".join(hp.reasons))
    lines = [
        f"Research instance : {hp.n_qubits} qubits (QUBO density {hp.qubo_density})",
        f"Backend target    : {target}",
        f"Est. 2-qubit depth: {hp.estimated_two_qubit_depth}",
        f"Est. cost         : ${hp.estimated_cost_usd:.2f}   Real QPU: {hp.real_qpu}",
        f"Algorithm         : {sel.algorithm}  (reps={sel.reps})",
        "",
        "Encoding losses the verifier will re-check:",
        *[f"  - {loss}" for loss in hp.encoding_losses],
    ]
    console.print(Panel(Text("\n".join(lines)), title="Experiment plan", border_style="cyan"))
    console.print("[dim]Approval required before any paid QPU execution (autonomy L3).[/]")
    console.print(f"[dim]{_DISCLAIMER}[/]")


@app.command()
def skills(extra_dir: Path = typer.Option(None, "--dir", help="Extra skills directory.")) -> None:
    """List installed Quantum Skills."""
    table = Table(title="Quantum Skills")
    for col in ("name", "version", "problem", "builtin"):
        table.add_column(col)
    for s in load_skills(extra_dir):
        table.add_row(
            str(s.get("name")),
            str(s.get("version", "—")),
            str(s.get("problem", "—")),
            "yes" if s.get("_builtin") else "no",
        )
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
) -> None:
    """Run the REST API (requires the 'server' extra)."""
    try:
        import uvicorn
    except Exception as exc:
        console.print(
            "[red]The REST API needs the server extra: pip install 'qf-agentos[server]'[/]"
        )
        raise typer.Exit(code=3) from exc
    import uvicorn

    uvicorn.run("qf_agentos.api:app", host=host, port=port, log_level="info")


@app.command()
def runs(
    out: Path = typer.Option(Path("evidence"), "--out", "-o", help="Evidence directory."),
) -> None:
    """List recorded runs from the evidence store."""
    records = EvidenceStore(out).list_runs()
    if not records:
        console.print("No runs recorded yet.")
        return
    table = Table(title="Recorded runs")
    for col in ("run_id", "created_at", "decision", "recommended", "digest"):
        table.add_column(col)
    for r in records:
        table.add_row(
            r.run_id, r.created_at, r.decision, r.recommended_method, r.evidence_digest[:12]
        )
    console.print(table)


@app.command()
def version() -> None:
    """Print the QF-AgentOS version."""
    from . import __version__

    console.print(f"QF-AgentOS {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
