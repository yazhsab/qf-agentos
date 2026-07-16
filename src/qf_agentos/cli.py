"""``qf-agent`` command-line interface.

qf-agent solve   examples/collateral-allocation.yaml     # full pipeline + evidence bundle
qf-agent explain examples/collateral-allocation.yaml     # L0: understand + formulate, no solving
qf-agent plan    examples/collateral-allocation.yaml     # L1: experiment plan, no execution
qf-agent skills                                          # list installed Quantum Skills
qf-agent backends                                        # list backends + availability (incl. qaoa_ibm)
qf-agent arena --out arena/                              # benchmark every family — honest leaderboard
qf-agent estimate --qubits 4                             # quantum amplitude estimation vs classical MC
qf-agent simulability examples/collateral-allocation.yaml # tensor-network classical-simulability check
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
def backends() -> None:
    """List quantum/classical backends and whether each is available right now."""
    from .backends.registry import discover_capabilities

    table = Table(title="Backends")
    for col in ("name", "available", "detail"):
        table.add_column(col)
    for c in discover_capabilities():
        table.add_row(
            c.name,
            "[green]yes[/]" if c.available else "[dim]no[/]",
            escape(c.detail),
        )
    console.print(table)
    console.print(
        "[dim]Set QF_IBM_TOKEN (and QF_IBM_INSTANCE if needed) to enable qaoa_ibm; "
        "real hardware also needs autonomy L3 + --approve.[/]"
    )


@app.command()
def arena(
    out: Path = typer.Option(None, "--out", "-o", help="Write arena.md + arena.json here."),
) -> None:
    """Benchmark every example problem across backends — an honest leaderboard."""
    import json
    from dataclasses import asdict

    from .arena import load_example_specs, render_leaderboard, run_arena

    specs = load_example_specs()
    console.print(
        f"[dim]Running QF-Arena on {len(specs)} problems (this solves each end-to-end)…[/]"
    )
    result = run_arena(specs)

    table = Table(title="QF-Arena leaderboard")
    for col in ("problem", "family", "classical", "quantum", "verdict", "helps"):
        table.add_column(col)
    for e in sorted(result.entries, key=lambda x: x.name):
        cs = "—" if e.classical_score is None else f"{e.classical_score:,.4g}"
        qs = "—" if e.quantum_score is None else f"{e.quantum_score:,.4g}"
        table.add_row(
            e.name,
            e.problem,
            f"{e.classical_method} {cs}",
            f"{e.quantum_method or 'none'} {qs}",
            e.verdict,
            "[green]yes[/]" if e.quantum_helps else "[dim]no[/]",
        )
    console.print(table)
    console.print(
        f"[bold]{result.n} problems[/] · {result.n_advantage} quantum advantage · "
        f"{result.n_parity} parity · {result.n_classical} classical preferred."
    )
    if result.n and result.n_advantage == 0:
        console.print(
            "[dim]Quantum does not beat the exact classical comparator on any problem — "
            "parity at best. The honest result the platform is built to surface.[/]"
        )
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "arena.md").write_text(render_leaderboard(result))
        (out / "arena.json").write_text(
            json.dumps([asdict(e) for e in result.entries], indent=2, default=str)
        )
        console.print(f"Wrote {out}/arena.md and {out}/arena.json")


@app.command()
def estimate(
    qubits: int = typer.Option(4, "--qubits", "-m", help="Distribution qubits (2^m loss levels)."),
    tail: float = typer.Option(
        None, "--tail", help="Tail threshold t for a VaR-style P(loss>t); omit for expected loss."
    ),
    shots: int = typer.Option(200, help="Shots per Grover power."),
    seed: int = typer.Option(7, help="Seed (deterministic)."),
    out: Path = typer.Option(None, "--out", "-o", help="Write estimate.json here."),
) -> None:
    """Quantum Amplitude Estimation vs classical — an honest risk-estimation demo."""
    from .finance.qae import (
        classical_monte_carlo,
        make_normal_loss_instance,
        mlae,
        quantum_available_for_qae,
        resource_analysis,
    )

    if not quantum_available_for_qae():
        console.print("[red]QAE needs qiskit: pip install 'qf-agentos[qiskit]'[/]")
        raise typer.Exit(code=3)

    inst = make_normal_loss_instance(qubits, tail_threshold=tail)
    console.rule(f"[bold]QF-AgentOS estimate[/] · {inst.label} · {qubits} qubits")
    r = mlae(inst, shots=shots, seed=seed)
    mc, se = classical_monte_carlo(inst, r.oracle_calls, seed=seed)
    ra = resource_analysis(inst)

    table = Table(title=f"Amplitude estimation — {inst.label}")
    for col in ("method", "estimate", "abs error", "queries"):
        table.add_column(col)
    table.add_row("exact (classical sum)", f"{r.exact:.6f}", "0", f"{2**qubits}")
    table.add_row("QAE (MLAE)", f"{r.estimate:.6f}", f"{r.abs_error:.5f}", f"{r.oracle_calls:,}")
    table.add_row(
        "classical Monte Carlo", f"{mc:.6f}", f"{abs(mc - r.exact):.5f}", f"{r.oracle_calls:,}"
    )
    console.print(table)
    console.print(
        f"[dim]QAE query complexity O(1/ε) vs MC O(1/ε²): for RMSE {ra['target_rmse']:.0e}, "
        f"{ra['qae_oracle_queries']:,} vs {ra['classical_mc_samples']:,} queries "
        f"({ra['quadratic_query_ratio']:.0f}× fewer) — but state prep is "
        f"{ra['state_preparation_gates']:,} gates (O(2^m)).[/]"
    )
    console.print(Panel(Text(ra["verdict"]), title="Honest verdict", border_style="yellow"))
    if out:
        import json

        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "instance": inst.label,
            "m_qubits": qubits,
            "exact": r.exact,
            "qae_estimate": r.estimate,
            "qae_abs_error": r.abs_error,
            "qae_oracle_calls": r.oracle_calls,
            "mc_estimate": mc,
            "mc_abs_error": abs(mc - r.exact),
            "resource": ra,
        }
        (out / "estimate.json").write_text(json.dumps(payload, indent=2, default=str))
        console.print(f"Wrote {out}/estimate.json")
    console.print(f"[dim]{_DISCLAIMER}[/]")


@app.command()
def simulability(
    spec_path: Path = typer.Argument(..., exists=True, readable=True),
    reps: int = typer.Option(1, help="QAOA reps."),
    fidelity: float = typer.Option(0.99, help="Target MPS fidelity."),
) -> None:
    """Tensor-network baseline: is this problem's QAOA circuit classically simulable?"""
    from .backends import quantum_available
    from .core.domain import ProblemDomain
    from .finance import get_domain
    from .finance.tensor_network import qaoa_statevector, simulability_analysis

    if not quantum_available():
        console.print("[red]Needs qiskit: pip install 'qf-agentos[qiskit]'[/]")
        raise typer.Exit(code=3)
    spec = _load_spec_or_exit(spec_path)
    domain = get_domain(spec.problem)
    if not isinstance(domain, ProblemDomain):
        console.print(
            "[red]simulability applies to optimisation problems (QAOA), not this task.[/]"
        )
        raise typer.Exit(code=2)

    pol = spec.execution_policy
    inst = domain.reduce_to_instance(spec, pol.max_effective_qubits)
    if getattr(inst, "degenerate", False):
        console.print("[yellow]Degenerate instance — no quantum circuit to analyse.[/]")
        return
    slack = max(0, min(4, pol.max_effective_qubits - inst.n_qubits))
    qubo = domain.build_qubo(inst, slack_bits=slack)
    if qubo.n == 0 or qubo.n > 22:
        console.print(f"[yellow]{qubo.n}-qubit instance outside the statevector budget.[/]")
        return

    console.rule(f"[bold]QF-AgentOS simulability[/] · {spec.problem} · {qubo.n} qubits")
    a = simulability_analysis(qaoa_statevector(qubo, reps=reps), qubo.n, fidelity=fidelity)
    table = Table(title="Tensor-network (MPS) classical-simulability analysis")
    for col in ("metric", "value"):
        table.add_column(col)
    table.add_row("max entanglement entropy", f"{a['max_entanglement_entropy_bits']:.3f} bits")
    table.add_row(
        f"bond dimension for {fidelity:.0%} fidelity",
        f"{a['bond_dimension_for_fidelity']} (exact-rank max {a['exact_max_bond_dimension']})",
    )
    table.add_row("truncated-MPS fidelity", f"{a['truncated_mps_fidelity']:.4f}")
    table.add_row(
        "MPS vs statevector params",
        f"{a['mps_parameters']:,} vs {a['statevector_parameters']:,} ({a['compression_ratio']:.1f}×)",
    )
    table.add_row("classically simulable", "yes" if a["classically_simulable"] else "no")
    console.print(table)
    console.print(Panel(Text(a["verdict"]), title="Honest verdict", border_style="yellow"))
    console.print(f"[dim]{_DISCLAIMER}[/]")


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
