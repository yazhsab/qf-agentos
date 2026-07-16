# CLI reference — `qf-agent`

Eleven commands. Every command is deterministic (same spec + seed ⇒ same result) and
prints the disclaimer; nothing places trades or moves money.

```bash
qf-agent --help
qf-agent <command> --help
```

| Command | Autonomy | Purpose |
|---|---|---|
| [`solve`](#solve) | L2 (L3 with `--approve`) | Run the full pipeline → decision + evidence bundle |
| [`explain`](#explain) | L0 | Understand + formulate only. No solving |
| [`plan`](#plan) | L1 | Experiment plan (reduction, backend, cost). No execution |
| [`arena`](#arena) | L2 | Benchmark every family → honest leaderboard |
| [`estimate`](#estimate) | L2 | Quantum amplitude estimation vs classical MC |
| [`simulability`](#simulability) | L2 | Tensor-network classical-simulability check |
| [`backends`](#backends) | — | List backends + real availability |
| [`skills`](#skills) | — | List installed Quantum Skills |
| [`runs`](#runs) | — | List recorded runs from the evidence store |
| [`serve`](#serve) | — | Run the REST API + Studio |
| [`version`](#version) | — | Version |

---

## `solve`

Runs all 9 agents and writes an evidence bundle.

```bash
qf-agent solve examples/collateral-allocation.yaml
qf-agent solve spec.yaml --out evidence/ --approve --quiet
```

| Option | Default | Meaning |
|---|---|---|
| `--out`, `-o` | `evidence` | Evidence output directory |
| `--yes`, `--approve` | off | Approve paid/irreversible steps (**required for real QPU**) |
| `--quiet`, `-q` | off | Suppress the per-step trace |

**Exit codes:** `0` success · `2` bad spec · `3` missing extra · `4` problem infeasible · `5` backend error.

Prints the `FINAL DECISION` block and writes `manifest.json`, `report.md`,
`model_card.md`. A failed quantum step is recorded and the run **continues** on the
classical baseline — it never silently aborts.

## `explain`

L0 — validate the spec, enumerate candidate formulations (LP / MILP / QUBO / Ising) and
what each can and cannot represent. No solving.

```bash
qf-agent explain examples/collateral-allocation.yaml
```

## `plan`

L1 — the experiment plan: reduced instance size, QUBO density, backend target (or
abstention + reasons), estimated depth/cost, and the **encoding losses the verifier will
re-check**. No execution.

```bash
qf-agent plan examples/settlement-netting.yaml
```

## `arena`

Benchmarks every bundled example across backends and prints the honest leaderboard.
This is the headline artifact — see [FINDINGS.md](FINDINGS.md).

```bash
qf-agent arena
qf-agent arena --out arena/     # also writes arena.md + arena.json
```

```
7 problems · 0 quantum advantage · 1 parity · 6 classical preferred.
```

## `estimate`

Quantum Amplitude Estimation (MLAE, real Grover operator) vs exact summation and
classical Monte Carlo, on a discretised loss distribution — plus an honest resource
analysis. Needs the `qiskit` extra.

```bash
qf-agent estimate --qubits 4                 # expected loss
qf-agent estimate --qubits 4 --tail 0.7      # VaR-style P(loss > 0.7)
qf-agent estimate --qubits 5 --out out/      # writes estimate.json
```

| Option | Default | Meaning |
|---|---|---|
| `--qubits`, `-m` | 4 | Distribution qubits (2^m loss levels) |
| `--tail` | — | Tail threshold → exceedance probability; omit for expected loss |
| `--shots` | 200 | Shots per Grover power |
| `--seed` | 7 | Deterministic |
| `--out`, `-o` | — | Write `estimate.json` |

## `simulability`

Tensor-network baseline: is this problem's QAOA circuit classically simulable by an MPS?
Reports entanglement entropy, the bond dimension for a target fidelity, truncated-MPS
fidelity, and the honest verdict. Needs `qiskit`; optimisation families only.

```bash
qf-agent simulability examples/collateral-allocation.yaml
qf-agent simulability spec.yaml --reps 2 --fidelity 0.999
```

## `backends`

Every backend and whether it is available **right now** (truthful discovery — a
credentialed backend reports exactly what is missing).

```bash
qf-agent backends
```

```
qaoa_sim            yes   qiskit statevector QAOA
qaoa_ibm            no    set QF_IBM_TOKEN (and optionally QF_IBM_INSTANCE/QF_IBM_BACKEND)
dwave_hybrid        no    install qf-agentos[dwave] (dwave-ocean-sdk)
```

## `skills`

Installed Quantum Skills (a skill = a `skill.yaml` manifest declaring a problem family,
its methods, and its verification checks).

```bash
qf-agent skills
qf-agent skills --dir ./my-skills
```

## `runs`

List recorded runs from the file evidence store.

```bash
qf-agent runs
qf-agent runs --out evidence/
```

## `serve`

Run the REST API and the bundled Studio. Needs the `server` extra. See [API.md](API.md).

```bash
qf-agent serve --host 0.0.0.0 --port 8000
# equivalently: uvicorn qf_agentos.api:app
```

## `version`

```bash
qf-agent version
```

---

## Running on real hardware

```bash
export QF_IBM_TOKEN='…'                       # your shell only; never committed
qf-agent backends                              # confirm qaoa_ibm available=yes
qf-agent solve spec.yaml --approve             # spec needs qpu_backend + autonomy L3
```

Real QPU execution requires **all** of: credentials, `qpu_backend: ibm|dwave`,
`autonomy_level: L3`, sufficient `max_qpu_budget_usd`, and `--approve`. Missing any one
falls back to the simulator with a recorded reason. Full runbook: [real-qpu.md](real-qpu.md).
