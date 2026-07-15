# QF-AgentOS

**An open-source agentic operating system for quantum finance.**

[![CI](https://github.com/qf-agentos/qf-agentos/actions/workflows/ci.yml/badge.svg)](https://github.com/qf-agentos/qf-agentos/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Typed](https://img.shields.io/badge/typing-strict-brightgreen)

Give QF-AgentOS a financial optimisation problem. A team of deterministic agents
formulates it, builds strong classical baselines, identifies a
quantum-compatible sub-problem, runs gate-model simulation (and, where
authorised, real QPUs), **verifies** feasibility and quantum contribution, and
honestly reports whether quantum technology should be used.

> The defining principle: *an agent that knows **when not** to use quantum
> computing is more valuable than one that always produces a circuit.*

This is closer to "Claude Code for quantum-finance engineering" than to a
conventional QML package.

---

## Status

`v0.1.0` (beta). A production-grade vertical slice for the **collateral-allocation**
use case, runnable out of the box with a real classical solver (SciPy/HiGHS MILP)
and real gate-model simulation (Qiskit QAOA). It ships a CLI, a REST API, a Docker
image, a typed SDK, a backend registry with provider-neutral adapters, a policy
engine, structured logging, and a governance/evidence layer.

- **Quality gates:** `ruff` (lint+format), `mypy --strict`, `pytest` (~91% coverage),
  CI across Python 3.11/3.12/3.13.

## Install

```bash
pip install -e ".[all,dev]"     # classical + qiskit + pennylane + server + tests
# minimal (classical only):  pip install -e .
# gate-model simulation:      pip install -e ".[qiskit]"
# REST API:                   pip install -e ".[server]"
# real IBM / D-Wave hardware: pip install -e ".[ibm]"  /  ".[dwave]"
```

Requires Python ≥ 3.11.

## Quickstart (CLI)

```bash
qf-agent solve   examples/collateral-allocation.yaml   # full pipeline + evidence bundle
qf-agent explain examples/collateral-allocation.yaml   # L0: understand + formulate only
qf-agent plan    examples/collateral-allocation.yaml   # L1: experiment plan, no execution
qf-agent solve   examples/abstention-demo.yaml         # honest quantum abstention
qf-agent solve   examples/infeasible.yaml              # honest infeasibility (exits 4)
qf-agent runs                                          # list recorded runs
qf-agent skills                                        # list installed Quantum Skills
qf-agent serve                                         # run the REST API
```

## SDK

```python
from qf_agentos import load_spec, solve

ctx = solve(load_spec("examples/collateral-allocation.yaml"))
print(ctx.state.audit.rendered)            # the FINAL DECISION block
print(ctx.state.bundle.manifest["evidence_digest"])   # deterministic per spec+seed
```

## REST API

```bash
qf-agent serve --host 0.0.0.0 --port 8000
# or: uvicorn qf_agentos.api:app
```

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | liveness + version |
| `GET /backends` | truthful backend capability discovery |
| `GET /skills` | installed Quantum Skills |
| `POST /solve` | run the pipeline on a spec → decision + evidence |
| `GET /runs`, `GET /runs/{id}` | the experiment registry |

## Docker

```bash
docker compose up --build          # serves the API on :8000, persists evidence/
docker run --rm qf-agentos qf-agent version
```

## What actually runs

For every problem the pipeline compares *like with like*:

| Layer | Method | Backend |
|---|---|---|
| Full problem — lower bound | Continuous LP relaxation | SciPy/HiGHS |
| Full problem — fair binary comparator & recommendation | Binary MILP (exact) | SciPy/HiGHS |
| Research instance — classical optimum | Binary MILP on the reduced instance | SciPy/HiGHS |
| Research instance — QUBO ground truth | Brute-force enumeration | CPU |
| Research instance — strong heuristic | Simulated annealing | CPU |
| Research instance — quantum | QUBO → Ising → **QAOA** | Qiskit statevector sim |

The QUBO encodes coverage as a proper inequality (`cov ≥ R'`) via binary slack
bits with an adaptive penalty, so its ground state is genuinely feasible on
coverage. It deliberately drops concentration and HQLA — and the **Verification
agent re-checks the decoded quantum solution against the full constraint set**,
so relaxation artifacts are caught, not hidden.

### Provider-neutral backends

`qaoa_sim` (Qiskit) and `qaoa_pennylane` run locally; `qaoa_ibm` (IBM Runtime)
and `dwave_hybrid` (D-Wave Leap) are **real, credential-gated** adapters that are
inert without their SDK + token and are gated behind autonomy **L3** + approval.

## The agent team

1. **Requirements** — validate the Finance IR, discover missing constraints, sanity-check feasibility.
2. **Formulation** — enumerate LP / MILP / QUBO / Ising and what each can and cannot represent.
3. **Classical Baseline** — serious LP + MILP via HiGHS (never a strawman).
4/5. **Hardware Planner** — *agent–hardware negotiation*: reduce to a qubit-sized instance, discover backends, estimate depth/cost, and **decide whether quantum is warranted — abstention is a first-class outcome.**
6. **Execution** — run the instance ladder: MILP → QUBO-exact → SA → QAOA, enforcing the policy engine.
7. **Verification** — independent objective recomputation, exact constraint checks, QUBO-ground-state gap, and **quantum-contribution accounting** (did the circuit beat random sampling?).
8. **Quantum-Advantage Auditor** — assigns an honest outcome category and renders the `FINAL DECISION` block.
9. **Governance** — machine-readable manifest, technical report, model card, deterministic evidence digest.

### Outcome categories

`CLASSICAL PREFERRED` · `QUANTUM NOT FEASIBLE ON PRESENT HARDWARE` ·
`QUANTUM PARITY` · `QUANTUM RESEARCH CANDIDATE` · `QUANTUM IMPROVEMENT OBSERVED` ·
`INDEPENDENT REPRODUCTION REQUIRED` · `POTENTIAL OPERATIONAL ADVANTAGE`

The auditor's bias is toward classical: a quantum result must be **verified
feasible** and at least match the classical optimum before it earns anything
better than "classical preferred".

## Human-control model (autonomy levels)

| Level | Capability | Extra gate |
|---|---|---|
| L0 | Explain and recommend | — |
| L1 | Generate experiment plan | — |
| L2 | Execute simulators automatically | — |
| L3 | Execute paid QPU jobs | budget + **human approval** |
| L4 | Recommend production decisions | **human approval** mandatory |

The system never places trades, moves money, alters limits/fraud rules, runs
paid QPU jobs without approval, or claims quantum advantage without verification.

## Configuration

Environment variables (prefix `QF_`, see [`.env.example`](.env.example)):
`QF_LOG_LEVEL`, `QF_LOG_FORMAT` (`text`/`json`), `QF_DEFAULT_SEED`,
`QF_STATEVECTOR_QUBIT_LIMIT`, `QF_EVIDENCE_DIR`, and credentials
`QF_IBM_TOKEN` / `QF_DWAVE_TOKEN` (held as secrets, never logged).

## Project layout

```
src/qf_agentos/
├── core/         # Finance IR, typed state, artifacts, policy, config, logging, workflow
├── finance/      # collateral domain: MILP / QUBO / Ising + constraint checking
├── backends/     # base protocol, registry, solvers (exact/SA/QAOA), IBM/D-Wave/PennyLane
├── agents/       # the 9 agents (one module each)
├── governance/   # evidence bundle + file-based experiment store
├── skills/       # Quantum Skills plugin registry
├── api.py        # FastAPI service
└── cli.py        # the `qf-agent` command
docs/             # ARCHITECTURE.md
examples/         # runnable specs (collateral, abstention, infeasible)
tests/            # 80+ tests, property-based invariants, ~91% coverage
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Contributing & extension guide](CONTRIBUTING.md)
- [Security & safety model](SECURITY.md)
- [Changelog](CHANGELOG.md)

## Disclaimer

Experimental research software. Not investment advice, not a licensed advisor,
and not a production trading system. Outputs are decision-support only.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
