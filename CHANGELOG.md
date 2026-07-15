# Changelog

All notable changes to QF-AgentOS are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Typed pipeline state** (`core/state.py`, `core/artifacts.py`) replacing the
  stringly-typed inter-agent bus; the whole contract is now mypy-checked.
- **Backend abstraction**: `QuboSolver` protocol + registry, with real
  credential-gated adapters for IBM Runtime and D-Wave, plus a PennyLane
  simulator backend (`backends/`).
- **Policy enforcement** at real call sites: `RUN_SIMULATOR`/`RUN_PAID_QPU`
  (Execution) and `RECOMMEND_PRODUCTION` (Auditor) gates.
- **REST API** (`api.py`, FastAPI) and an `EvidenceStore` experiment registry
  (`governance/store.py`); new `qf-agent serve` and `qf-agent runs` commands.
- **Docker** image + `docker-compose.yml`; CI (lint/type/test matrix + build +
  wheel-content + Docker smoke) and a PyPI release workflow.
- Typed exception hierarchy (`core/errors.py`), env-driven settings with secret
  handling (`core/config.py`), and structured `run_id`-correlated logging with
  optional JSON output (`core/observability.py`).
- `py.typed` marker; wheel now ships skill manifests and the typing marker.
- Quality toolchain: ruff, mypy (strict), pytest coverage gate (85%).
- Comprehensive test suite (80+ tests, property-based QUBO/Ising invariants,
  CLI + API tests, ~91% coverage).
- Docs: `docs/ARCHITECTURE.md`, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT,
  CHANGELOG, NOTICE, CITATION.cff; abstention and infeasible example specs.

### Changed
- **QUBO encoding** now models coverage as a proper inequality (`cov ≥ R'`) via
  binary slack bits with an adaptive penalty, so the QUBO ground state respects
  the coverage requirement instead of being systematically infeasible.
- Robust spec loading: `load_spec`/`parse_spec` raise actionable `SpecError`s
  (no tracebacks, no input leakage); concentration attributes and inventory
  size are validated.
- Workflow steps are isolated: a failing agent is recorded and the run still
  emits a (partial) evidence bundle.

### Fixed
- Packaging: skill manifests and `py.typed` are now included in the wheel.
- Edge cases: empty/all-ineligible inventory, degenerate instances, and
  divide-by-zero in instance reduction no longer crash the pipeline.
- Latent `None`-dereference/arithmetic bugs in the auditor and classical agent.

## [0.1.0] - 2026-07-15

### Added
- Initial runnable vertical slice for the `collateral_allocation` use case.
- Finance IR (Pydantic v2), deterministic workflow engine, and policy engine
  with autonomy levels L0–L4.
- Nine agents: Requirements, Formulation, Classical Baseline, Hardware Planner,
  Quantum Algorithm, Execution, Verification, Quantum-Advantage Auditor,
  Governance.
- Classical backend (SciPy/HiGHS LP + binary MILP) and quantum backend
  (QUBO → Ising → QAOA on a Qiskit statevector simulator).
- Deterministic verification with quantum-contribution accounting.
- Evidence bundle: experiment manifest, technical report, model card.
- `qf-agent` CLI (`solve`, `explain`, `plan`, `skills`, `version`) and Quantum
  Skills registry.

[Unreleased]: https://github.com/qf-agentos/qf-agentos/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/qf-agentos/qf-agentos/releases/tag/v0.1.0
