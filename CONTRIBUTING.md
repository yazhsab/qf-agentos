# Contributing to QF-AgentOS

Thanks for your interest in improving QF-AgentOS. This document explains how to
set up a development environment, the quality bar for changes, and how to extend
the platform with new skills, backends, and problem families.

## Development setup

```bash
git clone https://github.com/qf-agentos/qf-agentos
cd qf-agentos
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pre-commit install            # optional but recommended
```

## Quality gates (must pass before a PR is merged)

```bash
ruff format .                 # format
ruff check .                  # lint
mypy                          # static types (strict)
pytest --cov=qf_agentos       # tests + coverage
```

CI runs all of the above on Python 3.11, 3.12, and 3.13. New code must keep the
suite green and must not lower coverage.

## Design principles (please preserve)

QF-AgentOS's value is its *honesty*, not raw quantum enthusiasm. When
contributing, keep these invariants:

1. **Fair comparison is structural.** The classical comparator solves the *same*
   instance the quantum backend sees. Never benchmark quantum against a strawman.
2. **Encoding loss is mechanical.** Anything a relaxation drops must be re-checked
   by the Verification agent against the full constraint set.
3. **The auditor biases toward classical.** A quantum result must be verified
   feasible *and* at least match the classical optimum before earning more than
   `CLASSICAL PREFERRED`.
4. **Abstention is a valid outcome.** The planner may decline quantum entirely.
5. **Determinism.** Same spec + same seed ⇒ identical evidence bundle.
6. **No autonomous financial actions.** The system never trades, moves money,
   changes limits/fraud rules, runs paid QPUs without approval, or claims quantum
   advantage without verification.

## Extending the platform

### Add a Quantum Skill

Create `src/qf_agentos/skills/<your-skill>/skill.yaml` following
[`collateral_optimizer/skill.yaml`](src/qf_agentos/skills/collateral_optimizer/skill.yaml).
It is discovered automatically by `qf-agent skills`.

### Add a solver backend

Implement the `Backend` protocol in
[`backends/base.py`](src/qf_agentos/backends/base.py) and register it in the
backend registry. Real hardware backends must:

- import their SDK lazily and raise `BackendUnavailableError` with an install hint
  when missing;
- treat credentials as `SecretStr` from `core.config.Settings`;
- be gated behind autonomy level **L3** and explicit approval for paid execution.

### Add a problem family

1. Extend the Finance IR in [`core/ir.py`](src/qf_agentos/core/ir.py) (add fields,
   never loosen validation).
2. Add a domain module under `finance/` with the MILP/QUBO builders and a
   `check_constraints` that evaluates the *full* constraint set.
3. Wire it into the pipeline behind the `problem` discriminator.

## Commit & PR conventions

- Keep PRs focused; one logical change per PR.
- Write a clear description and link any related issue.
- Add or update tests for every behavioural change.
- By contributing you agree your work is licensed under Apache-2.0.

## Reporting issues

Use GitHub Issues for bugs and feature requests. For security-sensitive reports,
see [SECURITY.md](SECURITY.md).
