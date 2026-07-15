# Security Policy

## Reporting a vulnerability

Please report security vulnerabilities privately. **Do not open a public issue
for security problems.**

- Email: `security@qf-agentos.org` (or use GitHub's private "Report a
  vulnerability" flow on the repository's Security tab).
- Include a description, reproduction steps, affected versions, and impact.
- We aim to acknowledge within 3 business days and to provide a remediation
  timeline after triage.

Please give us a reasonable window to fix the issue before public disclosure.

## Supported versions

QF-AgentOS is pre-1.0. Security fixes are applied to the latest released minor
version. Pin a version range in production and upgrade promptly.

## Security & safety model

QF-AgentOS is decision-support software for quantitative finance. Its threat model
and safety guarantees:

### Financial safety (by design)

The system **never** performs autonomous financial actions. It will not:

- place trades or move money;
- alter portfolio limits or fraud rules;
- execute paid QPU jobs without explicit human approval (autonomy **L3**);
- claim quantum advantage without passing deterministic verification.

These are enforced by the policy engine (`core/policy.py`) and the autonomy
levels L0–L4. Outputs are explicitly labelled as research artifacts, not
investment advice.

### Credential handling

- Backend credentials (IBM, D-Wave) are read from the environment as
  `SecretStr` via `core/config.Settings` and are **never** logged or written into
  evidence bundles.
- Do not commit `.env` files or tokens. `.env` is git-ignored.

### Untrusted input

- Problem specs are parsed with `yaml.safe_load` (no arbitrary object
  construction) and validated by Pydantic before use.
- The platform does not execute code contained in specs or in tool/observed
  content.

### Dependencies

- Runtime dependencies are pinned to minimum-version ranges; CI builds against
  the resolved set. Report any dependency with a known CVE via the process above.

## Disclaimer

This software is provided for research and engineering purposes. It is not a
licensed financial advisor and its outputs must not be used as the sole basis for
financial decisions.
