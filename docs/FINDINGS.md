# Findings — does quantum help in finance?

**Short answer: no. Not on any problem this platform can run, and not by a small
margin.** This document is the evidence, and — more usefully — the *reasons*.

Everything here is reproducible: `qf-agent arena` regenerates the leaderboard, and a
regression test (`test_full_suite_has_no_quantum_advantage`) fails the build if the
platform ever starts claiming otherwise.

---

## 1. The leaderboard

```
$ qf-agent arena
7 problems · 0 quantum advantage · 1 parity · 6 classical preferred.
```

| problem | family | task | classical | quantum | verdict |
|---|---|---|---|---|---|
| abstention-demo | collateral_allocation | optimization | MILP 3,550 | none | QUANTUM NOT FEASIBLE ON PRESENT HARDWARE |
| collateral-allocation | collateral_allocation | optimization | MILP 7,140 | QAOA — *(infeasible)* | CLASSICAL PREFERRED |
| payment-routing | payment_routing | optimization | MILP 1,869 | QAOA — *(infeasible)* | CLASSICAL PREFERRED |
| settlement-netting | settlement_netting | optimization | MILP 0 | QAOA 0 | QUANTUM PARITY |
| fraud-detection | fraud_detection | classification | AUC 0.9826 | quantum kernel AUC 0.5568 | CLASSICAL PREFERRED |
| rfq-fill | rfq_fill | classification | AUC 0.945 | quantum kernel AUC 0.4964 | CLASSICAL PREFERRED |
| infeasible | collateral_allocation | optimization | infeasible | none | CLASSICAL PREFERRED |

Score convention: objective = lower better; AUC = higher better. Deterministic per seed.

**Note what "—" means for quantum**: the QAOA *did* run and *did* decode a solution —
it was **infeasible** against the exact constraints, so it earns no score. That is the
verification agent doing its job.

## 2. Why this is trustworthy

A negative result is only worth something if the comparison was fair. The guardrails:

- **The classical baseline is not a strawman.** Exact binary MILP via SciPy/HiGHS
  (plus an LP lower bound), solving these instances in ~1 ms. For classification,
  logistic regression *and* RBF kernel-ridge — with the quantum kernel judged against
  the **same** kernel-ridge learner, isolating the kernel.
- **Every quantum solution is mechanically re-verified** against the full exact
  constraint set — never trusted from the QUBO energy.
- **Statistical honesty in classification** — temporal holdout, data-leakage checks,
  bootstrap significance vs the RBF comparator.
- **Abstention is a first-class outcome** — the planner refuses when quantum isn't warranted.
- **The auditor is biased toward classical** — quantum must be *verified feasible* and at
  least match the classical optimum to earn better than "classical preferred".
- **Determinism** — same spec + seed ⇒ byte-identical `evidence_digest`.

## 3. Why quantum loses — per family

The interesting part. **In most cases the encoding decided the outcome before the
solver ever ran.**

### collateral_allocation / payment_routing — the QUBO drops or distorts constraints
The QUBO encodes the objective plus *some* constraints via penalties/slack; others
(concentration, HQLA, route capacity, approval floors) are **verification-only**. The
QAOA optimum therefore breaches the real constraint set, and the verifier catches it.
Classical MILP handles all constraints exactly, in milliseconds.

### settlement_netting — a structural impossibility result
The most technically interesting finding. Liquidity (`net outflow ≤ balance`) is a
**packing constraint**. Settling an obligation of amount `a` that overshoots a payer's
headroom by `δ` gains reward `~a` but costs only the quadratic penalty `A·δ²`. Because
`δ` can be arbitrarily small while `a` is large, **no tractable penalty weight `A` makes
the QUBO's ground state feasible** — you would need `A ~ max_amount · total`, which
swamps the objective and destroys the QAOA landscape.

An adversarial sweep of **320 random instances** confirmed it: 25–90% of QUBO ground
states are liquidity-infeasible, **and every single one was caught and reported
honestly** (zero laundered as feasible). Parity here means QAOA matched the MILP on a
benign instance — not that it is competitive.

> **This is not a tuning failure. It is a property of the problem class.** Some financial
> constraints simply do not survive QUBO encoding at usable penalty weights.

### fraud_detection / rfq_fill — the quantum kernel doesn't separate the data
The ZZ-feature-map fidelity kernel scores AUC **0.557** and **0.496** — the latter is
*worse than a coin flip* — against classical **0.983** / **0.945** on identical data,
the same learner, and the same temporal holdout. The bootstrap significance test finds
no quantum improvement. There is no subtlety here: it just doesn't work.

## 4. Real quantum hardware (IBM)

A full 9-step pipeline run on **`ibm_kingston`** (IBM Quantum, Open plan):

| Signal | Value |
|---|---|
| Steps / errors | 9 / **0** |
| Backend | `real hardware · ibm_kingston` |
| QPU access time | **3.0 s** |
| Reached QUBO optimum | **yes** (hardware sampled the ground-state energy) |
| Quantum contribution | **yes** (shot distribution beat uniform random) |
| Decoded solution | **infeasible** (violated coverage) |
| **Verdict** | **CLASSICAL PREFERRED** |

Read that carefully, because it is the whole thesis in one run: **the quantum hardware
genuinely did something — it beat random sampling and found the QUBO's optimum — and it
still lost**, because the QUBO's optimum is not a feasible answer to the real problem.
Good quantum execution of a lossy encoding is still a wrong answer.

Getting there required fixing three real bugs the live run exposed (a retired IBM API
channel, the free plan forbidding session mode, and a verification crash on the
hardware's shot histogram). All are fixed; see [real-qpu.md](real-qpu.md).

## 5. Research-grade extras

### Quantum Amplitude Estimation (`qf-agent estimate`)
QAE has a **proven quadratic speedup** over Monte Carlo — `O(1/ε)` vs `O(1/ε²)` queries
(Brassard et al. 2002). Our MLAE implementation recovers the exact expectation to ~1e-3
and the amplitude oracle is validated to machine precision. And yet:

- At runnable sizes, **exact summation is instant and error-free**, and classical MC at
  the same query budget **matches or beats** the QAE estimate.
- **State preparation costs `O(2^m)` gates** — the binding constraint. The speedup
  survives only if amortised across many pricings (Stamatopoulos et al. 2020).
- A useful advantage needs **early fault tolerance** (Chakrabarti et al. 2021) — a 10+
  year horizon.

**Verdict: CLASSICAL PREFERRED.** The speedup is real, proven, and irrelevant today.

### Tensor-network baseline (`qf-agent simulability`)
Measures whether the QAOA circuit is classically simulable by an MPS. On these dense
QUBOs the output states are *highly* entangled (13-qubit collateral: entropy ≈ 5.2 bits,
bond dimension ≈ 43 vs an exact-rank max of 64), so an MPS gives **no compression** —
reported honestly as a small-instance artefact, **not** as evidence of hardware advantage.
(Exact statevector already solves it classically.)

Methodology + citations: [research-note-quantum-extras.md](research-note-quantum-extras.md).

## 6. Limitations — what these findings do *not* prove

Honesty cuts both ways. This platform **does not** show that quantum computing is
useless in finance forever. It shows:

- **At these scales** (≤ ~22 qubits, reduced research instances), classical wins — which
  is unsurprising, since exact methods solve these instances outright. The platform
  **cannot** locate a crossover at production scale; that question stays open.
- **For these encodings.** A better formulation (not a better solver) is where any future
  quantum result would have to come from. The settlement finding suggests some constraint
  classes are structurally hostile to QUBO — a testable, generalisable claim.
- **For these techniques** (QAOA, quantum kernels, QAE). Fault-tolerant algorithms are
  out of reach of any simulator.

What it *does* establish, reproducibly: **the popular near-term quantum-finance
demonstrations do not survive a fair comparison** — and the reason is usually the
encoding, not the hardware.

## 7. Reproducing this

```bash
pip install -e ".[all,dev]"
qf-agent arena --out arena/        # the leaderboard above
qf-agent solve examples/settlement-netting.yaml
qf-agent estimate --qubits 4       # QAE vs classical MC
qf-agent simulability examples/collateral-allocation.yaml
pytest -q                          # 229 tests, incl. the no-advantage regression test
```

Every run writes an evidence bundle (`manifest.json`, `report.md`, `model_card.md`) with
a deterministic digest. Same spec + seed ⇒ identical digest.

---

_Research artifact — decision-support only, not investment advice._
