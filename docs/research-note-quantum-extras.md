# Research Note: Amplitude Estimation and Tensor-Network Baselines in QF-AgentOS

**Author:** Quantum Finance Scientist (QF-AgentOS)
**Date:** 2026-07-16
**Status:** Working draft

Two research-grade quantum-finance extras were added to QF-AgentOS, each chosen
because it sharpens the platform's core question — *does quantum actually help?* —
rather than blurring it. This note records the methodology, the honest complexity
analysis, and the verdict for both.

---

# Part A — Quantum Amplitude Estimation for risk

*Implementation: `qf_agentos/finance/qae.py`; CLI: `qf-agent estimate`.*

## 1. Problem statement

Estimate an expectation under a discretised distribution:

$$\mathbb{E}[f(X)] = \sum_{i=0}^{2^m-1} p_i\, f_i,\qquad f_i \in [0,1].$$

Concrete instances shipped: the **expected loss** of a discretised truncated-normal
loss distribution over $2^m$ levels, and a **VaR-style exceedance probability**
$\Pr[\text{loss} > t]$ (payoff = tail indicator). Both are the canonical targets of
quantum risk analysis (Woerner & Egger 2019) and derivative pricing (Stamatopoulos
et al. 2020).

## 2. Classical baseline

**Best classical method:** For the small discretised distributions a statevector can
hold ($m \le \sim 12$), the expectation is an **exact sum** — $O(2^m)$, instant, zero
error. For continuous/high-dimensional versions the workhorses are **Monte Carlo**
($O(1/\epsilon^2)$ samples for RMSE $\epsilon$) and its stronger cousins **quasi-MC**
(Sobol) and **multilevel MC** (Giles) — all production-scale.
**Why this is the right baseline:** any quantum claim must beat *exact summation* at
these sizes and *MC/QMC/MLMC* asymptotically — not a naive comparator (guardrail G-baseline).

## 3. Quantum approach

**Primitive:** Quantum Amplitude Estimation, specifically **Maximum-Likelihood AE**
(Suzuki et al. 2020) — chosen over canonical (QPE-based) QAE because it avoids deep
phase-estimation circuits, and over a raw single-shot estimate because MLAE resolves
the amplitude-aliasing across Grover powers.

```
Algorithm: MLAE for E[f]
Input: probabilities p, payoff f in [0,1], Grover powers {k}, shots N
Output: estimate of a = E[f]
1. Build A: |0> -> sum_i sqrt(p_i)|i>(sqrt(1-f_i)|0> + sqrt(f_i)|1>)   # good-state prob == a
2. Grover operator Q = -A S_0 A^dagger O   (O marks objective qubit = |1>)
3. For each power k: P_k = <good| Q^k A |0> = sin^2((2k+1)theta);  h_k ~ Binomial(N, P_k)
4. theta_hat = argmax_theta sum_k [ h_k log sin^2((2k+1)theta) + (N-h_k) log cos^2(...) ]
5. return a_hat = sin^2(theta_hat)
```

The amplitude oracle is built and **validated exactly**: the objective-qubit good-state
probability equals $\sum_i p_i f_i$ to machine precision (a unit test enforces this).
The MLAE loop uses a real Grover operator on the statevector; only the shot sampling is
stochastic.

## 4. Complexity analysis

| Cost component | Classical (MC) | Quantum (QAE) |
|---|---|---|
| Per-query | $O(1)$ sample | $O(1)$ oracle call |
| State preparation | — | $O(2^m)$ (arbitrary distribution) |
| Queries for RMSE $\epsilon$ | $O(1/\epsilon^2)$ | $O(1/\epsilon)$ |
| Readout | — | $O(1)$ (single amplitude) |
| **End-to-end** | $O(1/\epsilon^2)$ | $O(2^m + 1/\epsilon)$ |

**Speedup status:** **Proven** quadratic in queries (Brassard, Høyer, Mosca, Tapp 2002;
Montanaro 2015).
**Assumptions:** amplitude-oracle access — i.e. an efficient state-preparation unitary.
For an *arbitrary* financial distribution this costs $O(2^m)$ and is the binding
constraint (guardrail G2/G3): the quadratic advantage survives only when state prep is
amortised across many pricings (same distribution, many strikes) (Stamatopoulos et al.
2020) or the distribution is efficiently loadable (Grover–Rudolph for log-concave;
qGAN, Zoufal et al. 2019).

## 5. Hardware feasibility

**Required regime:** early fault tolerance. The quadratic advantage requires large
Grover powers → deep *coherent* circuits; NISQ noise destroys them at useful precision.
**Circuit depth:** $\propto \max_k(2k+1)\cdot \text{depth}(A)$, with $\text{depth}(A)$
itself $O(2^m)$ for exact loading.
**Realistic horizon:** the threshold for a genuine advantage in derivative pricing is
estimated at thousands of logical qubits and very deep circuits (Chakrabarti et al.
2021) — a 10+ year horizon.

## 6. Empirical result (this implementation)

On a 4-qubit expected-loss instance (`qf-agent estimate --qubits 4`): MLAE recovers the
exact expectation to $\sim 10^{-3}$, but **classical MC at the same query budget matches
or beats it**, and exact summation ($2^4=16$ terms) is instant and error-free. The
resource analysis reports the proven $1/\epsilon$ vs $1/\epsilon^2$ ratio while flagging
the $O(2^m)$ state-prep bottleneck. **Verdict: CLASSICAL PREFERRED** at every size the
platform can run.

## 7. Open questions

1. State preparation competitive with the QAE inner loop (closing the end-to-end gap).
2. QAE + multilevel-MC coupling — can they compose below either alone?
3. Error-mitigated QAE that preserves the quadratic scaling.

---

# Part B — Tensor-network classical-simulability baseline

*Implementation: `qf_agentos/finance/tensor_network.py`; CLI: `qf-agent simulability`.*

## 1. Problem statement

Given the QAOA output state $|\psi\rangle$ for a reduced portfolio/collateral/routing
QUBO, decide: **is this quantum circuit classically simulable by a tensor network?** If
a matrix-product state (MPS) of modest bond dimension $\chi$ reproduces $|\psi\rangle$,
then a classical algorithm with $O(n\chi^2)$ resources matches the circuit — undercutting
any quantum-advantage claim from that instance.

## 2. Classical baseline

**Method:** MPS / DMRG-style simulation. An $n$-qubit state has an exact MPS with bond
dimension up to $2^{\lfloor n/2\rfloor}$; the *useful* regime is when the entanglement
across every cut is low, so $\chi$ stays small and simulation is polynomial (Vidal 2003).
This is simultaneously a classical *competitor* and a diagnostic.

## 3. Method

For each bipartition, compute the Schmidt spectrum (SVD of the reshaped amplitude
tensor), the bipartite von-Neumann entropy $S = -\sum_i \lambda_i^2\log_2\lambda_i^2$,
and the bond dimension $\chi_\epsilon$ needed to capture fidelity $1-\epsilon$. Then
reconstruct a truncated bond-$\chi$ MPS and report its fidelity $|\langle\psi_\text{MPS}|
\psi\rangle|^2$. Validated on product ($S=0,\chi=1$), Bell/GHZ ($S=1,\chi=2$), and
random ($\chi=2^{n/2}$ for exact recovery) states.

## 4. Complexity

| Representation | Parameters | Simulable when |
|---|---|---|
| Statevector | $2^n$ | always (small $n$) |
| MPS (bond $\chi$) | $O(n\chi^2)$ | $\chi \ll 2^{n/2}$ (low entanglement) |

## 5. Empirical result

The shipped dense-QUBO instances produce **highly entangled** QAOA states — e.g. the
13-qubit collateral instance has max bipartite entropy $\approx 5.2$ bits and needs
$\chi\approx 43$ against an exact-rank max of $64$. So an MPS gives *no* compression here;
the analysis reports this honestly as a **small-instance artefact, not hardware advantage**
(exact statevector already solves it classically). The value delivered is a *quantified*
entanglement/compressibility measure and an honest crossover statement: MPS-simulable
(low-$\chi$) circuits are trivially classical; high-$\chi$ ones still fall to exact
statevector at these sizes.

## 6. Caveats

- Dense (all-to-all) QUBOs generate near-maximal entanglement — MPS is the wrong classical
  tool for them; the honest baseline there is exact enumeration / MILP (already the platform's).
- The analysis uses the exact statevector, so it *quantifies* rather than *replaces*
  classical simulation — which is the intended, honest role.

---

## Overall conclusion

Neither extra produces a quantum advantage on any instance QF-AgentOS can run — and both
are engineered to *say so with evidence*. QAE's advantage is proven but asymptotic and
state-prep-bottlenecked (early-FT horizon); the tensor-network baseline quantifies exactly
when a QAOA circuit is classically simulable. Both are faithful to the platform's thesis:
an agent that runs the quantum method honestly and reports, with proof, that classical
still wins.

## References

- Brassard, Høyer, Mosca, Tapp 2002. *Quantum amplitude amplification and estimation.*
- Montanaro 2015. *Quantum speedup of Monte Carlo methods.* Proc. R. Soc. A.
- Suzuki et al. 2020. *Amplitude estimation without phase estimation.* Quantum Inf. Process.
- Grinko et al. 2021. *Iterative quantum amplitude estimation.* npj Quantum Information.
- Woerner, Egger 2019. *Quantum risk analysis.* npj Quantum Information.
- Stamatopoulos, Egger et al. 2020. *Option pricing using quantum computers.* Quantum.
- Zoufal, Lucchi, Woerner 2019. *qGANs for learning and loading random distributions.*
- Chakrabarti et al. 2021. *A threshold for quantum advantage in derivative pricing.* Quantum.
- Vidal 2003. *Efficient classical simulation of slightly entangled quantum computations.* PRL.

_Research artifact — decision-support only, not investment advice._
