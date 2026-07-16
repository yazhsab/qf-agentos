"""Settlement / liquidity-saving optimisation (RTGS gridlock resolution).

In a real-time gross settlement (RTGS) system, participants hold queued payment
obligations to one another. Settling them one-by-one requires each payer to have
enough liquidity at the moment of settlement, so a queue can *gridlock*: everyone
is waiting on an incoming payment before they can fund an outgoing one. A
liquidity-saving mechanism (LSM) resolves this by choosing a *subset* of queued
obligations to settle **simultaneously** (a netting batch) so that every
participant's net outflow within the batch stays within their available
liquidity, while settling as much value as possible.

    maximise   sum of settled obligation amounts
    subject to (for each participant k)
               outflow_k(batch) - inflow_k(batch) <= balance_k
    (optional) settled value >= min_settled_ratio * total queued value

This plugs into the same agent pipeline as collateral allocation and payment
routing via :class:`ProblemDomain`. The classical comparator is an exact binary
MILP (HiGHS). The QUBO encodes the objective (settle value) **and** each
participant's liquidity inequality via binary slack bits — so, unlike the other
families where a capacity constraint is dropped entirely, here the liquidity
constraints are genuinely in the quantum sub-problem, as a SOFT penalty. Liquidity
is a packing constraint, so a small overshoot is only weakly penalised: the QUBO
ground state is approximately (not exactly) liquidity-feasible, which is itself
concrete evidence that QUBO/QAOA is a poor fit for constrained settlement. The
optional settled-value floor stays verification-only. The decoded batch is always
re-checked EXACTLY against liquidity, so an infeasible batch is caught and never
reported as feasible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ..core.artifacts import Formulation, FormulationCatalogue, RequirementsReport
from ..core.domain import ClassicalBaseline, ProblemDomain
from ..core.ir import Obligation, Participant, ProblemSpec
from ..core.result import Allocation, ConstraintCheck, SolveResult, VerificationReport
from .collateral import LinearSolveOutput, Qubo

# Max binary slack bits *per participant* used to encode each liquidity
# inequality. The reducer uses as many as the qubit budget allows, capped here.
SETTLEMENT_SLACK_BITS = 3
_TOL = 1e-6


# ---------------------------------------------------------------------------
# Classical MILP (real HiGHS via scipy.optimize.milp), sparse
# ---------------------------------------------------------------------------


def solve_settlement_milp(
    obligations: list[Obligation],
    participants: list[Participant],
    min_settled_ratio: float | None,
    *,
    integer: bool,
) -> LinearSolveOutput:
    """Select a settlement batch maximising settled value s.t. per-participant
    liquidity (and an optional settled-value floor).

    The reported objective is the *unsettled* value (total queued minus settled),
    so lower is better — consistent with every other optimisation family.
    """
    p_n = len(obligations)
    if p_n == 0:
        return LinearSolveOutput(True, 0.0, Allocation(x={}), "no obligations")
    if not participants:
        return LinearSolveOutput(False, None, None, "no participants")

    amount = np.array([o.amount for o in obligations], dtype=float)
    total = float(amount.sum())
    # Minimise -settled value == maximise settled value.
    cost = -amount
    ub = np.ones(p_n)

    balance = {p.id: p.balance for p in participants}
    ridx: list[int] = []
    cidx: list[int] = []
    vals: list[float] = []
    lb_c: list[float] = []
    ub_c: list[float] = []
    row = 0

    # (1) liquidity: per participant, outflow - inflow <= balance.
    for part in participants:
        touched = False
        for p, o in enumerate(obligations):
            coef = 0.0
            if o.payer == part.id:
                coef += amount[p]
            if o.payee == part.id:
                coef -= amount[p]
            if coef != 0.0:
                ridx.append(row)
                cidx.append(p)
                vals.append(coef)
                touched = True
        if touched:
            lb_c.append(-np.inf)
            ub_c.append(float(balance[part.id]))
            row += 1

    # (2) optional settled-value floor: sum amount_p x_p >= ratio * total.
    if min_settled_ratio is not None:
        for p in range(p_n):
            ridx.append(row)
            cidx.append(p)
            vals.append(float(amount[p]))
        lb_c.append(min_settled_ratio * total)
        ub_c.append(np.inf)
        row += 1

    if row == 0:  # no binding constraints at all → settle everything
        alloc = Allocation(x={o.id: 1.0 for o in obligations})
        return LinearSolveOutput(True, 0.0, alloc, "optimal")

    A = coo_matrix((vals, (ridx, cidx)), shape=(row, p_n)).tocsr()
    constraints = LinearConstraint(A, np.array(lb_c), np.array(ub_c))
    integrality = np.ones(p_n) if integer else np.zeros(p_n)
    bounds = Bounds(np.zeros(p_n), ub)

    res = milp(c=cost, constraints=constraints, integrality=integrality, bounds=bounds)
    if not res.success or res.x is None:
        return LinearSolveOutput(False, None, None, str(res.message))

    x = np.asarray(res.x, dtype=float)
    if integer:
        x = np.round(x)
    settled = float(amount @ x)
    alloc = Allocation(x={obligations[p].id: 1.0 for p in range(p_n) if x[p] > 0.5})
    return LinearSolveOutput(True, float(total - settled), alloc, "optimal")


# ---------------------------------------------------------------------------
# Constraint checking (solver-agnostic) — the exact, full check
# ---------------------------------------------------------------------------


def check_settlement_constraints(
    obligations: list[Obligation],
    participants: list[Participant],
    min_settled_ratio: float | None,
    allocation: Allocation,
) -> tuple[bool, float, list[ConstraintCheck]]:
    """Re-check any candidate batch against exact liquidity (+ optional floor).

    Returns (feasible, unsettled_value, checks).
    """
    settled_ids = {oid for oid, v in allocation.x.items() if v > 0.5}
    total = sum(o.amount for o in obligations)
    settled_value = sum(o.amount for o in obligations if o.id in settled_ids)

    outflow: dict[str, float] = {}
    inflow: dict[str, float] = {}
    for o in obligations:
        if o.id in settled_ids:
            outflow[o.payer] = outflow.get(o.payer, 0.0) + o.amount
            inflow[o.payee] = inflow.get(o.payee, 0.0) + o.amount

    checks: list[ConstraintCheck] = []

    # Liquidity: report the most-overdrawn participant.
    worst_over = -np.inf
    worst_id, worst_net, worst_bal = "", 0.0, 0.0
    for part in participants:
        net_out = outflow.get(part.id, 0.0) - inflow.get(part.id, 0.0)
        over = net_out - part.balance
        if over > worst_over:
            worst_over, worst_id, worst_net, worst_bal = over, part.id, net_out, part.balance
    checks.append(
        ConstraintCheck(
            name="liquidity",
            satisfied=worst_over <= _TOL,
            value=float(worst_net),
            limit=float(worst_bal),
            slack=float(worst_bal - worst_net),
            detail=f"most-stretched participant '{worst_id}': net outflow must be <= liquidity",
        )
    )

    # Optional settled-value floor.
    if min_settled_ratio is not None:
        ratio = settled_value / total if total > 0 else 0.0
        checks.append(
            ConstraintCheck(
                name="min_settled_ratio",
                satisfied=ratio >= min_settled_ratio - 1e-9,
                value=ratio,
                limit=min_settled_ratio,
                slack=ratio - min_settled_ratio,
                detail="settled value as a share of total queued value",
            )
        )

    feasible = all(c.satisfied for c in checks)
    return feasible, float(total - settled_value), checks


# ---------------------------------------------------------------------------
# Reduced instance + QUBO
# ---------------------------------------------------------------------------


@dataclass
class SettlementInstance:
    obligations: list[Obligation]
    participants: list[Participant]
    min_settled_ratio: float | None = None
    penalty_scale: float = 8.0
    slack_bits: int = SETTLEMENT_SLACK_BITS
    degenerate: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_qubits(self) -> int:
        # Upper bound: one bit per obligation + up to slack_bits per participant.
        return len(self.obligations) + len(self.participants) * self.slack_bits

    @property
    def target(self) -> float:
        return float(sum(o.amount for o in self.obligations))


def reduce_to_settlement_instance(spec: ProblemSpec, max_qubits: int) -> SettlementInstance:
    settlement = spec.settlement
    assert settlement is not None
    obs_all = spec.obligations
    part_by_id = {p.id: p for p in spec.participants}
    if not obs_all or not spec.participants:
        return SettlementInstance(
            obligations=[],
            participants=[],
            min_settled_ratio=settlement.min_settled_ratio,
            penalty_scale=settlement.penalty_scale,
            degenerate=True,
            provenance={"reason": "no obligations or participants to settle"},
        )

    # Admit as many high-value obligations as possible FIRST (they are the actual
    # decision variables), reserving at least one slack bit per involved
    # participant; then spend the leftover qubit budget on extra slack resolution.
    # More obligations beats finer slack — a richer instance is the point.
    ordered = sorted(obs_all, key=lambda o: (-o.amount, o.id))

    def admit(reserve_slack: int) -> tuple[list[Obligation], list[str]]:
        sel: list[Obligation] = []
        seen: set[str] = set()
        order: list[str] = []
        for o in ordered:
            parties = seen | {o.payer, o.payee}
            if (len(sel) + 1) + len(parties) * reserve_slack <= max_qubits:
                sel.append(o)
                for pid in (o.payer, o.payee):
                    if pid not in seen:
                        seen.add(pid)
                        order.append(pid)
        return sel, order

    selected, involved = admit(1)
    if not selected:  # not even one obligation fits with a slack bit → drop slack
        selected, involved = admit(0)
        slack_bits = 0
    else:
        k = len(involved)
        slack_bits = min(SETTLEMENT_SLACK_BITS, (max_qubits - len(selected)) // k) if k else 0
        slack_bits = max(0, slack_bits)

    if not selected:  # pragma: no cover — max_qubits >= 1 always fits one obligation
        return SettlementInstance(
            obligations=[],
            participants=[],
            min_settled_ratio=settlement.min_settled_ratio,
            penalty_scale=settlement.penalty_scale,
            degenerate=True,
            provenance={"reason": f"no obligation fits the {max_qubits}-qubit budget"},
        )

    participants = [part_by_id[pid] for pid in involved]
    return SettlementInstance(
        obligations=selected,
        participants=participants,
        min_settled_ratio=settlement.min_settled_ratio,
        penalty_scale=settlement.penalty_scale,
        slack_bits=slack_bits,
        provenance={
            "selection_rule": "largest obligations that fit the qubit budget",
            "obligations_selected": len(selected),
            "participants_involved": len(participants),
            "slack_bits_per_participant": slack_bits,
            "from_obligations": len(obs_all),
        },
    )


def build_settlement_qubo(instance: SettlementInstance) -> Qubo:
    """Encode 'settle as much value as possible without overdrawing anyone' as a QUBO.

    Objective (per obligation p): reward settling by ``-lambda * a~_p`` on the
    diagonal. Liquidity (per participant k): ``outflow_k - inflow_k <= b_k`` is a
    proper inequality encoded with binary slack bits ``s_k`` via the penalty
    ``A * (g_k + s_k - b~_k)^2``, where ``g_k`` is k's normalised net outflow. All
    amounts are non-dimensionalised so the energy landscape is O(1). The optional
    settled-value floor is NOT encoded (verification-only).
    """
    obs = instance.obligations
    parts = instance.participants
    p_n = len(obs)
    if p_n == 0:
        return Qubo(
            Q={},
            offset=0.0,
            n=0,
            num_decision=0,
            ids=[],
            encoding_losses=["Degenerate instance: no QUBO to build."],
        )

    amount = np.array([o.amount for o in obs], dtype=float)
    scale = float(amount.sum()) or 1.0
    a_t = amount / scale  # normalised amounts (sum to 1)
    b_t = {p.id: p.balance / scale for p in parts}
    lam = 1.0  # objective weight (settled value is in [0, 1])

    # Penalty weight. ``penalty_scale`` is always honoured as a FLOOR (a user who
    # raises it to strengthen the constraint is never silently overridden); the
    # auto term is capped to keep the QAOA landscape numerically sane.
    #
    # IMPORTANT (encoding loss): liquidity is a *packing* constraint. Settling an
    # obligation of amount ``a`` that overshoots a payer's remaining headroom by a
    # small ``δ`` gains reward ~``a`` but costs only the quadratic penalty
    # ``A·δ²``. Because ``δ`` can be arbitrarily small while ``a`` is large, NO
    # moderate ``A`` makes the penalty dominate for every breach — guaranteeing a
    # feasible ground state would need ``A ~ max_amount·total`` (which then swamps
    # the objective and cripples QAOA). So the QUBO ground state can be slightly
    # liquidity-infeasible; that is an inherent limitation of the penalty encoding
    # for this problem, and the Verification agent catches it EXACTLY on the
    # decoded batch (an infeasible batch is never reported feasible).
    min_step = float(a_t.min()) if p_n else 1.0
    auto = min(2.5 / (min_step**2), 1000.0) if min_step > 0 else 0.0
    A = max(instance.penalty_scale, auto)

    Q: dict[tuple[int, int], float] = {}

    def add(i: int, j: int, v: float) -> None:
        key = (i, j) if i <= j else (j, i)
        Q[key] = Q.get(key, 0.0) + v

    # Objective: maximise settled value.
    for p in range(p_n):
        add(p, p, -lam * float(a_t[p]))

    offset = 0.0
    n = p_n
    encode_liquidity = instance.slack_bits > 0
    if encode_liquidity:
        # Allocate per-participant slack blocks first (so indices are stable).
        slack_layout: dict[str, list[tuple[int, float]]] = {}
        for part in parts:
            inflow_max = float(sum(a_t[p] for p, o in enumerate(obs) if o.payee == part.id))
            s_max = b_t.get(part.id, 0.0) + inflow_max  # max unused-liquidity slack
            if s_max > 1e-12:
                g = s_max / (2**instance.slack_bits - 1)
                weights = [g * (2.0**j) for j in range(instance.slack_bits)]
            else:
                weights = []
            idxs = list(range(n, n + len(weights)))
            n += len(weights)
            slack_layout[part.id] = list(zip(idxs, weights, strict=True))

        for part in parts:
            # Coefficients of (g_k + s_k - b~_k) over the variables it touches.
            terms: list[tuple[int, float]] = []
            for p, o in enumerate(obs):
                coef = 0.0
                if o.payer == part.id:
                    coef += float(a_t[p])
                if o.payee == part.id:
                    coef -= float(a_t[p])
                if abs(coef) > 1e-15:
                    terms.append((p, coef))
            terms.extend(slack_layout[part.id])
            const = -b_t.get(part.id, 0.0)
            offset += A * const * const
            for vi, ci in terms:  # diagonal: v^2 == v
                add(vi, vi, A * (ci * ci + 2.0 * const * ci))
            for a_ix in range(len(terms)):
                for b_ix in range(a_ix + 1, len(terms)):
                    vi, ci = terms[a_ix]
                    vj, cj = terms[b_ix]
                    add(vi, vj, A * 2.0 * ci * cj)

    if encode_liquidity:
        liq_note = (
            f"Each participant's liquidity inequality (net outflow <= balance) is encoded "
            f"with {instance.slack_bits} binary slack bits as a soft quadratic penalty. Liquidity "
            "is a PACKING constraint, so a small overshoot (settling an obligation that exceeds a "
            "payer's remaining headroom by a little) incurs only a small quadratic penalty while "
            "still reaping the full settled-value reward — no tractable penalty weight fully "
            "prevents this. The QUBO ground state is therefore only APPROXIMATELY "
            "liquidity-feasible, so the decoded batch is ALWAYS re-checked against exact liquidity "
            "by the Verification agent — an infeasible batch is caught there and never reported "
            "feasible. (This lossy encoding is itself concrete evidence that QUBO/QAOA is a poor "
            "fit for constrained settlement, where the exact classical MILP is strongly preferred.)"
        )
    else:
        liq_note = (
            "Liquidity constraints are NOT encoded (no slack-bit budget at this qubit count); "
            "the QUBO only maximises settled value and the Verification agent re-checks "
            "liquidity on the decoded batch."
        )
    losses = [
        "Continuous settle fractions x in [0,1] hardened to binary x in {0,1} (settle a whole "
        "obligation or none).",
        liq_note,
        "The optional settled-value floor is NOT encoded in the QUBO; the Verification agent "
        "re-checks it on the decoded batch.",
    ]
    return Qubo(
        Q=Q,
        offset=offset,
        n=n,
        num_decision=p_n,
        ids=[o.id for o in obs],
        encoding_losses=losses,
        slack_bits=n - p_n,
    )


def bits_to_settlement_allocation(
    instance: SettlementInstance, bits: NDArray[Any] | list[int]
) -> Allocation:
    """Decode the leading obligation bits (slack bits, if any, are ignored)."""
    obs = instance.obligations
    return Allocation(x={obs[p].id: 1.0 for p in range(len(obs)) if int(bits[p]) == 1})


# ---------------------------------------------------------------------------
# Domain adapter
# ---------------------------------------------------------------------------


class SettlementDomain(ProblemDomain):
    problem = "settlement_netting"

    def requirements(self, spec: ProblemSpec) -> RequirementsReport:
        settlement = spec.settlement
        assert settlement is not None
        obs, parts = spec.obligations, spec.participants
        total = sum(o.amount for o in obs)
        liquidity = sum(p.balance for p in parts)

        # Can everyone settle in isolation? (Gridlock is when the naive answer is no
        # but a simultaneous batch resolves it.) Flag participants who could not
        # fund all their outgoing obligations even with zero incoming.
        out_by: dict[str, float] = {}
        for o in obs:
            out_by[o.payer] = out_by.get(o.payer, 0.0) + o.amount
        bal_by = {p.id: p.balance for p in parts}
        illiquid = [pid for pid, out in out_by.items() if out > bal_by.get(pid, 0.0) + 1e-9]

        gaps: list[str] = []
        if not obs:
            gaps.append("No obligations queued to settle.")
        if liquidity <= 0:
            gaps.append("No participant has any liquidity — only self-funding batches can settle.")
        if not illiquid:
            gaps.append(
                "Every participant can fund all their outgoing obligations directly — "
                "there is no gridlock to resolve; the batch trivially settles everything."
            )
        return RequirementsReport(
            problem=self.problem,
            summary=(
                f"{len(obs)} obligations among {len(parts)} participants, "
                f"total value {total:,.0f}, liquidity {liquidity:,.0f}"
            ),
            metrics={
                "n_obligations": float(len(obs)),
                "n_participants": float(len(parts)),
                "total_value": float(total),
                "total_liquidity": float(liquidity),
                "n_potentially_gridlocked": float(len(illiquid)),
            },
            feasible_upper_bound=bool(obs) and bool(parts),
            discovered_gaps=gaps,
            assumptions=[
                "A settled obligation moves its full amount at once (no partial settlement).",
                "Inflows within the batch count toward funding a participant's outflows.",
            ],
            autonomy_level=spec.execution_policy.autonomy_level.value,
        )

    def formulations(self, spec: ProblemSpec) -> FormulationCatalogue:
        p_n, k_n = len(spec.obligations), len(spec.participants)
        return FormulationCatalogue(
            catalogue=[
                Formulation(
                    name="settlement_lp",
                    kind="Linear Program",
                    variables=f"{p_n} continuous x_p in [0,1]",
                    represents="settled value + liquidity + optional settled-value floor",
                    note="LP relaxation; an upper bound on settleable value.",
                ),
                Formulation(
                    name="settlement_milp",
                    kind="Mixed-Integer Linear Program",
                    variables=f"{p_n} binary x_p, {k_n} liquidity rows",
                    represents="settled value + all constraints exactly",
                    note="The FAIR classical comparator (exact liquidity-saving batch).",
                ),
                Formulation(
                    name="qubo",
                    kind="Quadratic Unconstrained Binary Optimisation",
                    variables="binary x_p + per-participant slack bits on a reduced instance",
                    represents="settled value + liquidity (slack-encoded inequality)",
                    note="Liquidity IS in the QUBO; the settled-value floor is verification-only.",
                ),
                Formulation(
                    name="ising",
                    kind="Ising Hamiltonian",
                    variables="spins z, x = (1 - z)/2",
                    represents="same as the QUBO",
                    note="Direct input to gate-model QAOA and to annealers.",
                ),
            ],
            selected_classical="settlement_milp",
            selected_quantum_path="qubo -> ising -> QAOA",
            encoding_loss_note=(
                "The quantum path solves a reduced settlement instance; the decoded batch "
                "is re-checked against the full liquidity constraints and the value floor."
            ),
        )

    def solve_classical_full(self, spec: ProblemSpec) -> ClassicalBaseline:
        import time

        settlement = spec.settlement
        assert settlement is not None
        ratio = settlement.min_settled_ratio
        t0 = time.perf_counter()
        lp = solve_settlement_milp(spec.obligations, spec.participants, ratio, integer=False)
        lp_dt = time.perf_counter() - t0
        t0 = time.perf_counter()
        mp = solve_settlement_milp(spec.obligations, spec.participants, ratio, integer=True)
        mp_dt = time.perf_counter() - t0

        lp_res = SolveResult(
            method="classical_lp_relaxation",
            kind="classical",
            backend="scipy/HiGHS",
            scope="full_problem",
            feasible=lp.feasible,
            objective=lp.objective,
            allocation=lp.allocation or Allocation(x={}),
            runtime_s=lp_dt,
            metadata={"role": "upper bound on settleable value", "status": lp.status},
        )
        n_settled = len(mp.allocation.x) if mp.allocation is not None else 0
        milp_res = SolveResult(
            method="classical_milp",
            kind="classical",
            backend="scipy/HiGHS",
            scope="full_problem",
            feasible=mp.feasible,
            objective=mp.objective,
            allocation=mp.allocation or Allocation(x={}),
            runtime_s=mp_dt,
            metadata={
                "role": "fair binary comparator / production recommendation",
                "status": mp.status,
                "n_settled": n_settled,
            },
        )
        gap = (
            mp.objective - lp.objective
            if (
                mp.feasible
                and lp.feasible
                and mp.objective is not None
                and lp.objective is not None
            )
            else None
        )
        return ClassicalBaseline(milp=milp_res, lp=lp_res, integrality_gap=gap)

    def reduce_to_instance(self, spec: ProblemSpec, max_qubits: int) -> SettlementInstance:
        return reduce_to_settlement_instance(spec, max_qubits)

    def build_qubo(self, instance: SettlementInstance, *, slack_bits: int) -> Qubo:  # type: ignore[override]
        # Per-participant slack is fixed by reduce_to_instance to fit the qubit
        # budget; the single planner-provided slack_bits does not apply here.
        return build_settlement_qubo(instance)

    def solve_instance_classical(self, instance: SettlementInstance) -> SolveResult:  # type: ignore[override]
        import time

        t0 = time.perf_counter()
        out = solve_settlement_milp(
            instance.obligations, instance.participants, instance.min_settled_ratio, integer=True
        )
        dt = time.perf_counter() - t0
        if out.feasible and out.allocation is not None:
            feasible, obj, _ = check_settlement_constraints(
                instance.obligations,
                instance.participants,
                instance.min_settled_ratio,
                out.allocation,
            )
            return SolveResult(
                method="instance_milp",
                kind="classical",
                backend="scipy/HiGHS",
                scope="research_instance",
                feasible=feasible,
                objective=obj,
                allocation=out.allocation,
                runtime_s=dt,
            )
        return SolveResult(
            method="instance_milp",
            kind="classical",
            backend="scipy/HiGHS",
            scope="research_instance",
            feasible=False,
            objective=None,
            allocation=Allocation(x={}),
            runtime_s=dt,
            metadata={"status": out.status},
        )

    def evaluate_bits(  # type: ignore[override]
        self,
        instance: SettlementInstance,
        bits: NDArray[Any] | list[int],
        *,
        method: str,
        kind: str,
        backend: str,
        runtime_s: float = 0.0,
        qpu_time_s: float = 0.0,
        cost_usd: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> SolveResult:
        alloc = bits_to_settlement_allocation(instance, bits)
        feasible, obj, _ = check_settlement_constraints(
            instance.obligations, instance.participants, instance.min_settled_ratio, alloc
        )
        return SolveResult(
            method=method,
            kind=kind,
            backend=backend,
            scope="research_instance",
            feasible=feasible,
            objective=obj,
            allocation=alloc,
            runtime_s=runtime_s,
            qpu_time_s=qpu_time_s,
            cost_usd=cost_usd,
            metadata=metadata or {},
        )

    def _verify(
        self,
        result: SolveResult,
        obligations: list[Obligation],
        participants: list[Participant],
        min_settled_ratio: float | None,
    ) -> VerificationReport:
        if result.allocation is None:
            return VerificationReport(
                method=result.method,
                scope=result.scope,
                feasible=False,
                recomputed_objective=None,
                objective_matches_solver=(result.objective is None),
                notes=["No allocation to verify."],
            )
        feasible, obj, checks = check_settlement_constraints(
            obligations, participants, min_settled_ratio, result.allocation
        )
        matches = result.objective is None or abs(obj - result.objective) <= 1e-6 * max(
            1.0, abs(obj)
        )
        notes = [] if matches else [f"Solver objective {result.objective} != recomputed {obj:.4f}."]
        return VerificationReport(
            method=result.method,
            scope=result.scope,
            feasible=feasible,
            recomputed_objective=obj,
            objective_matches_solver=matches,
            checks=checks,
            notes=notes,
        )

    def verify_full(self, spec: ProblemSpec, result: SolveResult) -> VerificationReport:
        settlement = spec.settlement
        assert settlement is not None
        return self._verify(
            result, spec.obligations, spec.participants, settlement.min_settled_ratio
        )

    def verify_instance(  # type: ignore[override]
        self, instance: SettlementInstance, result: SolveResult
    ) -> VerificationReport:
        return self._verify(
            result, instance.obligations, instance.participants, instance.min_settled_ratio
        )

    def instance_warm_start(  # type: ignore[override]
        self, instance: SettlementInstance, qubo: Qubo
    ) -> list[float] | None:
        """Warm-start biases from the instance LP relaxation (slack bits → 0.5)."""
        if qubo.n == 0:
            return None
        lp = solve_settlement_milp(
            instance.obligations, instance.participants, instance.min_settled_ratio, integer=False
        )
        if not lp.feasible or lp.allocation is None:
            return None
        biases = [float(lp.allocation.x.get(o.id, 0.0)) for o in instance.obligations]
        biases += [0.5] * (qubo.n - len(biases))
        return biases
