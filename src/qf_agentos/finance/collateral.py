"""Collateral-allocation problem: formulations, reductions, and checks.

The *same* economic problem is expressed several ways so the pipeline can compare
like with like:

* Continuous LP relaxation  — the theoretical best (a lower bound on cost).
* Binary MILP               — post-a-lot-or-not; the *fair* classical comparator
                              for anything a QUBO/QAOA can express.
* QUBO / Ising              — a small, reduced "research instance" the quantum
                              backend can actually run.

The QUBO encodes the coverage requirement as a proper inequality (``cov >= R``)
via binary *slack* variables, so its ground state is genuinely feasible on
coverage rather than being forced to an exact-equality point. Concentration and
HQLA remain verification-only constraints.

Crucially, :func:`check_constraints` evaluates ANY candidate allocation against
the full, exact constraint set — including constraints the QUBO never encoded —
so a relaxation artifact is caught, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ..core.ir import ProblemSpec, Security
from ..core.result import Allocation, ConstraintCheck

# Default number of binary slack bits used to encode the coverage inequality.
SLACK_BITS_DEFAULT = 4

# ---------------------------------------------------------------------------
# Array views
# ---------------------------------------------------------------------------


@dataclass
class Arrays:
    coverage: NDArray[np.float64]  # a_i  = (1-h_i) v_i
    cost: NDArray[np.float64]  # c_i   = (cost_bps/1e4) v_i
    hqla: NDArray[np.float64]  # 1.0 if HQLA else 0.0
    groups: dict[str, dict[str, list[int]]]  # attr -> group value -> row indices
    ids: list[str]


def build_arrays(securities: list[Security], concentration_attrs: list[str]) -> Arrays:
    coverage = np.array([s.coverage for s in securities], dtype=float)
    cost = np.array([s.cost for s in securities], dtype=float)
    hqla = np.array([1.0 if s.hqla else 0.0 for s in securities], dtype=float)
    groups: dict[str, dict[str, list[int]]] = {}
    for attr in concentration_attrs:
        gmap: dict[str, list[int]] = {}
        for i, s in enumerate(securities):
            gmap.setdefault(str(getattr(s, attr)), []).append(i)
        groups[attr] = gmap
    return Arrays(coverage, cost, hqla, groups, [s.id for s in securities])


def _efficiency(s: Security) -> float:
    """Coverage per unit posting cost. Scale-free; cost 0 ⇒ maximally efficient."""
    return float("inf") if s.cost <= 0 else s.coverage / s.cost


# ---------------------------------------------------------------------------
# Classical solvers (real HiGHS via scipy.optimize.milp)
# ---------------------------------------------------------------------------


@dataclass
class LinearSolveOutput:
    feasible: bool
    objective: float | None
    allocation: Allocation | None
    status: str


def _linear_solve(
    securities: list[Security],
    required_collateral: float,
    minimum_hqla: float,
    concentration: dict[str, float],
    *,
    integer: bool,
) -> LinearSolveOutput:
    """Minimise posting cost s.t. coverage / HQLA / concentration constraints.

    ``integer=True`` restricts each x_i to {0,1} (MILP); otherwise x_i in [0,1] (LP).
    """
    n = len(securities)
    if n == 0:
        return LinearSolveOutput(False, None, None, "empty inventory")

    arr = build_arrays(securities, list(concentration.keys()))
    a, c = arr.coverage, arr.cost

    # An auxiliary continuous variable T = total posted coverage (index n) keeps
    # the concentration constraints SPARSE: each row touches only its group's
    # columns plus T, instead of every column. The whole matrix is O(n) nonzeros,
    # not O(n_groups x n) dense — critical for large inventories.
    num_vars = n + 1
    t = n
    ridx: list[int] = []
    cidx: list[int] = []
    vals: list[float] = []
    lb: list[float] = []
    ub: list[float] = []
    row = 0

    # (1) definition: sum_i a_i x_i - T = 0
    for i in range(n):
        ridx.append(row)
        cidx.append(i)
        vals.append(float(a[i]))
    ridx.append(row)
    cidx.append(t)
    vals.append(-1.0)
    lb.append(0.0)
    ub.append(0.0)
    row += 1

    # (2) coverage: T >= required
    ridx.append(row)
    cidx.append(t)
    vals.append(1.0)
    lb.append(required_collateral)
    ub.append(np.inf)
    row += 1

    # (3) minimum HQLA: sum_{i: hqla} a_i x_i >= minimum_hqla
    if minimum_hqla > 0:
        for i in range(n):
            if arr.hqla[i]:
                ridx.append(row)
                cidx.append(i)
                vals.append(float(a[i]))
        lb.append(minimum_hqla)
        ub.append(np.inf)
        row += 1

    # (4) concentration: for each group g, sum_{i in g} a_i x_i - frac*T <= 0
    for attr, frac in concentration.items():
        for _val, idx in arr.groups[attr].items():
            for i in idx:
                ridx.append(row)
                cidx.append(i)
                vals.append(float(a[i]))
            ridx.append(row)
            cidx.append(t)
            vals.append(-float(frac))
            lb.append(-np.inf)
            ub.append(0.0)
            row += 1

    A = coo_matrix((vals, (ridx, cidx)), shape=(row, num_vars)).tocsr()
    constraints = LinearConstraint(A, np.array(lb), np.array(ub))

    integrality = np.zeros(num_vars)
    if integer:
        integrality[:n] = 1
    lb_v = np.zeros(num_vars)
    ub_v = np.ones(num_vars)
    ub_v[t] = float(a.sum())  # T in [0, total reachable coverage]
    bounds = Bounds(lb_v, ub_v)
    cost_vec = np.concatenate([c, [0.0]])

    res = milp(c=cost_vec, constraints=constraints, integrality=integrality, bounds=bounds)
    if not res.success or res.x is None:
        return LinearSolveOutput(False, None, None, str(res.message))

    x = np.asarray(res.x[:n], dtype=float)
    if integer:
        x = np.round(x)
    alloc = Allocation(x={arr.ids[i]: float(x[i]) for i in range(n) if x[i] > 1e-9})
    return LinearSolveOutput(True, float(c @ x), alloc, "optimal")


def solve_lp_relaxation(spec: ProblemSpec) -> LinearSolveOutput:
    cn = spec.constraints
    return _linear_solve(
        spec.eligible_inventory,
        cn.required_collateral,
        cn.minimum_hqla,
        cn.concentration,
        integer=False,
    )


def solve_binary_milp(spec: ProblemSpec) -> LinearSolveOutput:
    cn = spec.constraints
    return _linear_solve(
        spec.eligible_inventory,
        cn.required_collateral,
        cn.minimum_hqla,
        cn.concentration,
        integer=True,
    )


def solve_instance_milp(instance: ResearchInstance) -> LinearSolveOutput:
    """Exact binary optimum of the research instance under ALL its constraints —
    the fair classical comparator the QUBO/QAOA result is judged against."""
    return _linear_solve(
        instance.securities,
        instance.required_collateral,
        instance.minimum_hqla,
        instance.concentration,
        integer=True,
    )


# ---------------------------------------------------------------------------
# Reduction to a quantum-sized research instance
# ---------------------------------------------------------------------------


@dataclass
class ResearchInstance:
    """A small, self-contained sub-problem the quantum backend can run."""

    securities: list[Security]
    required_collateral: float  # instance-local coverage target R'
    minimum_hqla: float
    concentration: dict[str, float]
    penalty_scale: float = 8.0
    degenerate: bool = False
    provenance: dict[str, object] = field(default_factory=dict)

    @property
    def n_qubits(self) -> int:
        """Decision qubits (one per selected security), excluding slack bits."""
        return len(self.securities)


def reduce_to_instance(
    spec: ProblemSpec, max_qubits: int, *, slack_bits: int = SLACK_BITS_DEFAULT
) -> ResearchInstance:
    """Select the most coverage-efficient securities into a research instance that
    fits ``max_qubits`` (reserving room for slack bits).

    The instance target R' is a non-trivial fraction of the instance's own
    reachable coverage, so the QUBO is a genuine subset-selection problem rather
    than "post everything".
    """
    elig = spec.eligible_inventory
    if not elig:
        return ResearchInstance(
            securities=[],
            required_collateral=0.0,
            minimum_hqla=0.0,
            concentration=dict(spec.constraints.concentration),
            degenerate=True,
            provenance={"reason": "no eligible securities in inventory"},
        )

    # Leave room for slack bits within the qubit budget.
    n_sec = max(1, min(len(elig), max_qubits - max(0, slack_bits)))
    scored = sorted(elig, key=lambda s: (_efficiency(s), s.coverage), reverse=True)
    selected = scored[:n_sec]

    total_cov = sum(s.coverage for s in selected)
    if total_cov <= 0:
        return ResearchInstance(
            securities=selected,
            required_collateral=0.0,
            minimum_hqla=0.0,
            concentration=dict(spec.constraints.concentration),
            degenerate=True,
            provenance={"reason": "selected securities have zero coverage"},
        )

    target = max(1.0, round(0.6 * total_cov))
    has_hqla = any(s.hqla for s in selected)
    inst_min_hqla = (
        round(0.25 * target) if (has_hqla and spec.constraints.minimum_hqla > 0) else 0.0
    )

    return ResearchInstance(
        securities=selected,
        required_collateral=float(target),
        minimum_hqla=float(inst_min_hqla),
        concentration=dict(spec.constraints.concentration),
        provenance={
            "selection_rule": "coverage_per_unit_cost desc",
            "selected_from": len(elig),
            "instance_reachable_coverage": total_cov,
            "target_fraction_of_reachable": 0.6,
            "slack_bits_reserved": max(0, slack_bits),
        },
    )


# ---------------------------------------------------------------------------
# QUBO / Ising for the research instance
# ---------------------------------------------------------------------------


@dataclass
class Qubo:
    """Normalised QUBO over ``n`` binary variables (``num_decision`` securities +
    slack bits). ``Q`` is upper-triangular (i<=j). Energy is used only to rank
    bitstrings; true financial cost is always recomputed via check_constraints."""

    Q: dict[tuple[int, int], float]
    offset: float
    n: int  # total variables (decision + slack)
    num_decision: int  # security-decision variables (first `num_decision` bits)
    ids: list[str]  # security ids, length == num_decision
    encoding_losses: list[str] = field(default_factory=list)
    slack_bits: int = 0


def build_qubo(instance: ResearchInstance, *, slack_bits: int = SLACK_BITS_DEFAULT) -> Qubo:
    """Encode 'reach collateral target R' at minimum cost' as a QUBO.

    Coverage is a proper inequality ``cov >= R'`` encoded with binary slack bits
    (``cov - R' = sum_k w_k y_k``, y in {0,1}); cost is linear on the security
    bits. Concentration/HQLA are dropped and become verification-only constraints.
    All quantities are non-dimensionalised so QAOA/SA operate on O(1) numbers.
    """
    secs = instance.securities
    n = len(secs)
    R = instance.required_collateral
    if n == 0 or R <= 0:
        return Qubo(
            Q={},
            offset=0.0,
            n=0,
            num_decision=0,
            ids=[],
            encoding_losses=["Degenerate instance: no QUBO to build."],
        )

    a = np.array([s.coverage for s in secs], dtype=float)
    c = np.array([s.cost for s in secs], dtype=float)
    cost_scale = float(c.sum()) or 1.0
    a_t = a / R  # normalised coverage; target becomes 1.0
    c_t = c / cost_scale  # normalised cost (sums to 1)

    # Penalty must dominate cost so the QUBO optimum respects coverage. The
    # smallest coverage step is min(a_t); a shortfall of that size must cost more
    # in penalty than the entire normalised cost budget (which sums to 1).
    min_step = float(a_t.min()) if len(a_t) else 1.0
    A = max(instance.penalty_scale, 2.5 / (min_step**2)) if min_step > 0 else instance.penalty_scale
    A = min(A, 1000.0)  # keep the energy landscape sane for QAOA

    # Slack bits to represent surplus (cov - R') >= 0, up to the reachable maximum.
    s_max = max(0.0, float(a_t.sum()) - 1.0)
    k = slack_bits if s_max > 1e-9 else 0
    if k > 0:
        g = s_max / (2**k - 1)
        weights = [g * (2.0**b) for b in range(k)]
    else:
        weights = []

    # Combined coefficient vector p over [securities..., slack...]:
    #   penalty = A * ( sum_i a_t_i x_i  -  sum_b w_b y_b  -  1 )^2
    p = np.concatenate([a_t, -np.array(weights, dtype=float)]) if weights else a_t
    total = n + k

    Q: dict[tuple[int, int], float] = {}
    for i in range(total):
        Q[(i, i)] = float(A * (p[i] * p[i] - 2.0 * p[i]))
    for i in range(total):
        for j in range(i + 1, total):
            Q[(i, j)] = float(A * 2.0 * p[i] * p[j])
    for i in range(n):  # linear posting cost on securities only
        Q[(i, i)] += float(c_t[i])
    offset = float(A * 1.0)

    if k > 0:
        cov_note = (
            f"Coverage inequality (cov >= R') encoded with {k} binary slack bits; "
            "the QUBO ground state respects the coverage requirement."
        )
    else:
        cov_note = (
            "Coverage encoded as an exact-equality penalty (no slack-bit budget): "
            "over-collateralisation is penalised even though it is feasible."
        )
    losses = [
        "Continuous posting fractions x in [0,1] hardened to binary x in {0,1} (post whole lot).",
        cov_note,
        f"Concentration caps {instance.concentration or '{}'} and minimum HQLA "
        f"({instance.minimum_hqla:.0f}) are NOT encoded in the QUBO; the Verification agent "
        "re-checks the decoded solution against them.",
    ]
    return Qubo(
        Q=Q,
        offset=offset,
        n=total,
        num_decision=n,
        ids=[s.id for s in secs],
        encoding_losses=losses,
        slack_bits=k,
    )


def qubo_energy(qubo: Qubo, bits: NDArray[Any] | list[int]) -> float:
    b = np.asarray(bits, dtype=float)
    e = qubo.offset
    for (i, j), coeff in qubo.Q.items():
        e += coeff * (b[i] if i == j else b[i] * b[j])
    return float(e)


def qubo_to_ising(
    qubo: Qubo,
) -> tuple[float, NDArray[np.float64], dict[tuple[int, int], float]]:
    """Map QUBO (over x in {0,1}) to Ising (over z in {+1,-1}) via x = (1 - z)/2.

    Returns (constant, h, J) where H = constant*I + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j.
    """
    n = qubo.n
    const = qubo.offset
    h = np.zeros(n)
    J: dict[tuple[int, int], float] = {}
    for (i, j), coeff in qubo.Q.items():
        if i == j:
            const += 0.5 * coeff
            h[i] += -0.5 * coeff
        else:
            const += 0.25 * coeff
            h[i] += -0.25 * coeff
            h[j] += -0.25 * coeff
            J[(i, j)] = J.get((i, j), 0.0) + 0.25 * coeff
    return const, h, J


# ---------------------------------------------------------------------------
# Decoding + full constraint checking
# ---------------------------------------------------------------------------


def bits_to_allocation(ids: list[str], bits: NDArray[Any] | list[int]) -> Allocation:
    """Decode the leading security bits (slack bits, if any, are ignored)."""
    return Allocation(x={ids[i]: 1.0 for i in range(len(ids)) if int(bits[i]) == 1})


def check_constraints(
    securities: list[Security],
    allocation: Allocation,
    required_collateral: float,
    minimum_hqla: float,
    concentration: dict[str, float],
    *,
    rel_tol: float = 1e-6,
) -> tuple[bool, float, list[ConstraintCheck]]:
    """Evaluate an allocation against the exact constraint set. Solver-agnostic."""
    by_id = {s.id: s for s in securities}
    x = {sid: allocation.x.get(sid, 0.0) for sid in by_id}

    tol = rel_tol * max(1.0, required_collateral)
    posted_cov = {sid: by_id[sid].coverage * x[sid] for sid in by_id}
    total_cov = sum(posted_cov.values())
    total_cost = sum(by_id[sid].cost * x[sid] for sid in by_id)

    checks: list[ConstraintCheck] = []

    # Coverage
    checks.append(
        ConstraintCheck(
            name="coverage",
            satisfied=total_cov >= required_collateral - tol,
            value=total_cov,
            limit=required_collateral,
            slack=total_cov - required_collateral,
            detail="post-haircut collateral value must meet the requirement",
        )
    )

    # Minimum HQLA
    if minimum_hqla > 0:
        hqla_cov = sum(posted_cov[sid] for sid in by_id if by_id[sid].hqla)
        checks.append(
            ConstraintCheck(
                name="minimum_hqla",
                satisfied=hqla_cov >= minimum_hqla - tol,
                value=hqla_cov,
                limit=minimum_hqla,
                slack=hqla_cov - minimum_hqla,
                detail="post-haircut HQLA value within the posted pool",
            )
        )

    # Concentration (only meaningful when something is posted)
    for attr, frac in concentration.items():
        groups: dict[str, float] = {}
        for sid in by_id:
            g = str(getattr(by_id[sid], attr))
            groups[g] = groups.get(g, 0.0) + posted_cov[sid]
        worst_group = max(groups, key=lambda g: groups[g]) if groups else ""
        worst_val = groups.get(worst_group, 0.0)
        limit = frac * total_cov
        satisfied = total_cov <= tol or worst_val <= limit + tol
        checks.append(
            ConstraintCheck(
                name=f"concentration[{attr}]",
                satisfied=satisfied,
                value=worst_val,
                limit=limit,
                slack=limit - worst_val,
                detail=f"largest '{attr}' group = '{worst_group}' must be <= {frac:.0%} of pool",
            )
        )

    feasible = all(c.satisfied for c in checks)
    return feasible, float(total_cost), checks
