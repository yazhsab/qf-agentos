"""Payment-routing optimisation: a generalized assignment problem (GAP).

Route each transaction to exactly one eligible payment route (acquirer / network
/ processor path) to minimise total expected cost:

    processing fee + fixed fee + expected fraud loss + latency cost
    + expected decline cost  ( (1 - approval_rate) * decline_penalty )

subject to per-route capacity, optional per-network diversification, and an
optional portfolio approval floor.

This plugs into the same agent pipeline as collateral allocation via the
:class:`ProblemDomain` interface. The QUBO encodes only the one-hot assignment
(each transaction to exactly one route) + cost; capacity / network / approval are
verification-only constraints re-checked on the decoded solution.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ..core.artifacts import Formulation, FormulationCatalogue, RequirementsReport
from ..core.domain import ClassicalBaseline, ProblemDomain
from ..core.ir import PaymentRoute, ProblemSpec, RoutingConstraints, Transaction
from ..core.result import Allocation, ConstraintCheck, SolveResult, VerificationReport
from .collateral import LinearSolveOutput, Qubo

_KEY = "=>"  # allocation key format: "<tx_id>=><route_id>"
ROUTES_PER_INSTANCE = 3


# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------


def is_eligible(tx: Transaction, route_id: str) -> bool:
    return not tx.eligible_routes or route_id in tx.eligible_routes


def assignment_cost(tx: Transaction, route: PaymentRoute, routing: RoutingConstraints) -> float:
    """Expected total cost of routing ``tx`` via ``route`` (currency)."""
    amt = tx.amount
    processing = (route.cost_bps / 10_000.0) * amt + route.fixed_fee
    fraud = (route.fraud_bps / 10_000.0) * amt
    latency = routing.latency_weight * route.latency_ms
    decline = (1.0 - route.approval_rate) * (routing.decline_penalty_bps / 10_000.0) * amt
    return processing + fraud + latency + decline


# ---------------------------------------------------------------------------
# MILP (real HiGHS via scipy.optimize.milp), sparse
# ---------------------------------------------------------------------------


def solve_routing_milp(
    transactions: list[Transaction],
    routes: list[PaymentRoute],
    routing: RoutingConstraints,
    *,
    integer: bool,
) -> LinearSolveOutput:
    """Assign every transaction to exactly one eligible route at minimum cost."""
    t_n, r_n = len(transactions), len(routes)
    if t_n == 0:
        return LinearSolveOutput(True, 0.0, Allocation(x={}), "no transactions")
    if r_n == 0:
        return LinearSolveOutput(False, None, None, "no routes")

    def idx(t: int, r: int) -> int:
        return t * r_n + r

    n = t_n * r_n
    cost = np.zeros(n)
    ub = np.ones(n)
    for t, tx in enumerate(transactions):
        for r, route in enumerate(routes):
            cost[idx(t, r)] = assignment_cost(tx, route, routing)
            if not is_eligible(tx, route.id):
                ub[idx(t, r)] = 0.0  # forbid ineligible assignments

    ridx: list[int] = []
    cidx: list[int] = []
    vals: list[float] = []
    lb_c: list[float] = []
    ub_c: list[float] = []
    row = 0

    # (1) assignment: each transaction exactly one route
    for t in range(t_n):
        for r in range(r_n):
            ridx.append(row)
            cidx.append(idx(t, r))
            vals.append(1.0)
        lb_c.append(1.0)
        ub_c.append(1.0)
        row += 1

    # (2) capacity: per route, count <= capacity
    for r, route in enumerate(routes):
        for t in range(t_n):
            ridx.append(row)
            cidx.append(idx(t, r))
            vals.append(1.0)
        lb_c.append(-np.inf)
        ub_c.append(float(route.capacity))
        row += 1

    # (3) network concentration (optional): per network, count <= frac * T
    if routing.network_concentration is not None:
        nets: dict[str, list[int]] = {}
        for r, route in enumerate(routes):
            nets.setdefault(route.network, []).append(r)
        for _net, rs in nets.items():
            for r in rs:
                for t in range(t_n):
                    ridx.append(row)
                    cidx.append(idx(t, r))
                    vals.append(1.0)
            lb_c.append(-np.inf)
            ub_c.append(routing.network_concentration * t_n)
            row += 1

    # (4) portfolio approval floor (optional): sum approval_r x >= floor * T
    if routing.min_overall_approval is not None:
        for t in range(t_n):
            for r, route in enumerate(routes):
                ridx.append(row)
                cidx.append(idx(t, r))
                vals.append(route.approval_rate)
        lb_c.append(routing.min_overall_approval * t_n)
        ub_c.append(np.inf)
        row += 1

    A = coo_matrix((vals, (ridx, cidx)), shape=(row, n)).tocsr()
    constraints = LinearConstraint(A, np.array(lb_c), np.array(ub_c))
    integrality = np.ones(n) if integer else np.zeros(n)
    bounds = Bounds(np.zeros(n), ub)

    res = milp(c=cost, constraints=constraints, integrality=integrality, bounds=bounds)
    if not res.success or res.x is None:
        return LinearSolveOutput(False, None, None, str(res.message))

    x = np.asarray(res.x, dtype=float)
    if integer:
        x = np.round(x)
    alloc = Allocation(
        x={
            f"{transactions[t].id}{_KEY}{routes[r].id}": 1.0
            for t in range(t_n)
            for r in range(r_n)
            if x[idx(t, r)] > 0.5
        }
    )
    return LinearSolveOutput(True, float(cost @ x), alloc, "optimal")


# ---------------------------------------------------------------------------
# Constraint checking (solver-agnostic)
# ---------------------------------------------------------------------------


def check_routing_constraints(
    transactions: list[Transaction],
    routes: list[PaymentRoute],
    routing: RoutingConstraints,
    allocation: Allocation,
) -> tuple[bool, float, list[ConstraintCheck]]:
    route_by_id = {r.id: r for r in routes}
    tx_by_id = {t.id: t for t in transactions}
    by_tx: dict[str, list[str]] = {}
    for key in allocation.x:
        tx_id, route_id = key.split(_KEY, 1)
        by_tx.setdefault(tx_id, []).append(route_id)

    total_cost = 0.0
    n_valid = 0
    route_counts: Counter[str] = Counter()
    approvals: list[float] = []
    for tx in transactions:
        assigned = by_tx.get(tx.id, [])
        for rid in assigned:
            route = route_by_id.get(rid)
            if route is not None:
                total_cost += assignment_cost(tx, route, routing)
                route_counts[rid] += 1
                approvals.append(route.approval_rate)
        if len(assigned) == 1 and assigned[0] in route_by_id and is_eligible(tx, assigned[0]):
            n_valid += 1

    t_n = len(transactions)
    checks: list[ConstraintCheck] = []

    # Assignment validity (one eligible route per transaction).
    checks.append(
        ConstraintCheck(
            name="assignment",
            satisfied=n_valid == t_n,
            value=float(n_valid),
            limit=float(t_n),
            slack=float(n_valid - t_n),
            detail="each transaction must be assigned exactly one eligible route",
        )
    )

    # Capacity (worst-loaded route).
    worst_over, worst_route, worst_count, worst_cap = 0.0, "", 0, 0
    for route in routes:
        cnt = route_counts.get(route.id, 0)
        over = cnt - route.capacity
        if over > worst_over or worst_route == "":
            worst_over, worst_route, worst_count, worst_cap = over, route.id, cnt, route.capacity
    checks.append(
        ConstraintCheck(
            name="capacity",
            satisfied=worst_over <= 0,
            value=float(worst_count),
            limit=float(worst_cap),
            slack=float(worst_cap - worst_count),
            detail=f"busiest route '{worst_route}' must stay within capacity",
        )
    )

    # Network diversification (optional).
    if routing.network_concentration is not None and t_n > 0:
        net_counts: Counter[str] = Counter()
        for route in routes:
            net_counts[route.network] += route_counts.get(route.id, 0)
        worst_net = max(net_counts, key=lambda k: net_counts[k]) if net_counts else ""
        share = net_counts.get(worst_net, 0) / t_n
        limit = routing.network_concentration
        checks.append(
            ConstraintCheck(
                name="network_concentration",
                satisfied=share <= limit + 1e-9,
                value=share,
                limit=limit,
                slack=limit - share,
                detail=f"largest network '{worst_net}' share of routed volume",
            )
        )

    # Portfolio approval floor (optional).
    if routing.min_overall_approval is not None:
        mean_appr = float(np.mean(approvals)) if approvals else 0.0
        checks.append(
            ConstraintCheck(
                name="min_overall_approval",
                satisfied=(n_valid == t_n and mean_appr >= routing.min_overall_approval - 1e-9),
                value=mean_appr,
                limit=routing.min_overall_approval,
                slack=mean_appr - routing.min_overall_approval,
                detail="portfolio expected approval rate floor",
            )
        )

    feasible = all(c.satisfied for c in checks)
    _ = tx_by_id  # kept for symmetry/readability
    return feasible, total_cost, checks


# ---------------------------------------------------------------------------
# Reduced instance + QUBO
# ---------------------------------------------------------------------------


@dataclass
class PaymentInstance:
    transactions: list[Transaction]
    routes: list[PaymentRoute]
    routing: RoutingConstraints
    penalty_scale: float = 8.0
    degenerate: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_qubits(self) -> int:
        return len(self.transactions) * len(self.routes)

    @property
    def target(self) -> float:
        return float(len(self.transactions))


def reduce_to_routing_instance(spec: ProblemSpec, max_qubits: int) -> PaymentInstance:
    routing = spec.routing
    assert routing is not None
    routes_all = spec.routes
    if not routes_all or not spec.transactions:
        return PaymentInstance(
            transactions=[],
            routes=routes_all[:1],
            routing=routing,
            degenerate=True,
            provenance={"reason": "no transactions or routes to route"},
        )

    # Prefer cheap, high-approval routes; keep the instance small (T*R <= budget).
    routes = sorted(routes_all, key=lambda r: (r.cost_bps, -r.approval_rate))[:ROUTES_PER_INSTANCE]
    route_ids = {r.id for r in routes}
    routable = [t for t in spec.transactions if any(is_eligible(t, rid) for rid in route_ids)]
    t_budget = max(1, max_qubits // max(1, len(routes)))
    # Route the largest transactions (highest stakes) in the research instance.
    selected = sorted(routable, key=lambda t: t.amount, reverse=True)[:t_budget]

    if not selected:
        return PaymentInstance(
            transactions=[],
            routes=routes,
            routing=routing,
            degenerate=True,
            provenance={"reason": "no transactions eligible for the selected routes"},
        )

    return PaymentInstance(
        transactions=selected,
        routes=routes,
        routing=routing,
        provenance={
            "selection_rule": "cheapest/highest-approval routes; largest transactions",
            "routes_selected": len(routes),
            "transactions_selected": len(selected),
            "from_transactions": len(spec.transactions),
        },
    )


def build_routing_qubo(instance: PaymentInstance, penalty_scale: float | None = None) -> Qubo:
    """One-hot assignment QUBO: minimise cost s.t. each transaction -> one route.

    Capacity / network / approval are NOT encoded (verification-only). Ineligible
    assignments carry a large penalty.
    """
    txs, routes = instance.transactions, instance.routes
    t_n, r_n = len(txs), len(routes)
    n = t_n * r_n
    if n == 0:
        return Qubo(
            Q={},
            offset=0.0,
            n=0,
            num_decision=0,
            ids=[],
            encoding_losses=["Degenerate instance: no QUBO to build."],
        )

    def idx(t: int, r: int) -> int:
        return t * r_n + r

    cost = np.zeros(n)
    for t, tx in enumerate(txs):
        for r, route in enumerate(routes):
            cost[idx(t, r)] = assignment_cost(tx, route, routing=instance.routing)
    cost_scale = float(cost.sum()) or 1.0
    c_t = cost / cost_scale

    A = penalty_scale if penalty_scale is not None else instance.penalty_scale
    big = 5.0 * A  # eligibility penalty; strongly discourages ineligible slots

    Q: dict[tuple[int, int], float] = {}

    def add(i: int, j: int, v: float) -> None:
        key = (i, j) if i <= j else (j, i)
        Q[key] = Q.get(key, 0.0) + v

    for t, tx in enumerate(txs):
        for r, route in enumerate(routes):
            i = idx(t, r)
            add(i, i, float(c_t[i]))  # objective
            add(i, i, -A)  # one-hot linear (from x^2 - 2x, +constant below)
            if not is_eligible(tx, route.id):
                add(i, i, big)  # forbid ineligible
        # one-hot quadratic: penalise picking two routes for the same transaction
        for r1 in range(r_n):
            for r2 in range(r1 + 1, r_n):
                add(idx(t, r1), idx(t, r2), 2.0 * A)

    offset = float(A * t_n)  # sum_t A*1 from (·-1)^2 constant term
    losses = [
        "Each transaction is assigned exactly one route via a one-hot penalty; "
        "the QUBO ground state is a valid assignment when the penalty dominates.",
        "Ineligible (transaction, route) pairs carry a large penalty rather than being removed.",
        "Route capacity, network diversification, and the approval floor are NOT encoded; "
        "the Verification agent re-checks the decoded assignment against them.",
    ]
    ids = [f"{txs[t].id}{_KEY}{routes[r].id}" for t in range(t_n) for r in range(r_n)]
    return Qubo(Q=Q, offset=offset, n=n, num_decision=n, ids=ids, encoding_losses=losses)


def bits_to_routing_allocation(
    instance: PaymentInstance, bits: NDArray[Any] | list[int]
) -> Allocation:
    txs, routes = instance.transactions, instance.routes
    r_n = len(routes)
    x: dict[str, float] = {}
    for t, tx in enumerate(txs):
        for r, route in enumerate(routes):
            if int(bits[t * r_n + r]) == 1:
                x[f"{tx.id}{_KEY}{route.id}"] = 1.0
    return Allocation(x=x)


# ---------------------------------------------------------------------------
# Domain adapter
# ---------------------------------------------------------------------------


class PaymentRoutingDomain(ProblemDomain):
    problem = "payment_routing"

    def requirements(self, spec: ProblemSpec) -> RequirementsReport:
        routing = spec.routing
        assert routing is not None
        txs, routes = spec.transactions, spec.routes
        route_ids = {r.id for r in routes}
        unroutable = [t.id for t in txs if not any(is_eligible(t, rid) for rid in route_ids)]
        gaps: list[str] = []
        if not txs:
            gaps.append("No transactions to route.")
        if unroutable:
            gaps.append(
                f"{len(unroutable)} transaction(s) have no eligible route: "
                f"{', '.join(unroutable[:5])}."
            )
        if all(r.capacity >= len(txs) for r in routes):
            gaps.append("No binding route capacities — capacity constraints are slack.")
        total_amount = sum(t.amount for t in txs)
        avg_appr = float(np.mean([r.approval_rate for r in routes])) if routes else 0.0
        return RequirementsReport(
            problem=self.problem,
            summary=(
                f"{len(txs)} transactions across {len(routes)} routes, "
                f"total amount {total_amount:,.0f}"
            ),
            metrics={
                "n_transactions": float(len(txs)),
                "n_routes": float(len(routes)),
                "total_amount": total_amount,
                "avg_route_approval": avg_appr,
            },
            feasible_upper_bound=not unroutable and bool(txs),
            discovered_gaps=gaps,
            assumptions=["Empty eligible_routes means a transaction may use any route."],
            autonomy_level=spec.execution_policy.autonomy_level.value,
        )

    def formulations(self, spec: ProblemSpec) -> FormulationCatalogue:
        t_n, r_n = len(spec.transactions), len(spec.routes)
        return FormulationCatalogue(
            catalogue=[
                Formulation(
                    name="assignment_lp",
                    kind="Linear Program",
                    variables=f"{t_n}x{r_n} continuous x_(t,r) in [0,1]",
                    represents="cost, one-hot assignment, capacity, network, approval",
                    note="LP relaxation; a lower bound on routing cost.",
                ),
                Formulation(
                    name="assignment_milp",
                    kind="Mixed-Integer Linear Program (GAP)",
                    variables=f"{t_n}x{r_n} binary x_(t,r)",
                    represents="cost + all constraints exactly",
                    note="The FAIR classical comparator (generalized assignment).",
                ),
                Formulation(
                    name="qubo",
                    kind="Quadratic Unconstrained Binary Optimisation",
                    variables="binary x_(t,r) on a reduced instance",
                    represents="cost + one-hot assignment (penalty)",
                    note="Capacity/network/approval become verification-only constraints.",
                ),
                Formulation(
                    name="ising",
                    kind="Ising Hamiltonian",
                    variables="spins z, x = (1 - z)/2",
                    represents="same as the QUBO",
                    note="Direct input to gate-model QAOA and to annealers.",
                ),
            ],
            selected_classical="assignment_milp",
            selected_quantum_path="qubo -> ising -> QAOA",
            encoding_loss_note=(
                "The quantum path solves a reduced routing instance; the decoded "
                "assignment is re-checked against the full instance constraints."
            ),
        )

    def solve_classical_full(self, spec: ProblemSpec) -> ClassicalBaseline:
        import time

        routing = spec.routing
        assert routing is not None
        t0 = time.perf_counter()
        lp = solve_routing_milp(spec.transactions, spec.routes, routing, integer=False)
        lp_dt = time.perf_counter() - t0
        t0 = time.perf_counter()
        mp = solve_routing_milp(spec.transactions, spec.routes, routing, integer=True)
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
            metadata={"role": "lower bound on cost", "status": lp.status},
        )
        n_routed = len(mp.allocation.x) if mp.allocation is not None else 0
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
                "n_posted": n_routed,
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

    def reduce_to_instance(self, spec: ProblemSpec, max_qubits: int) -> PaymentInstance:
        return reduce_to_routing_instance(spec, max_qubits)

    def build_qubo(self, instance: PaymentInstance, *, slack_bits: int) -> Qubo:  # type: ignore[override]
        return build_routing_qubo(instance)

    def solve_instance_classical(self, instance: PaymentInstance) -> SolveResult:  # type: ignore[override]
        import time

        t0 = time.perf_counter()
        out = solve_routing_milp(
            instance.transactions, instance.routes, instance.routing, integer=True
        )
        dt = time.perf_counter() - t0
        if out.feasible and out.allocation is not None:
            feasible, obj, _ = check_routing_constraints(
                instance.transactions, instance.routes, instance.routing, out.allocation
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
        instance: PaymentInstance,
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
        alloc = bits_to_routing_allocation(instance, bits)
        feasible, obj, _ = check_routing_constraints(
            instance.transactions, instance.routes, instance.routing, alloc
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
        txs: list[Transaction],
        routes: list[PaymentRoute],
        routing: RoutingConstraints,
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
        feasible, obj, checks = check_routing_constraints(txs, routes, routing, result.allocation)
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
        routing = spec.routing
        assert routing is not None
        return self._verify(result, spec.transactions, spec.routes, routing)

    def verify_instance(  # type: ignore[override]
        self, instance: PaymentInstance, result: SolveResult
    ) -> VerificationReport:
        return self._verify(result, instance.transactions, instance.routes, instance.routing)

    def instance_warm_start(  # type: ignore[override]
        self, instance: PaymentInstance, qubo: Qubo
    ) -> list[float] | None:
        """Warm-start: spread each transaction's mass uniformly over its eligible
        routes (a max-entropy assignment respecting eligibility)."""
        if qubo.n == 0:
            return None
        txs, routes = instance.transactions, instance.routes
        r_n = len(routes)
        biases = [0.0] * qubo.n
        for t, tx in enumerate(txs):
            elig = [r for r, route in enumerate(routes) if is_eligible(tx, route.id)]
            if not elig:
                continue
            w = 1.0 / len(elig)
            for r in elig:
                biases[t * r_n + r] = w
        return biases
