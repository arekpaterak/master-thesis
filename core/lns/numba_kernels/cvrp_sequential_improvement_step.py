"""Numba CPU kernels for the sequential CVRP improvement step.

The kernels operate on raw numpy ``next``/``prev`` arrays (the linked-list
representation of :class:`CVRPGraphSolution`) plus the instance arrays. They fold
the whole rollout loop -- stitch-destroy, best-of-n greedy cheapest-insertion and
acceptance -- into a single ``njit(parallel=True)`` pass over the flattened batch,
replacing the hundreds of thousands of tiny GPU kernel launches the torch path
issues for the same work.

``route_id`` is intentionally NOT maintained here; the caller recomputes it once
via :meth:`CVRPGraphSolution._compute_route_ids`. The greedy-insert policy mirrors
:class:`core.lns.operators.repair.insert.cvrp.CVRPGreedyInsertRepair` (random free
customer, then cheapest valid position, with an always-valid new-route fallback)
and acceptance mirrors :mod:`core.lns.acceptance_criterion`.

Randomness uses numba's per-thread ``np.random``. To stay reproducible under
``prange`` (where draws would otherwise depend on the thread count), each parallel
iteration reseeds ``np.random`` from a seed derived only from ``base_seed`` and the
iteration index -- see :func:`seed_for`.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .common import dist, seed_for

DEPOT = 0
UNASSIGNED = -1


def run_with_removals(
    *,
    next: np.ndarray,
    prev: np.ndarray,
    nodes_to_remove: np.ndarray,
    locations: np.ndarray,
    demand: np.ndarray,
    capacity: np.ndarray,
    n_repairs: int,
    accept_kind: int,
    temperature: float,
    eps: float,
    base_seed: int,
    first_repair_uses_order: bool = False,
    insert_in_new_routes_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential step with externally supplied nodes to remove.

    When ``first_repair_uses_order`` is set, the first of the ``n_repairs`` greedy
    insertions follows the provided removal order instead of a random order.

    When ``insert_in_new_routes_only`` is set, removed customers may only be reinserted
    into routes the destroy touched (their remnants) or routes created during this repair,
    never an untouched route.

    Returns:
        (next_out, prev_out) both (G, N+1) int64.
    """
    return _run_for_batch(
        next=np.ascontiguousarray(next, dtype=np.int64),
        prev=np.ascontiguousarray(prev, dtype=np.int64),
        nodes_to_remove=np.ascontiguousarray(nodes_to_remove, dtype=np.int64),
        locations=np.ascontiguousarray(locations, dtype=np.float64),
        demand=np.ascontiguousarray(demand, dtype=np.float64),
        capacity=np.ascontiguousarray(capacity, dtype=np.float64),
        n_repairs=int(n_repairs),
        accept_kind=int(accept_kind),
        temperature=float(temperature),
        eps=float(eps),
        base_seed=int(base_seed),
        first_repair_uses_order=bool(first_repair_uses_order),
        insert_in_new_routes_only=bool(insert_in_new_routes_only),
    )


@njit(parallel=True, cache=True)
def _run_for_batch(
    *,
    next: np.ndarray,
    prev: np.ndarray,
    nodes_to_remove: np.ndarray,
    locations: np.ndarray,
    demand: np.ndarray,
    capacity: np.ndarray,
    n_repairs: int,
    accept_kind: int,
    temperature: float,
    eps: float,
    base_seed: int,
    first_repair_uses_order: bool,
    insert_in_new_routes_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    global_size = next.shape[0]
    next_out = next.copy()
    prev_out = prev.copy()
    for global_idx in prange(global_size):
        np.random.seed(seed_for(base_seed, global_idx, global_size))
        _run_for_single_sequentially(
            next_out=next_out[global_idx],
            prev_out=prev_out[global_idx],
            nodes_to_remove=nodes_to_remove[global_idx],
            locations=locations[global_idx],
            demand=demand[global_idx],
            capacity=capacity[global_idx],
            n_repairs=n_repairs,
            accept_kind=accept_kind,
            temperature=temperature,
            eps=eps,
            first_repair_uses_order=first_repair_uses_order,
            insert_in_new_routes_only=insert_in_new_routes_only,
        )
    return next_out, prev_out


@njit
def _run_for_single_sequentially(
    next_out: np.ndarray,
    prev_out: np.ndarray,
    nodes_to_remove: np.ndarray,
    locations: np.ndarray,
    demand: np.ndarray,
    capacity: float,
    n_repairs: int,
    accept_kind: int,
    temperature: float,
    eps: float,
    first_repair_uses_order: bool,
    insert_in_new_routes_only: bool,
) -> None:
    """Run the full sequential rollout loop for one batch element.

    Assumes ``np.random`` has already been reseeded for this element by the caller.
    """
    num_nodes = next_out.shape[0]
    num_customers = num_nodes - 1
    num_rollouts = nodes_to_remove.shape[0]
    deconstruction_size = nodes_to_remove.shape[1]

    deconstructed_next = np.empty(num_nodes, np.int64)
    deconstructed_prev = np.empty(num_nodes, np.int64)
    repaired_next = np.empty(num_nodes, np.int64)
    repaired_prev = np.empty(num_nodes, np.int64)
    best_repaired_next = np.empty(num_nodes, np.int64)
    best_repaired_prev = np.empty(num_nodes, np.int64)
    route_demand = np.empty(num_nodes, np.float64)
    # base_insertable is the destroy-determined mask (same across the n_repairs tries);
    # insertable is the per-repair working copy that _insert_customer_at_cheapest mutates.
    base_insertable = np.empty(num_nodes, np.bool_)
    insertable = np.empty(num_nodes, np.bool_)

    current_objective = _objective(
        nxt=next_out, prv=prev_out, locations=locations, num_customers=num_customers
    )

    for r in range(num_rollouts):
        # Mark insertable routes before stitching, while next_out/prev_out (the current
        # solution this rollout destroys) is still intact.
        if insert_in_new_routes_only:
            base_insertable[:] = False
            for m in range(deconstruction_size):
                c = nodes_to_remove[r, m]
                if c < 1:
                    continue
                if prev_out[c] == UNASSIGNED and next_out[c] == UNASSIGNED:
                    continue
                _mark_route(next_out, prev_out, c, base_insertable)
        else:
            base_insertable[:] = True

        deconstructed_next[:] = next_out
        deconstructed_prev[:] = prev_out
        for m in range(deconstruction_size):
            c = nodes_to_remove[r, m]
            if c < 1:
                continue
            if deconstructed_prev[c] == UNASSIGNED and deconstructed_next[c] == UNASSIGNED:
                continue
            _remove_customer_and_stitch(deconstructed_next, deconstructed_prev, c)

        best_repaired_objective = np.inf
        for i in range(n_repairs):
            repaired_next[:] = deconstructed_next
            repaired_prev[:] = deconstructed_prev
            insertable[:] = base_insertable

            order = None
            if first_repair_uses_order and i == 0:
                order = nodes_to_remove[r]
            _greedy_insert_repair(
                nxt=repaired_next,
                prv=repaired_prev,
                locations=locations,
                demand=demand,
                capacity=capacity,
                num_customers=num_customers,
                route_demand=route_demand,
                insertable=insertable,
                order=order,
            )
            objective = _objective(
                nxt=repaired_next,
                prv=repaired_prev,
                locations=locations,
                num_customers=num_customers,
            )
            if objective < best_repaired_objective:
                best_repaired_objective = objective
                best_repaired_next[:] = repaired_next
                best_repaired_prev[:] = repaired_prev

        if accept_kind == 0:
            accepted = best_repaired_objective < current_objective
        else:
            delta = best_repaired_objective - current_objective
            if delta < eps:
                accepted = True
            else:
                accepted = np.random.random() < np.exp(-max(delta, 0.0) / temperature)

        if accepted:
            next_out[:] = best_repaired_next
            prev_out[:] = best_repaired_prev
            current_objective = best_repaired_objective


# --- graph helpers (single element, length N+1 arrays) ----------------------


@njit(inline="always")
def _remove_customer_and_stitch(nxt: np.ndarray, prv: np.ndarray, customer: int) -> None:
    """Detach customer and stitch its neighbours."""
    u = prv[customer]
    v = nxt[customer]
    if u > 0:
        nxt[u] = v
    if v > 0:
        prv[v] = u
    prv[customer] = UNASSIGNED
    nxt[customer] = UNASSIGNED
    nxt[DEPOT] = UNASSIGNED
    prv[DEPOT] = UNASSIGNED


@njit
def _mark_route(nxt: np.ndarray, prv: np.ndarray, member: int, insertable: np.ndarray) -> None:
    """Mark every customer in member's route as insertable. Route must be intact."""
    head = member
    while prv[head] != DEPOT:
        head = prv[head]
    node = head
    while True:
        insertable[node] = True
        node = nxt[node]
        if node == DEPOT:
            break


@njit
def _objective(
    nxt: np.ndarray, prv: np.ndarray, locations: np.ndarray, num_customers: int
) -> float:
    total_distance = 0.0
    for head in range(1, num_customers + 1):
        if prv[head] == DEPOT:
            total_distance += dist(locations, DEPOT, head)
            node = head
            while True:
                next_node = nxt[node]
                if next_node == DEPOT:
                    total_distance += dist(locations, node, DEPOT)
                    break
                total_distance += dist(locations, node, next_node)
                node = next_node
    return total_distance


@njit
def _compute_route_demand(
    nxt: np.ndarray,
    prv: np.ndarray,
    demand: np.ndarray,
    num_customers: int,
    route_demand: np.ndarray,
) -> None:
    """Fill ``route_demand[x]`` with the total demand of x's route (placed x only)."""
    for i in range(num_customers + 1):
        route_demand[i] = 0.0
    for head in range(1, num_customers + 1):
        if prv[head] == DEPOT:
            s = 0.0
            node = head
            while True:
                s += demand[node]
                nn = nxt[node]
                if nn == DEPOT:
                    break
                node = nn
            node = head
            while True:
                route_demand[node] = s
                nn = nxt[node]
                if nn == DEPOT:
                    break
                node = nn


@njit
def _pick_random_free_customer(nxt: np.ndarray, prv: np.ndarray, num_customers: int) -> int:
    """Pick a random free customer (1..N), or -1 if none are free."""
    chosen = -1
    best_score = -1.0
    for c in range(1, num_customers + 1):
        score = np.random.random()
        if prv[c] == UNASSIGNED and nxt[c] == UNASSIGNED and score > best_score:
            best_score = score
            chosen = c
    return chosen


@njit
def _insert_customer_at_cheapest(
    customer: int,
    nxt: np.ndarray,
    prv: np.ndarray,
    locations: np.ndarray,
    demand: np.ndarray,
    capacity: float,
    num_customers: int,
    route_demand: np.ndarray,
    insertable: np.ndarray,
) -> None:
    """Insert free ``customer`` at its cheapest valid position. In-place.

    Only positions ``x`` with ``insertable[x]`` set are considered (everything is
    insertable unless the caller restricts to new routes). The just-placed customer is
    marked insertable so it can host later customers in the same repair.
    """
    demand_c = demand[customer]
    _compute_route_demand(
        nxt=nxt, prv=prv, demand=demand, num_customers=num_customers, route_demand=route_demand
    )

    # New-route fallback (depot -> c -> depot) is always valid for a feasible
    # single customer, so the argmin is always defined.
    best_cost = 2.0 * dist(locations, DEPOT, customer)
    best_u = DEPOT
    best_v = DEPOT
    for x in range(1, num_customers + 1):
        if prv[x] == UNASSIGNED and nxt[x] == UNASSIGNED:
            continue  # free customer, not a position
        if not insertable[x]:
            continue  # untouched route, off-limits when inserting in new routes only
        if route_demand[x] + demand_c <= capacity:
            u = prv[x]
            cost = (
                dist(locations, u, customer) + dist(locations, customer, x) - dist(locations, u, x)
            )
            if cost < best_cost:
                best_cost = cost
                best_u = u
                best_v = x
            if nxt[x] == DEPOT:
                cost = (
                    dist(locations, x, customer)
                    + dist(locations, customer, DEPOT)
                    - dist(locations, x, DEPOT)
                )
                if cost < best_cost:
                    best_cost = cost
                    best_u = x
                    best_v = DEPOT

    prv[customer] = best_u
    nxt[customer] = best_v
    if best_u > 0:
        nxt[best_u] = customer
    if best_v > 0:
        prv[best_v] = customer
    nxt[DEPOT] = UNASSIGNED
    prv[DEPOT] = UNASSIGNED
    # The customer now sits in an affected/new route, so later customers may join it.
    insertable[customer] = True


@njit
def _greedy_insert_repair(
    nxt: np.ndarray,
    prv: np.ndarray,
    locations: np.ndarray,
    demand: np.ndarray,
    capacity: float,
    num_customers: int,
    route_demand: np.ndarray,
    insertable: np.ndarray,
    order: np.ndarray | None = None,
) -> None:
    """Insert every free customer at its cheapest valid position. In-place.

    When ``order`` is given (the model rollout removal order for this rollout),
    free customers are inserted in that order first, mirroring the original NDS
    behaviour where the first repair follows the model order. Any customers not
    covered by ``order`` are then drained in random order. When ``order`` is
    omitted all customers are inserted in random order.

    ``insertable`` is the per-repair position mask forwarded to each insertion.
    """
    if order is not None:
        for k in range(order.shape[0]):
            customer = order[k]
            if customer < 1:
                continue
            if prv[customer] == UNASSIGNED and nxt[customer] == UNASSIGNED:
                _insert_customer_at_cheapest(
                    customer,
                    nxt,
                    prv,
                    locations,
                    demand,
                    capacity,
                    num_customers,
                    route_demand,
                    insertable,
                )

    while True:
        customer = _pick_random_free_customer(nxt, prv, num_customers)
        if customer == -1:
            break
        _insert_customer_at_cheapest(
            customer,
            nxt,
            prv,
            locations,
            demand,
            capacity,
            num_customers,
            route_demand,
            insertable,
        )
