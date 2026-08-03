"""Numba CPU kernels for CVRP apply-and-repair.

Both entry points take externally supplied node sets and, per element, deconstruct
the solution (stitch-remove those customers) and greedily reconstruct it via
best-of-n cheapest-insertion. They differ only in how the node sets are applied:

- ``apply_and_repair_sequential`` applies a solution's rollouts one after another,
  threading acceptance through a running solution (mirrors ``SequentialImprovementStep``).
- ``apply_and_repair_parallel`` applies every rollout independently from the same
  solution and returns all candidates with their objectives, no acceptance
  (mirrors ``ParallelImprovementStep``; used by training to score rollouts).

Both share :func:`_apply_and_repair_one`, the per-element deconstruct + best-of-n
repair core. The kernels operate on raw numpy ``next``/``prev`` arrays (the linked-list
representation of :class:`CVRPGraphSolution`) plus the instance arrays, folding the
work into ``njit(parallel=True)`` passes and replacing the hundreds of thousands of
tiny GPU kernel launches the torch path issues for the same work. Inside the kernels
the per-element arrays travel bundled as :class:`Solution` (``next``/``prev``) and
:class:`Instance` (``locations``/``demand``/``capacity``); the batch drivers construct
these views per element before dispatching.

``route_id`` is intentionally NOT maintained here; the caller recomputes it once
via :meth:`CVRPGraphSolution._compute_route_ids`. The greedy-insert policy mirrors
:class:`core.lns.operators.repair.insert.cvrp.CVRPGreedyInsertRepair` (random free
customer, then cheapest valid position, with an always-valid new-route fallback).

Randomness uses numba's per-thread ``np.random``. To stay reproducible under
``prange`` (where draws would otherwise depend on the thread count), each parallel
iteration reseeds ``np.random`` from a seed derived only from ``base_seed`` and the
iteration index -- see :func:`seed_for`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import njit, prange

from .common import dist, seed_for

DEPOT = 0
UNASSIGNED = -1


class Instance(NamedTuple):
    locations: np.ndarray
    demand: np.ndarray
    capacity: float


class Solution(NamedTuple):
    next: np.ndarray
    prev: np.ndarray


def apply_and_repair_sequential(
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
    """Apply each solution's rollouts sequentially with acceptance.

    For every element, the rollouts in ``nodes_to_remove`` are applied one after
    another: each deconstructs the running solution, is repaired best-of-n, and
    replaces the running solution when acceptance succeeds.

    When ``first_repair_uses_order`` is set, the first of the ``n_repairs`` greedy
    insertions follows the provided removal order instead of a random order.

    When ``insert_in_new_routes_only`` is set, removed customers may only be reinserted
    into routes the destroy touched (their remnants) or routes created during this repair,
    never an untouched route.

    Returns:
        (next_out, prev_out) both (G, N+1) int64.
    """
    return _sequential_batch(
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


def apply_and_repair_parallel(
    *,
    next: np.ndarray,
    prev: np.ndarray,
    nodes: np.ndarray,
    locations: np.ndarray,
    demand: np.ndarray,
    capacity: np.ndarray,
    n_repairs: int,
    base_seed: int,
    first_repair_uses_order: bool = False,
    insert_in_new_routes_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply every rollout independently.

    Each of the ``B`` solutions is deconstructed and repaired once per rollout ``k``,
    all from the same starting solution. No acceptance -- every repaired candidate and
    its objective is returned so the caller (e.g. training) can score all rollouts.

    Args:
        next, prev: (B, N+1) the solution to deconstruct, one per instance.
        nodes: (B, K, M) the K rollouts (customer sets to remove) per instance.
        locations, demand, capacity: instance arrays, (B, N+1, 2)/(B, N+1)/(B,);
            shared across the K rollouts of an instance (not replicated).

    Returns:
        (next_out, prev_out, objective) shaped (B, K, N+1), (B, K, N+1), (B, K).
    """
    return _parallel_batch(
        next=np.ascontiguousarray(next, dtype=np.int64),
        prev=np.ascontiguousarray(prev, dtype=np.int64),
        nodes=np.ascontiguousarray(nodes, dtype=np.int64),
        locations=np.ascontiguousarray(locations, dtype=np.float64),
        demand=np.ascontiguousarray(demand, dtype=np.float64),
        capacity=np.ascontiguousarray(capacity, dtype=np.float64),
        n_repairs=int(n_repairs),
        base_seed=int(base_seed),
        first_repair_uses_order=bool(first_repair_uses_order),
        insert_in_new_routes_only=bool(insert_in_new_routes_only),
    )


@njit(parallel=True, cache=True)
def _sequential_batch(
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
        _sequential_single_(
            solution=Solution(next_out[global_idx], prev_out[global_idx]),
            nodes_to_remove=nodes_to_remove[global_idx],
            instance=Instance(locations[global_idx], demand[global_idx], capacity[global_idx]),
            n_repairs=n_repairs,
            accept_kind=accept_kind,
            temperature=temperature,
            eps=eps,
            first_repair_uses_order=first_repair_uses_order,
            insert_in_new_routes_only=insert_in_new_routes_only,
        )
    return next_out, prev_out


@njit(parallel=True, cache=True)
def _parallel_batch(
    *,
    next: np.ndarray,
    prev: np.ndarray,
    nodes: np.ndarray,
    locations: np.ndarray,
    demand: np.ndarray,
    capacity: np.ndarray,
    n_repairs: int,
    base_seed: int,
    first_repair_uses_order: bool,
    insert_in_new_routes_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_instances = next.shape[0]
    num_rollouts = nodes.shape[1]
    num_nodes = next.shape[1]
    total = num_instances * num_rollouts

    next_out = np.empty((num_instances, num_rollouts, num_nodes), np.int64)
    prev_out = np.empty((num_instances, num_rollouts, num_nodes), np.int64)
    objective = np.empty((num_instances, num_rollouts), np.float64)

    for b in prange(num_instances):
        instance = Instance(locations[b], demand[b], capacity[b])
        current = Solution(next[b], prev[b])
        for k in range(num_rollouts):
            np.random.seed(seed_for(base_seed, b * num_rollouts + k, total))
            objective[b, k] = _apply_and_repair_one(
                current=current,
                nodes=nodes[b, k],
                instance=instance,
                n_repairs=n_repairs,
                first_repair_uses_order=first_repair_uses_order,
                insert_in_new_routes_only=insert_in_new_routes_only,
                out=Solution(next_out[b, k], prev_out[b, k]),
            )
    return next_out, prev_out, objective


@njit
def _sequential_single_(
    solution: Solution,
    nodes_to_remove: np.ndarray,
    instance: Instance,
    n_repairs: int,
    accept_kind: int,
    temperature: float,
    eps: float,
    first_repair_uses_order: bool,
    insert_in_new_routes_only: bool,
) -> None:
    """Apply one element's rollouts sequentially, updating the running solution.

    Assumes ``np.random`` has already been reseeded for this element by the caller.
    Each rollout deconstructs and repairs the running ``solution`` via
    :func:`_apply_and_repair_one`; the running solution is replaced when accepted.
    """
    nxt, prv = solution
    num_nodes = nxt.shape[0]
    num_customers = num_nodes - 1
    num_rollouts = nodes_to_remove.shape[0]

    candidate = Solution(np.empty(num_nodes, np.int64), np.empty(num_nodes, np.int64))

    current_objective = _objective(solution, instance, num_customers)

    for r in range(num_rollouts):
        candidate_objective = _apply_and_repair_one(
            current=solution,
            nodes=nodes_to_remove[r],
            instance=instance,
            n_repairs=n_repairs,
            first_repair_uses_order=first_repair_uses_order,
            insert_in_new_routes_only=insert_in_new_routes_only,
            out=candidate,
        )

        if accept_kind == 0:
            accepted = candidate_objective < current_objective
        else:
            delta = candidate_objective - current_objective
            if delta < eps:
                accepted = True
            else:
                accepted = np.random.random() < np.exp(-max(delta, 0.0) / temperature)

        if accepted:
            nxt[:] = candidate.next
            prv[:] = candidate.prev
            current_objective = candidate_objective


@njit
def _apply_and_repair_one(
    current: Solution,
    nodes: np.ndarray,
    instance: Instance,
    n_repairs: int,
    first_repair_uses_order: bool,
    insert_in_new_routes_only: bool,
    out: Solution,
) -> float:
    """Deconstruct one customer set and greedily repair it as the best of ``n_repairs`` attempts.

    Writes the best repaired solution into ``out`` and returns its objective. Reads
    ``current`` (the solution to deconstruct) without mutating it, so callers may apply
    many node sets to the same solution.
    """
    current_next, current_prev = current
    num_nodes = current_next.shape[0]
    num_customers = num_nodes - 1

    deconstructed = Solution(np.empty(num_nodes, np.int64), np.empty(num_nodes, np.int64))
    repaired = Solution(np.empty(num_nodes, np.int64), np.empty(num_nodes, np.int64))
    route_demand = np.empty(num_nodes, np.float64)
    # base_insertable is the destroy-determined mask (same across the n_repairs tries);
    # insertable is the per-repair working copy that _insert_customer_at_cheapest_ mutates.
    base_insertable = np.empty(num_nodes, np.bool_)
    insertable = np.empty(num_nodes, np.bool_)

    # Mark insertable routes from the current solution, before it is deconstructed.
    if insert_in_new_routes_only:
        base_insertable[:] = False
        for c in nodes:
            if c >= 1 and not _is_free(current, c):
                _mark_route_(current, c, base_insertable)
    else:
        base_insertable[:] = True

    # Apply destroy
    deconstructed.next[:] = current_next
    deconstructed.prev[:] = current_prev
    for c in nodes:
        if c >= 1 and not _is_free(deconstructed, c):
            _remove_customer_and_stitch_(deconstructed, c)

    best_repaired_objective = np.inf
    for i in range(n_repairs):
        repaired.next[:] = deconstructed.next
        repaired.prev[:] = deconstructed.prev
        insertable[:] = base_insertable

        order = None
        if first_repair_uses_order and i == 0:
            order = nodes
        _greedy_insert_repair_(repaired, instance, num_customers, route_demand, insertable, order)
        objective = _objective(repaired, instance, num_customers)
        if objective < best_repaired_objective:
            best_repaired_objective = objective
            out.next[:] = repaired.next
            out.prev[:] = repaired.prev

    return best_repaired_objective


# --- graph helpers (single element, length N+1 arrays) ----------------------


@njit(inline="always")
def _remove_customer_and_stitch_(solution: Solution, customer: int) -> None:
    """Detach customer and stitch its neighbours."""
    nxt, prv = solution
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


@njit(inline="always")
def _is_free(solution: Solution, node: int) -> bool:
    """Whether ``node`` is unassigned (not placed in any route)."""
    return solution.prev[node] == UNASSIGNED and solution.next[node] == UNASSIGNED


@njit(inline="always")
def _get_route_head(solution: Solution, node: int) -> int:
    """Traverse backwards to find the first node in ``node``'s route."""
    prv = solution.prev
    head = node
    while prv[head] != DEPOT:
        head = prv[head]
    return head


@njit
def _mark_route_(solution: Solution, member: int, insertable: np.ndarray) -> None:
    """Mark every customer in member's route as insertable. Route must be intact."""
    nxt = solution.next
    node = _get_route_head(solution, member)
    while True:
        insertable[node] = True
        node = nxt[node]
        if node == DEPOT:
            break


@njit
def _objective(solution: Solution, instance: Instance, num_customers: int) -> float:
    nxt, prv = solution
    locations = instance.locations
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
def _compute_route_demand_(
    solution: Solution,
    instance: Instance,
    num_customers: int,
    route_demand: np.ndarray,
) -> None:
    """Fill ``route_demand[x]`` with the total demand of x's route (placed x only)."""
    nxt, prv = solution
    demand = instance.demand
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
def _pick_random_free_customer(solution: Solution, num_customers: int) -> int:
    """Pick a random free customer (1..N), or -1 if none are free."""
    chosen = -1
    best_score = -1.0
    for c in range(1, num_customers + 1):
        score = np.random.random()
        if _is_free(solution, c) and score > best_score:
            best_score = score
            chosen = c
    return chosen


@njit
def _insert_customer_at_cheapest_(
    customer: int,
    solution: Solution,
    instance: Instance,
    num_customers: int,
    route_demand: np.ndarray,
    insertable: np.ndarray,
) -> None:
    """Insert free ``customer`` at its cheapest valid position. In-place.

    Only positions ``x`` with ``insertable[x]`` set are considered (everything is
    insertable unless the caller restricts to new routes). The just-placed customer is
    marked insertable so it can host later customers in the same repair.

    ``route_demand`` must already reflect the current routes; it is maintained
    incrementally here after the customer is placed, so callers seed it once (see
    :func:`_greedy_insert_repair_`) rather than recomputing it every insertion.
    """
    nxt, prv = solution
    locations, demand, capacity = instance
    demand_c = demand[customer]

    # New-route fallback (depot -> c -> depot) is always valid for a feasible
    # single customer, so the argmin is always defined.
    best_cost = 2.0 * dist(locations, DEPOT, customer)
    best_u = DEPOT
    best_v = DEPOT
    for x in range(1, num_customers + 1):
        if _is_free(solution, x):
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

    # Only the joined route changed: bump every member's cached total by demand_c.
    if best_v != DEPOT:
        new_total = route_demand[best_v] + demand_c
    elif best_u != DEPOT:
        new_total = route_demand[best_u] + demand_c
    else:
        new_total = demand_c  # new route: depot -> customer -> depot
    node = _get_route_head(solution, customer)
    while True:
        route_demand[node] = new_total
        node = nxt[node]
        if node == DEPOT:
            break


@njit
def _greedy_insert_repair_(
    solution: Solution,
    instance: Instance,
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
    # Seed the route-demand cache once for the deconstructed routes; each insertion
    # then maintains it incrementally instead of recomputing all routes every time.
    _compute_route_demand_(solution, instance, num_customers, route_demand)

    if order is not None:
        for k in range(order.shape[0]):
            customer = order[k]
            if customer < 1:
                continue
            if _is_free(solution, customer):
                _insert_customer_at_cheapest_(
                    customer, solution, instance, num_customers, route_demand, insertable
                )

    while True:
        customer = _pick_random_free_customer(solution, num_customers)
        if customer == -1:
            break
        _insert_customer_at_cheapest_(
            customer, solution, instance, num_customers, route_demand, insertable
        )
