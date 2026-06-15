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
iteration index -- see :func:`_seed_for`.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

DEPOT = 0
UNASSIGNED = -1

# Independent RNG streams: random-destroy and repair/accept must not draw from the
# same reseeded sequence, so they use different channels.
_NUM_SEED_CHANNELS = 2
_REMOVAL_CHANNEL = 0
_REPAIR_CHANNEL = 1


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
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential step with externally supplied nodes to remove (e.g. model rollouts).

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
    )


def run_with_random(
    *,
    next: np.ndarray,
    prev: np.ndarray,
    locations: np.ndarray,
    demand: np.ndarray,
    capacity: np.ndarray,
    num_rollouts: int,
    destruction_size: int,
    n_repairs: int,
    accept_kind: int,
    temperature: float,
    eps: float,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential step with in-kernel random-node destroy (no model)."""
    next = np.ascontiguousarray(next, dtype=np.int64)
    prev = np.ascontiguousarray(prev, dtype=np.int64)
    global_size = next.shape[0]
    num_customers = next.shape[1] - 1
    nodes_to_remove = _select_random_nodes_for_removal(
        global_size=global_size,
        rollout_size=int(num_rollouts),
        deconstruction_size=int(destruction_size),
        num_customers=num_customers,
        base_seed=int(base_seed),
    )
    return _run_for_batch(
        next=next,
        prev=prev,
        nodes_to_remove=nodes_to_remove,
        locations=np.ascontiguousarray(locations, dtype=np.float64),
        demand=np.ascontiguousarray(demand, dtype=np.float64),
        capacity=np.ascontiguousarray(capacity, dtype=np.float64),
        n_repairs=int(n_repairs),
        accept_kind=int(accept_kind),
        temperature=float(temperature),
        eps=float(eps),
        base_seed=int(base_seed),
    )


@njit(inline="always")
def _seed_for(base_seed, channel, index, count):
    """Distinct 32-bit RNG seed for one (base_seed, channel, index) triple.

    numpy's legacy seeding (which numba mirrors) already disperses nearby integer
    seeds, so this only needs to be injective, not a hash. ``channel`` puts the
    random-destroy and repair draws on independent streams; ``count`` is the number
    of indices per channel (the batch size) so the channels never overlap. Masked
    to 32 bits because ``np.random.seed`` takes a 32-bit seed.
    """
    return ((base_seed * _NUM_SEED_CHANNELS + channel) * count + index) & 0xFFFFFFFF


@njit(parallel=True, cache=True)
def _run_for_batch(
    *,
    next,
    prev,
    nodes_to_remove,
    locations,
    demand,
    capacity,
    n_repairs,
    accept_kind,
    temperature,
    eps,
    base_seed,
):
    global_size = next.shape[0]
    next_out = next.copy()
    prev_out = prev.copy()
    for global_idx in prange(global_size):
        np.random.seed(_seed_for(base_seed, _REPAIR_CHANNEL, global_idx, global_size))
        _run_for_single(
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
        )
    return next_out, prev_out


@njit
def _run_for_single(
    next_out,
    prev_out,
    nodes_to_remove,
    locations,
    demand,
    capacity,
    n_repairs,
    accept_kind,
    temperature,
    eps,
):
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

    current_objective = _objective(
        nxt=next_out, prv=prev_out, locations=locations, num_customers=num_customers
    )

    for r in range(num_rollouts):
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
        for _ in range(n_repairs):
            repaired_next[:] = deconstructed_next
            repaired_prev[:] = deconstructed_prev
            _greedy_insert_repair(
                nxt=repaired_next,
                prv=repaired_prev,
                locations=locations,
                demand=demand,
                capacity=capacity,
                num_customers=num_customers,
                route_demand=route_demand,
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
def _dist(locations, i, j):
    dx = locations[i, 0] - locations[j, 0]
    dy = locations[i, 1] - locations[j, 1]
    return np.sqrt(dx * dx + dy * dy)


@njit(inline="always")
def _remove_customer_and_stitch(nxt, prv, customer):
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
def _objective(nxt, prv, locations, num_customers):
    total_distance = 0.0
    for head in range(1, num_customers + 1):
        if prv[head] == DEPOT:
            total_distance += _dist(locations, DEPOT, head)
            node = head
            while True:
                next_node = nxt[node]
                if next_node == DEPOT:
                    total_distance += _dist(locations, node, DEPOT)
                    break
                total_distance += _dist(locations, node, next_node)
                node = next_node
    return total_distance


@njit
def _compute_route_demand(nxt, prv, demand, num_customers, route_demand):
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
def _greedy_insert_repair(
    nxt,
    prv,
    locations,
    demand,
    capacity,
    num_customers,
    route_demand,
):
    """Insert every free customer at its cheapest valid position. In-place."""
    while True:
        # Pick a random free customer: draw a score per node, mask non-free, argmax.
        chosen = -1
        best_score = -1.0
        for c in range(1, num_customers + 1):
            score = np.random.random()
            if prv[c] == UNASSIGNED and nxt[c] == UNASSIGNED and score > best_score:
                best_score = score
                chosen = c
        if chosen == -1:
            break

        c = chosen
        demand_c = demand[c]
        _compute_route_demand(
            nxt=nxt, prv=prv, demand=demand, num_customers=num_customers, route_demand=route_demand
        )

        # New-route fallback (depot -> c -> depot) is always valid for a feasible
        # single customer, so the argmin is always defined.
        best_cost = 2.0 * _dist(locations, DEPOT, c)
        best_u = DEPOT
        best_v = DEPOT
        for x in range(1, num_customers + 1):
            if prv[x] == UNASSIGNED and nxt[x] == UNASSIGNED:
                continue  # free customer, not a position
            if route_demand[x] + demand_c <= capacity:
                u = prv[x]
                cost = _dist(locations, u, c) + _dist(locations, c, x) - _dist(locations, u, x)
                if cost < best_cost:
                    best_cost = cost
                    best_u = u
                    best_v = x
                if nxt[x] == DEPOT:
                    cost = (
                        _dist(locations, x, c)
                        + _dist(locations, c, DEPOT)
                        - _dist(locations, x, DEPOT)
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_u = x
                        best_v = DEPOT

        prv[c] = best_u
        nxt[c] = best_v
        if best_u > 0:
            nxt[best_u] = c
        if best_v > 0:
            prv[best_v] = c
        nxt[DEPOT] = UNASSIGNED
        prv[DEPOT] = UNASSIGNED


@njit
def _sample_distinct_customers(pool, out):
    """Draw ``len(out)`` distinct customers (1..N) uniformly without replacement.

    Partial Fisher-Yates over ``pool`` (scratch of length N): at step ``m`` it swaps
    a uniformly chosen element from the not-yet-drawn suffix ``[m, N)`` into position
    ``m``, so every emitted customer is distinct. ``pool`` is (re)initialised here, so
    the same buffer can be reused across calls.
    """
    num_customers = pool.shape[0]
    for i in range(num_customers):
        pool[i] = i + 1
    for m in range(out.shape[0]):
        j = np.random.randint(m, num_customers)
        chosen = pool[j]
        pool[j] = pool[m]
        pool[m] = chosen
        out[m] = chosen


@njit(parallel=True, cache=True)
def _select_random_nodes_for_removal(
    *,
    global_size,
    rollout_size,
    deconstruction_size,
    num_customers,
    base_seed,
):
    """Per (element, rollout): sample M distinct customers from 1..N (random-node destroy)."""
    nodes_to_remove = np.empty((global_size, rollout_size, deconstruction_size), np.int64)
    for global_idx in prange(global_size):
        pool = np.empty(num_customers, np.int64)
        np.random.seed(_seed_for(base_seed, _REMOVAL_CHANNEL, global_idx, global_size))
        for r in range(rollout_size):
            _sample_distinct_customers(pool, nodes_to_remove[global_idx, r])
    return nodes_to_remove
