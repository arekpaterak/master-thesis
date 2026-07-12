"""Numba CPU kernels for the sequential TSP improvement step.

Mirrors :mod:`core.lns.numba_kernels.cvrp_sequential_improvement_step`, minus the
depot/capacity/route bookkeeping CVRP needs: TSP has no depot and forms a single
cycle over all nodes, so destroy always stitches directly (no route fragments to
special-case) and repair has no new-route fallback, no capacity check and no
``insert_in_new_routes_only`` concept (there are no routes to restrict insertion to).

The kernels operate on raw numpy ``next``/``prev`` arrays (the linked-list
representation of :class:`TSPGraphSolution`) plus the instance locations. They fold
the whole rollout loop -- stitch-destroy, best-of-n greedy cheapest-insertion and
acceptance -- into a single ``njit(parallel=True)`` pass over the flattened batch,
replacing the hundreds of thousands of tiny GPU kernel launches the torch path
issues for the same work.

Randomness uses numba's per-thread ``np.random``. To stay reproducible under
``prange`` (where draws would otherwise depend on the thread count), each parallel
iteration reseeds ``np.random`` from a seed derived only from ``base_seed`` and the
iteration index -- see :func:`core.lns.numba_kernels.common.seed_for`.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .common import dist, seed_for

UNASSIGNED = -1


def run_with_removals(
    *,
    next: np.ndarray,
    prev: np.ndarray,
    nodes_to_remove: np.ndarray,
    locations: np.ndarray,
    n_repairs: int,
    accept_kind: int,
    temperature: float,
    eps: float,
    base_seed: int,
    first_repair_uses_order: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential step with externally supplied nodes to remove.

    When ``first_repair_uses_order`` is set, the first of the ``n_repairs`` greedy
    insertions follows the provided removal order instead of a random order.

    Returns:
        (next_out, prev_out) both (G, N) int64.
    """
    return _run_for_batch(
        next=np.ascontiguousarray(next, dtype=np.int64),
        prev=np.ascontiguousarray(prev, dtype=np.int64),
        nodes_to_remove=np.ascontiguousarray(nodes_to_remove, dtype=np.int64),
        locations=np.ascontiguousarray(locations, dtype=np.float64),
        n_repairs=int(n_repairs),
        accept_kind=int(accept_kind),
        temperature=float(temperature),
        eps=float(eps),
        base_seed=int(base_seed),
        first_repair_uses_order=bool(first_repair_uses_order),
    )


@njit(parallel=True, cache=True)
def _run_for_batch(
    *,
    next: np.ndarray,
    prev: np.ndarray,
    nodes_to_remove: np.ndarray,
    locations: np.ndarray,
    n_repairs: int,
    accept_kind: int,
    temperature: float,
    eps: float,
    base_seed: int,
    first_repair_uses_order: bool,
) -> tuple[np.ndarray, np.ndarray]:
    global_size = next.shape[0]
    next_out = next.copy()
    prev_out = prev.copy()
    for global_idx in prange(global_size):
        np.random.seed(seed_for(base_seed, global_idx, global_size))
        _run_for_single(
            next_out=next_out[global_idx],
            prev_out=prev_out[global_idx],
            nodes_to_remove=nodes_to_remove[global_idx],
            locations=locations[global_idx],
            n_repairs=n_repairs,
            accept_kind=accept_kind,
            temperature=temperature,
            eps=eps,
            first_repair_uses_order=first_repair_uses_order,
        )
    return next_out, prev_out


@njit
def _run_for_single(
    next_out: np.ndarray,
    prev_out: np.ndarray,
    nodes_to_remove: np.ndarray,
    locations: np.ndarray,
    n_repairs: int,
    accept_kind: int,
    temperature: float,
    eps: float,
    first_repair_uses_order: bool,
) -> None:
    """Run the full sequential rollout loop for one batch element.

    Assumes ``np.random`` has already been reseeded for this element by the caller.
    """
    num_nodes = next_out.shape[0]
    num_rollouts = nodes_to_remove.shape[0]
    deconstruction_size = nodes_to_remove.shape[1]

    deconstructed_next = np.empty(num_nodes, np.int64)
    deconstructed_prev = np.empty(num_nodes, np.int64)
    repaired_next = np.empty(num_nodes, np.int64)
    repaired_prev = np.empty(num_nodes, np.int64)
    best_repaired_next = np.empty(num_nodes, np.int64)
    best_repaired_prev = np.empty(num_nodes, np.int64)

    current_objective = _objective(nxt=next_out, locations=locations)

    for r in range(num_rollouts):
        deconstructed_next[:] = next_out
        deconstructed_prev[:] = prev_out
        for m in range(deconstruction_size):
            c = nodes_to_remove[r, m]
            if deconstructed_prev[c] == UNASSIGNED and deconstructed_next[c] == UNASSIGNED:
                continue
            _remove_node_and_stitch(deconstructed_next, deconstructed_prev, c)

        best_repaired_objective = np.inf
        for i in range(n_repairs):
            repaired_next[:] = deconstructed_next
            repaired_prev[:] = deconstructed_prev
            order = None
            if first_repair_uses_order and i == 0:
                order = nodes_to_remove[r]
            _greedy_insert_repair(
                nxt=repaired_next, prv=repaired_prev, locations=locations, order=order
            )
            objective = _objective(nxt=repaired_next, locations=locations)
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


# --- graph helpers (single element, length N arrays) -------------------------


@njit(inline="always")
def _remove_node_and_stitch(nxt: np.ndarray, prv: np.ndarray, node: int) -> None:
    """Detach node and stitch its neighbours. Assumes node is currently placed."""
    u = prv[node]
    v = nxt[node]
    nxt[u] = v
    prv[v] = u
    prv[node] = UNASSIGNED
    nxt[node] = UNASSIGNED


@njit
def _objective(nxt: np.ndarray, locations: np.ndarray) -> float:
    """Total tour length. Requires a complete tour (no UNASSIGNED entries)."""
    total_distance = 0.0
    for i in range(nxt.shape[0]):
        total_distance += dist(locations, i, nxt[i])
    return total_distance


@njit
def _pick_random_free_node(nxt: np.ndarray, prv: np.ndarray) -> int:
    """Pick a random free node, or -1 if none are free."""
    chosen = -1
    best_score = -1.0
    for c in range(nxt.shape[0]):
        score = np.random.random()
        if prv[c] == UNASSIGNED and nxt[c] == UNASSIGNED and score > best_score:
            best_score = score
            chosen = c
    return chosen


@njit
def _insert_node_at_cheapest(
    node: int,
    nxt: np.ndarray,
    prv: np.ndarray,
    locations: np.ndarray,
) -> None:
    """Insert free ``node`` at its cheapest valid position. In-place.

    Candidate positions are every existing edge ``(prev[v], v)`` for placed nodes
    ``v`` (``prev[v] != UNASSIGNED``); free nodes are naturally skipped since their
    ``prev`` is UNASSIGNED. At least one such edge always exists as long as one node
    survives destroy (a single surviving node self-loops, giving a valid edge).
    """
    best_cost = np.inf
    best_u = -1
    best_v = -1
    for v in range(nxt.shape[0]):
        if prv[v] == UNASSIGNED:
            continue
        u = prv[v]
        cost = dist(locations, u, node) + dist(locations, node, v) - dist(locations, u, v)
        if cost < best_cost:
            best_cost = cost
            best_u = u
            best_v = v

    prv[node] = best_u
    nxt[node] = best_v
    nxt[best_u] = node
    prv[best_v] = node


@njit
def _greedy_insert_repair(
    nxt: np.ndarray,
    prv: np.ndarray,
    locations: np.ndarray,
    order: np.ndarray | None = None,
) -> None:
    """Insert every free node at its cheapest valid position. In-place.

    When ``order`` is given (the model rollout removal order for this rollout),
    free nodes are inserted in that order first, mirroring the original NDS
    behaviour where the first repair follows the model order. Any nodes not
    covered by ``order`` are then drained in random order. When ``order`` is
    omitted all nodes are inserted in random order.
    """
    if order is not None:
        for k in range(order.shape[0]):
            node = order[k]
            if prv[node] == UNASSIGNED and nxt[node] == UNASSIGNED:
                _insert_node_at_cheapest(node, nxt, prv, locations)

    while True:
        node = _pick_random_free_node(nxt, prv)
        if node == -1:
            break
        _insert_node_at_cheapest(node, nxt, prv, locations)
