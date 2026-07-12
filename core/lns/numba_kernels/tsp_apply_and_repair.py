"""Numba CPU kernels for TSP apply-and-repair.

Mirrors :mod:`core.lns.numba_kernels.cvrp_apply_and_repair`, minus the
depot/capacity/route bookkeeping CVRP needs: TSP has no depot and forms a single
cycle over all nodes, so destroy always stitches directly (no route fragments to
special-case) and repair has no new-route fallback, no capacity check and no
``insert_in_new_routes_only`` concept (there are no routes to restrict insertion to).

Both entry points take externally supplied node sets and, per element, deconstruct
the tour (stitch-remove those nodes) and greedily reconstruct it via best-of-n
cheapest-insertion. They differ only in how the node sets are applied:

- ``apply_and_repair_sequential`` applies a tour's rollouts one after another,
  threading acceptance through a running solution (mirrors ``SequentialImprovementStep``).
- ``apply_and_repair_parallel`` applies every rollout independently from the same
  tour and returns all candidates with their objectives, no acceptance
  (mirrors ``ParallelImprovementStep``; used by training to score rollouts).

Both share :func:`_apply_and_repair_one`, the per-element deconstruct + best-of-n
repair core. The kernels operate on raw numpy ``next``/``prev`` arrays (the linked-list
representation of :class:`TSPGraphSolution`) plus the instance locations, folding the
work into ``njit(parallel=True)`` passes and replacing the hundreds of thousands of
tiny GPU kernel launches the torch path issues for the same work.

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


def apply_and_repair_sequential(
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
    """Apply each tour's rollouts sequentially with acceptance.

    For every element, the rollouts in ``nodes_to_remove`` are applied one after
    another: each deconstructs the running tour, is repaired best-of-n, and replaces
    the running tour when acceptance succeeds.

    When ``first_repair_uses_order`` is set, the first of the ``n_repairs`` greedy
    insertions follows the provided removal order instead of a random order.

    Returns:
        (next_out, prev_out) both (G, N) int64.
    """
    return _sequential_batch(
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


def apply_and_repair_parallel(
    *,
    next: np.ndarray,
    prev: np.ndarray,
    nodes: np.ndarray,
    locations: np.ndarray,
    n_repairs: int,
    base_seed: int,
    first_repair_uses_order: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply every rollout independently from the same tour, without acceptance.

    Each of the ``B`` tours is deconstructed and repaired once per rollout ``k``, all
    from the same starting tour. No acceptance -- every repaired candidate and its
    objective is returned so the caller (e.g. training) can score all rollouts.

    Args:
        next, prev: (B, N) the tour to deconstruct, one per instance.
        nodes: (B, K, M) the K rollouts (node sets to remove) per instance.
        locations: (B, N, 2) instance locations; shared across the K rollouts of an
            instance (not replicated).

    Returns:
        (next_out, prev_out, objective) shaped (B, K, N), (B, K, N), (B, K).
    """
    return _parallel_batch(
        next=np.ascontiguousarray(next, dtype=np.int64),
        prev=np.ascontiguousarray(prev, dtype=np.int64),
        nodes=np.ascontiguousarray(nodes, dtype=np.int64),
        locations=np.ascontiguousarray(locations, dtype=np.float64),
        n_repairs=int(n_repairs),
        base_seed=int(base_seed),
        first_repair_uses_order=bool(first_repair_uses_order),
    )


@njit(parallel=True, cache=True)
def _sequential_batch(
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
        _sequential_single(
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


@njit(parallel=True, cache=True)
def _parallel_batch(
    *,
    next: np.ndarray,
    prev: np.ndarray,
    nodes: np.ndarray,
    locations: np.ndarray,
    n_repairs: int,
    base_seed: int,
    first_repair_uses_order: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_instances = next.shape[0]
    num_rollouts = nodes.shape[1]
    num_nodes = next.shape[1]
    total = num_instances * num_rollouts

    next_out = np.empty((num_instances, num_rollouts, num_nodes), np.int64)
    prev_out = np.empty((num_instances, num_rollouts, num_nodes), np.int64)
    objective = np.empty((num_instances, num_rollouts), np.float64)

    for b in prange(num_instances):
        for k in range(num_rollouts):
            np.random.seed(seed_for(base_seed, b * num_rollouts + k, total))
            objective[b, k] = _apply_and_repair_one(
                current_next=next[b],
                current_prev=prev[b],
                nodes=nodes[b, k],
                locations=locations[b],
                n_repairs=n_repairs,
                first_repair_uses_order=first_repair_uses_order,
                out_next=next_out[b, k],
                out_prev=prev_out[b, k],
            )
    return next_out, prev_out, objective


@njit
def _sequential_single(
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
    """Apply one element's rollouts sequentially, updating the running tour.

    Assumes ``np.random`` has already been reseeded for this element by the caller.
    Each rollout deconstructs and repairs the current ``next_out``/``prev_out`` via
    :func:`_apply_and_repair_one`; the running tour is replaced when accepted.
    """
    num_nodes = next_out.shape[0]
    num_rollouts = nodes_to_remove.shape[0]

    candidate_next = np.empty(num_nodes, np.int64)
    candidate_prev = np.empty(num_nodes, np.int64)

    current_objective = _objective(nxt=next_out, locations=locations)

    for r in range(num_rollouts):
        candidate_objective = _apply_and_repair_one(
            current_next=next_out,
            current_prev=prev_out,
            nodes=nodes_to_remove[r],
            locations=locations,
            n_repairs=n_repairs,
            first_repair_uses_order=first_repair_uses_order,
            out_next=candidate_next,
            out_prev=candidate_prev,
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
            next_out[:] = candidate_next
            prev_out[:] = candidate_prev
            current_objective = candidate_objective


@njit
def _apply_and_repair_one(
    current_next: np.ndarray,
    current_prev: np.ndarray,
    nodes: np.ndarray,
    locations: np.ndarray,
    n_repairs: int,
    first_repair_uses_order: bool,
    out_next: np.ndarray,
    out_prev: np.ndarray,
) -> float:
    """Deconstruct one node set and greedily repair it, best of ``n_repairs``.

    Writes the best repaired tour into ``out_next``/``out_prev`` and returns its
    objective. Reads ``current_next``/``current_prev`` (the tour to deconstruct)
    without mutating them, so callers may apply many node sets to the same tour.
    No acceptance -- callers decide what to do with the result.
    """
    num_nodes = current_next.shape[0]
    deconstruction_size = nodes.shape[0]

    deconstructed_next = np.empty(num_nodes, np.int64)
    deconstructed_prev = np.empty(num_nodes, np.int64)
    repaired_next = np.empty(num_nodes, np.int64)
    repaired_prev = np.empty(num_nodes, np.int64)

    deconstructed_next[:] = current_next
    deconstructed_prev[:] = current_prev
    for m in range(deconstruction_size):
        c = nodes[m]
        if deconstructed_prev[c] == UNASSIGNED and deconstructed_next[c] == UNASSIGNED:
            continue
        _remove_node_and_stitch(deconstructed_next, deconstructed_prev, c)

    best_repaired_objective = np.inf
    for i in range(n_repairs):
        repaired_next[:] = deconstructed_next
        repaired_prev[:] = deconstructed_prev

        order = None
        if first_repair_uses_order and i == 0:
            order = nodes
        _greedy_insert_repair(
            nxt=repaired_next, prv=repaired_prev, locations=locations, order=order
        )
        objective = _objective(nxt=repaired_next, locations=locations)
        if objective < best_repaired_objective:
            best_repaired_objective = objective
            out_next[:] = repaired_next
            out_prev[:] = repaired_prev

    return best_repaired_objective


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
