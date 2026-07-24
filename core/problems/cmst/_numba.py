"""Numba CPU apply-and-repair for CMST.

Both entry points take externally supplied node sets and, per element, deconstruct the
solution (stitch-remove those terminals, reattaching their children to the nearest
surviving ancestor) and greedily reconstruct it via best-of-n cheapest attach. They
differ only in how the node sets are applied:

- ``apply_and_repair_sequential`` applies a solution's rollouts one after another,
  threading acceptance through a running solution (mirrors ``SequentialImprovementStep``).
- ``apply_and_repair_parallel`` applies every rollout independently from the same
  solution and returns all candidates with their objectives, no acceptance
  (mirrors ``ParallelImprovementStep``; used by training to score rollouts).

Both share :func:`_apply_and_repair_one`, the per-element deconstruct + best-of-n repair
core. The functions operate on raw numpy ``parent`` arrays (the parent-pointer
representation of :class:`CMSTSolution`) plus the instance arrays, folding the work into
``njit(parallel=True)`` passes -- the CMST analog of
:mod:`core.lns.numba_kernels.cvrp_apply_and_repair`, with ``ROOT``/``parent`` in place of
``DEPOT``/``next``/``prev`` and no route concept.

``subtree_id`` is intentionally NOT maintained here; the caller recomputes it once via
:meth:`CMSTSolution._compute_subtree_ids`. The greedy attach mirrors
:class:`core.lns.operators.repair.insert.cmst.CMSTGreedyInsertRepair` (random free
terminal, then cheapest capacity-feasible parent, with an always-valid attach-to-root
fallback).

Randomness uses numba's per-thread ``np.random``, reseeded per iteration from
``base_seed`` and the iteration index for reproducibility under ``prange`` -- see
:func:`seed_for`.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from core.lns.numba_kernels.common import dist, seed_for

ROOT = 0
UNASSIGNED = -1


def apply_and_repair_parallel(
    *,
    parent: np.ndarray,
    nodes: np.ndarray,
    locations: np.ndarray,
    demand: np.ndarray,
    capacity: np.ndarray,
    n_repairs: int,
    base_seed: int,
    first_repair_uses_order: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply every rollout independently from the same solution, without acceptance.

    Args:
        parent: (B, N+1) the tree to deconstruct, one per instance.
        nodes: (B, K, M) the K rollouts (terminal sets to remove) per instance.
        locations, demand, capacity: instance arrays, (B, N+1, 2)/(B, N+1)/(B,);
            shared across the K rollouts of an instance (not replicated).

    Returns:
        (parent_out, objective) shaped (B, K, N+1) and (B, K).
    """
    return _parallel_batch(
        parent=np.ascontiguousarray(parent, dtype=np.int64),
        nodes=np.ascontiguousarray(nodes, dtype=np.int64),
        locations=np.ascontiguousarray(locations, dtype=np.float64),
        demand=np.ascontiguousarray(demand, dtype=np.float64),
        capacity=np.ascontiguousarray(capacity, dtype=np.float64),
        n_repairs=int(n_repairs),
        base_seed=int(base_seed),
        first_repair_uses_order=bool(first_repair_uses_order),
    )


def apply_and_repair_sequential(
    *,
    parent: np.ndarray,
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
) -> np.ndarray:
    """Apply each solution's rollouts sequentially with acceptance.

    For every element, the rollouts in ``nodes_to_remove`` are applied one after another:
    each deconstructs the running solution, is repaired best-of-n, and replaces the
    running solution when acceptance succeeds.

    Returns:
        parent_out (G, N+1) int64.
    """
    return _sequential_batch(
        parent=np.ascontiguousarray(parent, dtype=np.int64),
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
    )


@njit(parallel=True, cache=True)
def _parallel_batch(
    *,
    parent: np.ndarray,
    nodes: np.ndarray,
    locations: np.ndarray,
    demand: np.ndarray,
    capacity: np.ndarray,
    n_repairs: int,
    base_seed: int,
    first_repair_uses_order: bool,
) -> tuple[np.ndarray, np.ndarray]:
    num_instances = parent.shape[0]
    num_rollouts = nodes.shape[1]
    num_nodes = parent.shape[1]
    total = num_instances * num_rollouts

    parent_out = np.empty((num_instances, num_rollouts, num_nodes), np.int64)
    objective = np.empty((num_instances, num_rollouts), np.float64)

    for b in prange(num_instances):
        for k in range(num_rollouts):
            np.random.seed(seed_for(base_seed, b * num_rollouts + k, total))
            objective[b, k] = _apply_and_repair_one(
                current_parent=parent[b],
                nodes=nodes[b, k],
                locations=locations[b],
                demand=demand[b],
                capacity=capacity[b],
                n_repairs=n_repairs,
                first_repair_uses_order=first_repair_uses_order,
                out_parent=parent_out[b, k],
            )
    return parent_out, objective


@njit(parallel=True, cache=True)
def _sequential_batch(
    *,
    parent: np.ndarray,
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
) -> np.ndarray:
    global_size = parent.shape[0]
    parent_out = parent.copy()
    for global_idx in prange(global_size):
        np.random.seed(seed_for(base_seed, global_idx, global_size))
        _sequential_single(
            parent_out=parent_out[global_idx],
            nodes_to_remove=nodes_to_remove[global_idx],
            locations=locations[global_idx],
            demand=demand[global_idx],
            capacity=capacity[global_idx],
            n_repairs=n_repairs,
            accept_kind=accept_kind,
            temperature=temperature,
            eps=eps,
            first_repair_uses_order=first_repair_uses_order,
        )
    return parent_out


@njit
def _sequential_single(
    parent_out: np.ndarray,
    nodes_to_remove: np.ndarray,
    locations: np.ndarray,
    demand: np.ndarray,
    capacity: float,
    n_repairs: int,
    accept_kind: int,
    temperature: float,
    eps: float,
    first_repair_uses_order: bool,
) -> None:
    """Apply one element's rollouts sequentially, updating the running solution.

    Assumes ``np.random`` has already been reseeded for this element by the caller.
    """
    num_nodes = parent_out.shape[0]
    candidate_parent = np.empty(num_nodes, np.int64)

    current_objective = _objective(parent_out, locations)

    num_rollouts = nodes_to_remove.shape[0]
    for r in range(num_rollouts):
        candidate_objective = _apply_and_repair_one(
            current_parent=parent_out,
            nodes=nodes_to_remove[r],
            locations=locations,
            demand=demand,
            capacity=capacity,
            n_repairs=n_repairs,
            first_repair_uses_order=first_repair_uses_order,
            out_parent=candidate_parent,
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
            parent_out[:] = candidate_parent
            current_objective = candidate_objective


@njit
def _apply_and_repair_one(
    current_parent: np.ndarray,
    nodes: np.ndarray,
    locations: np.ndarray,
    demand: np.ndarray,
    capacity: float,
    n_repairs: int,
    first_repair_uses_order: bool,
    out_parent: np.ndarray,
) -> float:
    """Deconstruct one terminal set and greedily repair it, best of ``n_repairs``.

    Writes the best repaired tree into ``out_parent`` and returns its objective. Reads
    ``current_parent`` (the tree to deconstruct) without mutating it.
    """
    num_nodes = current_parent.shape[0]

    is_removed = np.zeros(num_nodes, np.bool_)
    for m in range(nodes.shape[0]):
        c = nodes[m]
        if c >= 1:
            is_removed[c] = True

    deconstructed_parent = _deconstruct(current_parent, is_removed)

    repaired_parent = np.empty(num_nodes, np.int64)
    root_of = np.empty(num_nodes, np.int64)
    subtree_demand = np.empty(num_nodes, np.float64)

    best_objective = np.inf
    for i in range(n_repairs):
        repaired_parent[:] = deconstructed_parent
        _init_subtrees(repaired_parent, demand, root_of, subtree_demand)

        order = None
        if first_repair_uses_order and i == 0:
            order = nodes
        _greedy_attach_repair(
            parent=repaired_parent,
            locations=locations,
            demand=demand,
            capacity=capacity,
            root_of=root_of,
            subtree_demand=subtree_demand,
            order=order,
        )

        objective = _objective(repaired_parent, locations)
        if objective < best_objective:
            best_objective = objective
            out_parent[:] = repaired_parent

    return best_objective


# --- graph helpers (single element, length N+1 arrays) ----------------------


@njit
def _deconstruct(parent: np.ndarray, is_removed: np.ndarray) -> np.ndarray:
    """Stitch-remove the marked terminals.

    Non-removed nodes whose parent is removed are reattached to their nearest surviving
    ancestor (the grandparent generalisation); removed nodes are freed for reinsertion.
    Climbing uses the original ``parent`` so the survivor lookup is independent of edit
    order.
    """
    num_nodes = parent.shape[0]
    out = parent.copy()
    for v in range(1, num_nodes):
        if is_removed[v]:
            out[v] = UNASSIGNED
            continue
        p = parent[v]
        if p <= ROOT or not is_removed[p]:
            continue
        # Climb the original parents past removed ancestors to the nearest survivor.
        while p > ROOT and is_removed[p]:
            p = parent[p]
        out[v] = p if p > ROOT else ROOT
    return out


@njit
def _init_subtrees(
    parent: np.ndarray,
    demand: np.ndarray,
    root_of: np.ndarray,
    subtree_demand: np.ndarray,
) -> None:
    """Fill ``root_of[v]`` (v's direct-child-of-root ancestor) and ``subtree_demand[r]``.

    Free terminals (parent == UNASSIGNED) get ``root_of = UNASSIGNED`` and are skipped;
    they receive a subtree at attach time. ``subtree_demand`` is indexed by root node.
    """
    num_nodes = parent.shape[0]
    for i in range(num_nodes):
        root_of[i] = UNASSIGNED
        subtree_demand[i] = 0.0
    for v in range(1, num_nodes):
        if parent[v] == UNASSIGNED:
            continue
        node = v
        while parent[node] > ROOT:
            node = parent[node]
        root_of[v] = node  # node is a direct child of the root
    for v in range(1, num_nodes):
        if root_of[v] != UNASSIGNED:
            subtree_demand[root_of[v]] += demand[v]


@njit
def _pick_random_free_terminal(parent: np.ndarray) -> int:
    """Pick a random free terminal (parent == UNASSIGNED), or -1 if none are free."""
    num_nodes = parent.shape[0]
    chosen = -1
    best_score = -1.0
    for t in range(1, num_nodes):
        if parent[t] == UNASSIGNED:
            score = np.random.random()
            if score > best_score:
                best_score = score
                chosen = t
    return chosen


@njit
def _attach_terminal_at_cheapest(
    terminal: int,
    parent: np.ndarray,
    locations: np.ndarray,
    demand: np.ndarray,
    capacity: float,
    root_of: np.ndarray,
    subtree_demand: np.ndarray,
) -> None:
    """Attach free ``terminal`` to its cheapest capacity-feasible parent. In-place.

    Attaching to the root starts a new subtree (demand of a single terminal, always
    feasible), so the argmin is always defined. Attaching under an assigned node adds the
    terminal's demand to that node's subtree, allowed only within capacity.
    """
    num_nodes = parent.shape[0]
    demand_t = demand[terminal]

    # Root fallback: a new direct-child subtree holding only this terminal.
    best_cost = dist(locations, ROOT, terminal)
    best_parent = ROOT

    for p in range(1, num_nodes):
        if root_of[p] == UNASSIGNED:
            continue  # free terminal, not a valid parent
        r = root_of[p]
        if subtree_demand[r] + demand_t <= capacity:
            cost = dist(locations, p, terminal)
            if cost < best_cost:
                best_cost = cost
                best_parent = p

    parent[terminal] = best_parent
    if best_parent == ROOT:
        root_of[terminal] = terminal
        subtree_demand[terminal] = demand_t
    else:
        r = root_of[best_parent]
        root_of[terminal] = r
        subtree_demand[r] += demand_t


@njit
def _greedy_attach_repair(
    parent: np.ndarray,
    locations: np.ndarray,
    demand: np.ndarray,
    capacity: float,
    root_of: np.ndarray,
    subtree_demand: np.ndarray,
    order: np.ndarray | None = None,
) -> None:
    """Attach every free terminal at its cheapest valid parent. In-place.

    When ``order`` is given, free terminals are attached in that order first (the model
    rollout removal order), then any remainder in random order. Mirrors the CVRP greedy
    insert repair.
    """
    if order is not None:
        for k in range(order.shape[0]):
            terminal = order[k]
            if terminal >= 1 and parent[terminal] == UNASSIGNED:
                _attach_terminal_at_cheapest(
                    terminal, parent, locations, demand, capacity, root_of, subtree_demand
                )

    while True:
        terminal = _pick_random_free_terminal(parent)
        if terminal == -1:
            break
        _attach_terminal_at_cheapest(
            terminal, parent, locations, demand, capacity, root_of, subtree_demand
        )


@njit
def _objective(parent: np.ndarray, locations: np.ndarray) -> float:
    """Total length of all tree edges (each terminal to its parent)."""
    num_nodes = parent.shape[0]
    total_distance = 0.0
    for t in range(1, num_nodes):
        p = parent[t]
        if p >= ROOT:
            total_distance += dist(locations, t, p)
    return total_distance
