"""Numba CPU apply-and-repair for CMSTP."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from core.problems._numba_common import accept_move, dist, seed_for
from core.problems.cmstp.arrays import CMSTPInstanceArrays, CMSTPSolutionArrays

DEPOT = 0
UNASSIGNED = -1


def apply_and_repair_parallel(
    solution: CMSTPSolutionArrays,
    instance: CMSTPInstanceArrays,
    nodes: np.ndarray,
    *,
    n_repairs: int,
    base_seed: int,
    first_repair_uses_order: bool = False,
) -> tuple[CMSTPSolutionArrays, np.ndarray]:
    """Apply every rollout independently from the same solution, without acceptance.

    Args:
        solution: (G, N+1) the tree to deconstruct, one per instance.
        instance: instance arrays (G, N+1, 2)/(G, N+1)/(G,); shared across the K
            rollouts of an instance (not replicated).
        nodes: (G, K, M) the K rollouts (customer sets to remove) per instance.

    Returns:
        (CMSTPSolutionArrays, objective) shaped (G, K, N+1) and (G, K).
    """
    parent_out, objective = _parallel_batch(
        parent=solution.parent,
        nodes=nodes,
        locations=instance.locations,
        demand=instance.demand,
        capacity=instance.capacity,
        n_repairs=int(n_repairs),
        base_seed=int(base_seed),
        first_repair_uses_order=bool(first_repair_uses_order),
    )
    return CMSTPSolutionArrays(parent=parent_out), objective


def apply_and_repair_sequential(
    solution: CMSTPSolutionArrays,
    instance: CMSTPInstanceArrays,
    nodes_to_remove: np.ndarray,
    *,
    n_repairs: int,
    accept_kind: int,
    temperature: float,
    eps: float,
    base_seed: int,
    first_repair_uses_order: bool = False,
) -> CMSTPSolutionArrays:
    """Apply each solution's rollouts sequentially with acceptance.

    For every element, the rollouts in ``nodes_to_remove`` (``(G, R, M)`` int64) are
    applied one after another: each deconstructs the running solution, is repaired
    best-of-n, and replaces the running solution when acceptance succeeds.
    """
    parent_out = _sequential_batch(
        parent=solution.parent,
        nodes_to_remove=nodes_to_remove,
        locations=instance.locations,
        demand=instance.demand,
        capacity=instance.capacity,
        n_repairs=int(n_repairs),
        accept_kind=int(accept_kind),
        temperature=float(temperature),
        eps=float(eps),
        base_seed=int(base_seed),
        first_repair_uses_order=bool(first_repair_uses_order),
    )
    return CMSTPSolutionArrays(parent=parent_out)


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
        instance = CMSTPInstanceArrays(locations[b], demand[b], capacity[b])
        current = CMSTPSolutionArrays(parent[b])
        for k in range(num_rollouts):
            np.random.seed(seed_for(base_seed, b * num_rollouts + k, total))
            objective[b, k] = _apply_and_repair_one(
                current=current,
                nodes=nodes[b, k],
                instance=instance,
                n_repairs=n_repairs,
                first_repair_uses_order=first_repair_uses_order,
                out=CMSTPSolutionArrays(parent_out[b, k]),
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
        _sequential_single_(
            solution=CMSTPSolutionArrays(parent_out[global_idx]),
            nodes_to_remove=nodes_to_remove[global_idx],
            instance=CMSTPInstanceArrays(
                locations[global_idx], demand[global_idx], capacity[global_idx]
            ),
            n_repairs=n_repairs,
            accept_kind=accept_kind,
            temperature=temperature,
            eps=eps,
            first_repair_uses_order=first_repair_uses_order,
        )
    return parent_out


@njit
def _sequential_single_(
    solution: CMSTPSolutionArrays,
    nodes_to_remove: np.ndarray,
    instance: CMSTPInstanceArrays,
    n_repairs: int,
    accept_kind: int,
    temperature: float,
    eps: float,
    first_repair_uses_order: bool,
) -> None:
    """Apply one element's rollouts sequentially, updating the running solution.

    Assumes ``np.random`` has already been reseeded for this element by the caller.
    """
    parent = solution.parent
    num_nodes = parent.shape[0]
    candidate = CMSTPSolutionArrays(np.empty(num_nodes, np.int64))

    current_objective = _objective(solution, instance)

    num_rollouts = nodes_to_remove.shape[0]
    for r in range(num_rollouts):
        candidate_objective = _apply_and_repair_one(
            current=solution,
            nodes=nodes_to_remove[r],
            instance=instance,
            n_repairs=n_repairs,
            first_repair_uses_order=first_repair_uses_order,
            out=candidate,
        )

        accepted = accept_move(
            candidate_objective, current_objective, accept_kind, temperature, eps
        )

        if accepted:
            parent[:] = candidate.parent
            current_objective = candidate_objective


@njit
def _apply_and_repair_one(
    current: CMSTPSolutionArrays,
    nodes: np.ndarray,
    instance: CMSTPInstanceArrays,
    n_repairs: int,
    first_repair_uses_order: bool,
    out: CMSTPSolutionArrays,
) -> float:
    """Deconstruct one customer set and greedily repair it, best of ``n_repairs``.

    Writes the best repaired tree into ``out`` and returns its objective. Reads
    ``current`` (the tree to deconstruct) without mutating it.
    """
    current_parent = current.parent
    num_nodes = current_parent.shape[0]

    is_removed = np.zeros(num_nodes, np.bool_)
    for m in range(nodes.shape[0]):
        c = nodes[m]
        if c >= 1:
            is_removed[c] = True

    deconstructed_parent = _deconstruct(current_parent, is_removed)

    repaired = CMSTPSolutionArrays(np.empty(num_nodes, np.int64))
    subtree_root_of = np.empty(num_nodes, np.int64)
    subtree_demand = np.empty(num_nodes, np.float64)

    best_objective = np.inf
    for i in range(n_repairs):
        repaired.parent[:] = deconstructed_parent
        _init_subtrees_(repaired, instance, subtree_root_of, subtree_demand)

        order = None
        if first_repair_uses_order and i == 0:
            order = nodes
        _greedy_attach_repair_(repaired, instance, subtree_root_of, subtree_demand, order)

        objective = _objective(repaired, instance)
        if objective < best_objective:
            best_objective = objective
            out.parent[:] = repaired.parent

    return best_objective


# --- graph helpers (single element, length N+1 arrays) ----------------------


@njit(inline="always")
def _is_free(solution: CMSTPSolutionArrays, node: int) -> bool:
    """Whether ``node`` is a free customer (not attached to any parent)."""
    return solution.parent[node] == UNASSIGNED


@njit
def _deconstruct(parent: np.ndarray, is_removed: np.ndarray) -> np.ndarray:
    """Stitch-remove the marked customers.

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
        if p <= DEPOT or not is_removed[p]:
            continue
        # Climb the original parents past removed ancestors to the nearest survivor.
        while p > DEPOT and is_removed[p]:
            p = parent[p]
        out[v] = p if p > DEPOT else DEPOT
    return out


@njit
def _init_subtrees_(
    solution: CMSTPSolutionArrays,
    instance: CMSTPInstanceArrays,
    subtree_root_of: np.ndarray,
    subtree_demand: np.ndarray,
) -> None:
    """Fill ``subtree_root_of[v]`` (v's direct-child-of-depot ancestor) and ``subtree_demand[r]``.

    Free customers (parent == UNASSIGNED) get ``subtree_root_of = UNASSIGNED`` and are skipped;
    they receive a subtree at attach time. ``subtree_demand`` is indexed by subtree root node.
    """
    parent = solution.parent
    demand = instance.demand
    num_nodes = parent.shape[0]
    for i in range(num_nodes):
        subtree_root_of[i] = UNASSIGNED
        subtree_demand[i] = 0.0
    for v in range(1, num_nodes):
        if _is_free(solution, v):
            continue
        node = v
        while parent[node] > DEPOT:
            node = parent[node]
        subtree_root_of[v] = node  # node is a direct child of the depot
    for v in range(1, num_nodes):
        if subtree_root_of[v] != UNASSIGNED:
            subtree_demand[subtree_root_of[v]] += demand[v]


@njit
def _pick_random_free_customer(solution: CMSTPSolutionArrays) -> int:
    """Pick a random free customer (parent == UNASSIGNED), or -1 if none are free."""
    num_nodes = solution.parent.shape[0]
    chosen = -1
    best_score = -1.0
    for t in range(1, num_nodes):
        if _is_free(solution, t):
            score = np.random.random()
            if score > best_score:
                best_score = score
                chosen = t
    return chosen


@njit
def _attach_customer_at_cheapest_(
    customer: int,
    solution: CMSTPSolutionArrays,
    instance: CMSTPInstanceArrays,
    subtree_root_of: np.ndarray,
    subtree_demand: np.ndarray,
) -> None:
    """Attach free ``customer`` to its cheapest capacity-feasible parent. In-place.

    Attaching to the depot starts a new subtree (demand of a single customer, always
    feasible), so the argmin is always defined. Attaching under an assigned node adds the
    customer's demand to that node's subtree, allowed only within capacity.
    """
    parent = solution.parent
    locations, demand, capacity = instance
    num_nodes = parent.shape[0]
    demand_c = demand[customer]

    # Depot fallback: a new direct-child subtree holding only this customer.
    best_cost = dist(locations, DEPOT, customer)
    best_parent = DEPOT

    for p in range(1, num_nodes):
        if subtree_root_of[p] == UNASSIGNED:
            continue  # free customer, not a valid parent
        r = subtree_root_of[p]
        if subtree_demand[r] + demand_c <= capacity:
            cost = dist(locations, p, customer)
            if cost < best_cost:
                best_cost = cost
                best_parent = p

    parent[customer] = best_parent
    if best_parent == DEPOT:
        subtree_root_of[customer] = customer
        subtree_demand[customer] = demand_c
    else:
        r = subtree_root_of[best_parent]
        subtree_root_of[customer] = r
        subtree_demand[r] += demand_c


@njit
def _greedy_attach_repair_(
    solution: CMSTPSolutionArrays,
    instance: CMSTPInstanceArrays,
    subtree_root_of: np.ndarray,
    subtree_demand: np.ndarray,
    order: np.ndarray | None = None,
) -> None:
    """Attach every free customer at its cheapest valid parent. In-place.

    When ``order`` is given, free customers are attached in that order first (the model
    rollout removal order), then any remainder in random order. Mirrors the CVRP greedy
    insert repair.
    """
    if order is not None:
        for k in range(order.shape[0]):
            customer = order[k]
            if customer >= 1 and _is_free(solution, customer):
                _attach_customer_at_cheapest_(
                    customer, solution, instance, subtree_root_of, subtree_demand
                )

    while True:
        customer = _pick_random_free_customer(solution)
        if customer == -1:
            break
        _attach_customer_at_cheapest_(customer, solution, instance, subtree_root_of, subtree_demand)


@njit
def _objective(solution: CMSTPSolutionArrays, instance: CMSTPInstanceArrays) -> float:
    """Total length of all tree edges (each customer to its parent)."""
    parent = solution.parent
    locations = instance.locations
    num_nodes = parent.shape[0]
    total_distance = 0.0
    for t in range(1, num_nodes):
        p = parent[t]
        if p >= DEPOT:
            total_distance += dist(locations, t, p)
    return total_distance
