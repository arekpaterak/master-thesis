"""Numba Esau-Williams construction for CMSTP."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from core.problems._numba_common import dist
from core.problems.cmstp.lns.arrays import CMSTPInstanceArrays

DEPOT = 0
UNASSIGNED = -1


def esau_williams_batch(
    *, locations: np.ndarray, demand: np.ndarray, capacity: np.ndarray
) -> np.ndarray:
    locations = np.ascontiguousarray(locations, dtype=np.float64)
    demand = np.ascontiguousarray(demand, dtype=np.float64)
    capacity = np.ascontiguousarray(capacity, dtype=np.float64)
    return _esau_williams_batch(locations=locations, demand=demand, capacity=capacity)


@njit(parallel=True, cache=True)
def _esau_williams_batch(
    *, locations: np.ndarray, demand: np.ndarray, capacity: np.ndarray
) -> np.ndarray:
    num_instances = locations.shape[0]
    num_nodes = locations.shape[1]
    parent = np.empty((num_instances, num_nodes), np.int64)
    for b in prange(num_instances):
        instance = CMSTPInstanceArrays(locations[b], demand[b], capacity[b])
        parent[b] = _esau_williams_single(instance)
    return parent


@njit(inline="always")
def _find(comp_of: np.ndarray, x: int) -> int:
    root = x
    while comp_of[root] != root:
        root = comp_of[root]
    while comp_of[x] != root:
        nxt = comp_of[x]
        comp_of[x] = root
        x = nxt
    return root


@njit
def _esau_williams_single(instance: CMSTPInstanceArrays) -> np.ndarray:
    locations, demand, capacity = instance
    num_nodes = locations.shape[0]
    num_customers = num_nodes - 1

    comp_of = np.arange(num_nodes)
    comp_demand = np.zeros(num_nodes, np.float64)
    comp_gate_node = np.arange(num_nodes)
    comp_gate_cost = np.zeros(num_nodes, np.float64)
    for i in range(1, num_nodes):
        comp_demand[i] = demand[i]
        comp_gate_cost[i] = dist(locations, i, DEPOT)

    edge_u = np.empty(num_customers, np.int64)
    edge_v = np.empty(num_customers, np.int64)
    num_merge_edges = 0

    # --- EW main loop: repeatedly merge the most-negative-tradeoff pair -------
    while True:
        best_i = -1
        best_j = -1
        best_tradeoff = 0.0  # only strictly-improving (negative) merges are taken
        for i in range(1, num_nodes):
            ci = _find(comp_of, i)
            best_connect_cost = np.inf
            best_connect_j = -1
            for j in range(1, num_nodes):
                cj = _find(comp_of, j)
                if cj == ci:
                    continue
                if comp_demand[ci] + comp_demand[cj] > capacity:
                    continue
                d = dist(locations, i, j)
                if d < best_connect_cost:
                    best_connect_cost = d
                    best_connect_j = j
            if best_connect_j == -1:
                continue
            tradeoff = best_connect_cost - comp_gate_cost[ci]
            if tradeoff < best_tradeoff:
                best_tradeoff = tradeoff
                best_i = i
                best_j = best_connect_j
        if best_i == -1:
            break

        ci = _find(comp_of, best_i)
        cj = _find(comp_of, best_j)
        edge_u[num_merge_edges] = best_i
        edge_v[num_merge_edges] = best_j
        num_merge_edges += 1
        # Drop comp_i's gate; the merged component keeps comp_j's gate to the depot.
        comp_demand[cj] += comp_demand[ci]
        comp_of[ci] = cj

    return _build_parent(num_nodes, comp_of, comp_gate_node, edge_u, edge_v, num_merge_edges)


@njit
def _build_parent(
    num_nodes: int,
    comp_of: np.ndarray,
    comp_gate_node: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    num_merge_edges: int,
) -> np.ndarray:
    """Orient the accumulated undirected edges into a parent array by BFS from the depot."""
    num_edges = num_nodes - 1
    adj_head = np.full(num_nodes, -1, np.int64)
    adj_next = np.empty(2 * num_edges, np.int64)
    adj_to = np.empty(2 * num_edges, np.int64)
    ptr = 0

    for e in range(num_merge_edges):
        a = edge_u[e]
        b = edge_v[e]
        adj_to[ptr] = b
        adj_next[ptr] = adj_head[a]
        adj_head[a] = ptr
        ptr += 1
        adj_to[ptr] = a
        adj_next[ptr] = adj_head[b]
        adj_head[b] = ptr
        ptr += 1

    # Each surviving component contributes its gate edge (depot, gate).
    for r in range(1, num_nodes):
        if _find(comp_of, r) == r:
            gate = comp_gate_node[r]
            adj_to[ptr] = gate
            adj_next[ptr] = adj_head[DEPOT]
            adj_head[DEPOT] = ptr
            ptr += 1
            adj_to[ptr] = DEPOT
            adj_next[ptr] = adj_head[gate]
            adj_head[gate] = ptr
            ptr += 1

    parent = np.full(num_nodes, UNASSIGNED, np.int64)
    visited = np.zeros(num_nodes, np.bool_)
    queue = np.empty(num_nodes, np.int64)
    head = 0
    tail = 0
    queue[tail] = DEPOT
    tail += 1
    visited[DEPOT] = True
    while head < tail:
        u = queue[head]
        head += 1
        e = adj_head[u]
        while e != -1:
            w = adj_to[e]
            if not visited[w]:
                visited[w] = True
                parent[w] = u
                queue[tail] = w
                tail += 1
            e = adj_next[e]
    return parent
