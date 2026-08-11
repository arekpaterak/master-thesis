from __future__ import annotations

import torch

from core.problems.cmstp._esau_williams import esau_williams_batch
from core.problems.cmstp.instance import CMSTPInstance
from core.problems.cmstp.lns.arrays import CMSTPSolutionArrays
from core.problems.cmstp.solution import CMSTPSolution
from core.problems.construction.base import ConstructionHeuristic


def star(instance: CMSTPInstance) -> CMSTPSolution:
    """Every customer a direct child of the depot, each its own subtree."""
    N = instance.num_customers
    batch_size = instance.batch_size
    device = instance.locations.device

    parent = torch.full((*batch_size, N + 1), CMSTPSolution.DEPOT, dtype=torch.long, device=device)
    parent[..., 0] = CMSTPSolution.UNASSIGNED

    subtree_id = (
        torch.arange(N + 1, dtype=torch.long, device=device).expand(*batch_size, N + 1).clone()
    )
    subtree_id[..., 0] = CMSTPSolution.UNASSIGNED

    return CMSTPSolution(
        _instance=instance,
        _parent=parent,
        _subtree_id=subtree_id,
        batch_size=batch_size,
    )


def esau_williams(instance: CMSTPInstance) -> CMSTPSolution:
    """Esau-Williams greedy savings construction."""
    arrays = instance.to_numpy()
    parent = esau_williams_batch(
        locations=arrays.locations, demand=arrays.demand, capacity=arrays.capacity
    )
    return CMSTPSolution.from_numpy(
        instance,
        CMSTPSolutionArrays(parent=parent),
        batch_size=instance.batch_size,
        device=instance.locations.device,
    )


class StarConstruction(ConstructionHeuristic[CMSTPInstance, CMSTPSolution]):
    construction_name = "star"
    problem_name = "cmstp"

    def __call__(self, instance: CMSTPInstance) -> CMSTPSolution:
        return star(instance)


class EsauWilliamsConstruction(ConstructionHeuristic[CMSTPInstance, CMSTPSolution]):
    construction_name = "esau_williams"
    problem_name = "cmstp"

    def __call__(self, instance: CMSTPInstance) -> CMSTPSolution:
        return esau_williams(instance)
