import pytest
import torch

from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.problems.cvrp import CVRPDistribution, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.lns.operators.repair.insert import CVRPGreedyInsertRepair
from core.problems.cvrp.solve import one_customer_per_route_graph
from tests.core.lns.operators.repair.templates import FeasibleRepairContract


@pytest.fixture(params=[5, 10])
def destruction_size(request) -> int:
    return request.param


@pytest.fixture(params=[1, 4])
def batch_size(request) -> int:
    return request.param


class TestCVRPGreedyInsertRepair(FeasibleRepairContract):
    @pytest.fixture
    def complete_solution(self, batch_size: int) -> CVRPGraphSolution:
        config = CVRPDistribution.for_n(20)
        instance = CVRPSampler(config, seed=42)(batch_size)
        return one_customer_per_route_graph(instance)

    @pytest.fixture
    def deconstructed(
        self, complete_solution: CVRPGraphSolution, destruction_size: int
    ) -> CVRPGraphSolution:
        destroy = RandomNodeDestroy[CVRPGraphSolution](
            destruction_size=destruction_size, seed=42, stitch=True
        )
        return destroy(complete_solution)

    def make_repair(self, *, seed: int | None = None) -> CVRPGreedyInsertRepair:
        return CVRPGreedyInsertRepair(seed=seed)

    # -- CVRP specifics ---------------------------------------------------------

    def test_no_self_loops(self, deconstructed: CVRPGraphSolution) -> None:
        repaired = self.make_repair()(deconstructed)

        customers_next = repaired.next_nodes[..., 1:]
        customer_indices = torch.arange(
            1, repaired.instance.num_customers + 1, device=repaired.instance.locations.device
        )
        assert (customers_next != customer_indices).all(), "No customer should point to itself."

    def test_depot_slot_invariant_preserved(self, deconstructed: CVRPGraphSolution) -> None:
        repaired = self.make_repair()(deconstructed)

        assert (repaired.next_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
        assert (repaired.prev_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()

    def test_rollout_batch_shape(self, batch_size: int) -> None:
        """Repair must work when solution has 2D batch shape (B, R) from remove_customers."""
        B, R, N, K = batch_size, 5, 20, 3
        config = CVRPDistribution.for_n(N)
        instance = CVRPSampler(config, seed=42)(B)
        solution = one_customer_per_route_graph(instance)

        torch.manual_seed(42)
        to_remove = torch.randint(1, N + 1, (B, R, K))
        partial = solution.remove_customers(to_remove, stitch=True)

        assert partial.batch_size == torch.Size([B, R])

        repaired = self.make_repair()(partial)

        assert repaired.batch_size == torch.Size([B, R])
        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()

    def test_all_customers_have_depot_connection(self, deconstructed: CVRPGraphSolution) -> None:
        """Every customer in the repaired solution must have prev and next set."""
        repaired = self.make_repair()(deconstructed)

        assert (repaired.prev_nodes[..., 1:] != CVRPGraphSolution.UNASSIGNED).all()
        assert (repaired.next_nodes[..., 1:] != CVRPGraphSolution.UNASSIGNED).all()

    def test_objective_matches_route_traversal(self, deconstructed: CVRPGraphSolution) -> None:
        repaired = self.make_repair()(deconstructed)

        B = repaired.batch_size[0]
        device = repaired.instance.locations.device
        reference = torch.zeros(B, device=device)

        for b in range(B):
            single = repaired[b]
            locs = single.instance.locations
            depot = torch.zeros(1, dtype=torch.long, device=device)
            for route in single.routes():
                nodes_in_route = torch.cat([depot, route.to(device), depot])
                route_locs = locs[nodes_in_route]
                reference[b] += (route_locs[1:] - route_locs[:-1]).pow(2).sum(-1).sqrt().sum()

        torch.testing.assert_close(repaired.objective(), reference)

    def test_repair_is_stochastic(self, deconstructed: CVRPGraphSolution) -> None:
        """Different seeds must produce different solutions so n_repairs > 1 is meaningful."""
        first = self.make_repair(seed=0)(deconstructed)
        second = self.make_repair(seed=1)(deconstructed)

        assert first.is_complete.all() and second.is_complete.all()
        assert not torch.equal(first.next_nodes, second.next_nodes), (
            "Different seeds produced identical solutions; n_repairs > 1 would be wasted work."
        )

    def test_inserts_at_arbitrary_position(self) -> None:
        """Repair must be able to insert mid-route, not only at depot-connected ends."""
        config = CVRPDistribution(
            num_customers=4, vehicle_capacity_raw=10, min_demand_raw=1, max_demand_raw=1
        )
        instance = CVRPSampler(config, seed=0)(1)
        # Build a route containing 2: 0 -> 1 -> 2 -> 3 -> 4 -> 0
        solution = CVRPGraphSolution.create_from_sequence(
            instance, torch.tensor([[0, 1, 2, 3, 4, 0]], dtype=torch.long)
        )
        free = solution.remove_customers(torch.tensor([[2]]), stitch=True)

        repaired = self.make_repair(seed=0)(free)

        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()
        all_in_one_route = repaired.route_ids[0, 1:].unique().numel() == 1
        assert all_in_one_route, "Single-route insert should not create a new route"

    def test_capacity_respected_via_new_tour_fallback(self) -> None:
        """When every existing route is full, the repair creates a new singleton route."""
        config = CVRPDistribution(
            num_customers=3, vehicle_capacity_raw=2, min_demand_raw=2, max_demand_raw=2
        )
        instance = CVRPSampler(config, seed=0)(1)
        # Each customer has demand 2 == capacity, so each route holds exactly one customer.
        solution = CVRPGraphSolution.create_from_sequence(
            instance, torch.tensor([[0, 1, 0, 2, 0, 3, 0]], dtype=torch.long)
        )
        # Detach customer 2 -- its old route becomes empty (no fragments). All remaining
        # routes are at capacity, so the new-tour fallback must fire.
        partial = solution.remove_customers(torch.tensor([[2]]), stitch=True)

        repaired = self.make_repair(seed=0)(partial)

        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()
        assert repaired.prev_nodes[0, 2].item() == CVRPGraphSolution.DEPOT
        assert repaired.next_nodes[0, 2].item() == CVRPGraphSolution.DEPOT
