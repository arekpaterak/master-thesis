import torch

from core.states.constructive import TSPConstructiveState


class TestTSPConstructiveState:
    def test_build_initial(self) -> None:
        state = TSPConstructiveState.initial(
            batch_size=2,
            num_nodes=5,
            device=torch.device("cpu"),
        )

        assert state.action_mask.shape == (2, 5)
        assert (state.action_mask == False).all()
        assert state.first_node_idx is None
        assert state.last_node_idx is None

    def test_from_partial_tour(self) -> None:
        partial_tour = torch.tensor([[1, 3], [2, 4]], dtype=torch.long)
        state = TSPConstructiveState.from_partial(partial_tour=partial_tour, num_nodes=5)

        assert state.first_node_idx is not None
        assert state.last_node_idx is not None
        assert torch.equal(state.first_node_idx, partial_tour[:, :1])
        assert torch.equal(state.last_node_idx, partial_tour[:, -1:])
        assert (state.action_mask.gather(-1, partial_tour) == True).all()

    def test_transition(self) -> None:
        state = TSPConstructiveState.initial(
            batch_size=1,
            num_nodes=6,
            device=torch.device("cpu"),
        )
        action = torch.tensor([4], dtype=torch.long)

        next_state = state.transition(action)

        assert next_state.first_node_idx is not None
        assert next_state.last_node_idx is not None
        assert torch.equal(next_state.first_node_idx, torch.tensor([[4]], dtype=torch.long))
        assert torch.equal(next_state.last_node_idx, torch.tensor([[4]], dtype=torch.long))
        assert next_state.action_mask[0, 4] == True
