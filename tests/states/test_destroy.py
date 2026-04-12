import torch

from core.states.destroy import CVRPDestroyState, TSPDestroyState


class TestTSPDestroyState:
    def test_build_initial(self) -> None:
        state = TSPDestroyState.build_initial(
            batch_size=2,
            rollout_size=3,
            num_nodes=5,
            device=torch.device("cpu"),
        )

        assert state.action_mask.shape == (2, 3, 5)
        assert (state.action_mask == False).all()
        assert state.last_node_idx is None


class TestCVRPDestroyState:
    def test_build_initial(self) -> None:
        state = CVRPDestroyState.build_initial(
            batch_size=2,
            rollout_size=3,
            num_nodes=5,
            device=torch.device("cpu"),
        )

        assert state.action_mask.shape == (2, 3, 5)
        assert (state.action_mask[:, :, 0] == True).all()
        assert (state.action_mask[:, :, 1:] == False).all()
        assert state.last_node_idx is None

    def test_transition_mask_and_last_node_idx(self) -> None:
        state = CVRPDestroyState.build_initial(
            batch_size=1,
            rollout_size=2,
            num_nodes=6,
            device=torch.device("cpu"),
        )
        action = torch.tensor([[[2], [4]]], dtype=torch.long)

        next_state = state.transition(action)

        assert next_state.last_node_idx is not None
        assert torch.equal(next_state.last_node_idx, action)
        assert next_state.action_mask[0, 0, 2] == True
        assert next_state.action_mask[0, 1, 4] == True
