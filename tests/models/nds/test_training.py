import torch

from core.models.nds.functions import (
    noop_improvement_step,
    noop_repair,
    only_expanding_apply_rollout,
)
from core.models.nds.model import NDSModel
from core.models.nds.training import train
from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.cvrp.solve import one_customer_per_route


def test_train():
    B, R, N = 4, 10, 20
    sampler = CVRPSampler(CVRPConfig.for_n(N))
    model = NDSModel()
    optimizer = torch.optim.AdamW(model.parameters())

    train(
        model=model,
        optimizer=optimizer,
        num_iterations=3,
        rollouts_per_solution=R,
        num_initial_improvement_steps=10,
        deconstruction_size=15,
        max_epochs=2,
        generate_instance=lambda: sampler(B),
        generate_start_solution=lambda x: one_customer_per_route(x),
        improvement_step=noop_improvement_step,
        apply_rollout=only_expanding_apply_rollout,
        repair=noop_repair,
    )
