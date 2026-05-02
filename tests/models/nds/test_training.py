import torch

from core.models.nds.functions import (
    noop_improvement_step,
    noop_repair,
    only_expanding_apply_rollout,
)
from core.models.nds.model import NDSModel
from core.models.nds.training import configure_train_epoch
from core.models.train import Trainer
from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.cvrp.solve import one_customer_per_route


def test_train():
    B, R, N = 4, 10, 20
    sampler = CVRPSampler(CVRPConfig.for_n(N))
    model = NDSModel(problem_name="cvrp")
    optimizer = torch.optim.AdamW(model.parameters())

    train_epoch = configure_train_epoch(
        instance_sampler=sampler,
        batches_per_epoch=2,
        batch_size=B,
        num_main_improvement_steps=3,
        rollouts_per_solution=R,
        num_initial_improvement_steps=10,
        deconstruction_size=15,
        generate_start_solution=lambda x: one_customer_per_route(x),
        improvement_step=noop_improvement_step,
        apply_rollout=only_expanding_apply_rollout,
        repair=noop_repair,
    )

    history = Trainer(
        max_epochs=2,
        train_epoch=train_epoch,
    ).train(model=model, optimizer=optimizer)

    assert [summary.epoch for summary in history] == [1, 2]
