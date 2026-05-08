import torch

from core.models.nds.config import TrainConfig, TrainerParams
from core.models.nds.functions import noop_repair, only_expanding_apply_rollout
from core.models.nds.model import NDSModel
from core.models.nds.module import NDSModule
from core.models.trainer import Trainer
from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.cvrp.solve import one_customer_per_route


def test_train():
    B, R, N = 4, 10, 20
    sampler = CVRPSampler(CVRPConfig.for_n(N))
    model = NDSModel(problem_name="cvrp")

    nds_module = NDSModule(
        config=TrainConfig(
            trainer=TrainerParams(
                num_main_improvement_steps=3,
                rollouts_per_instance=R,
                num_initial_improvement_steps=10,
                deconstruction_size=15,
            )
        ),
        model=model,
        generate_start_solution=one_customer_per_route,
        apply_rollout=only_expanding_apply_rollout,
        repair=noop_repair,
    )
    optimizer = torch.optim.AdamW(nds_module.parameters())

    seen_epochs: list[int] = []
    Trainer(
        max_epochs=2,
        training_batches_per_epoch=2,
        batch_size=B,
        on_epoch_train_end=[lambda s: seen_epochs.append(s.epoch)],
    ).fit(model=nds_module, optimizer=optimizer, instance_sampler=sampler)

    assert seen_epochs == [0, 1]
