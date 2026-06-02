import torch

from core.lns.config import SearchParams
from core.models.nds.config import AmpConfig, TrainConfig, TrainerParams
from core.models.nds.models.ar import AutoregressiveNDSModel
from core.models.nds.module import NDSModule
from core.models.trainer import Trainer
from core.problems.cvrp import CVRPConfig, CVRPSampler


def test_train():
    B, R, N = 4, 10, 20
    sampler = CVRPSampler(CVRPConfig.for_n(N))
    model = AutoregressiveNDSModel(problem_name="cvrp")

    nds_module = NDSModule(
        config=TrainConfig(
            trainer=TrainerParams(
                num_main_improvement_steps=3,
            ),
            search=SearchParams(
                num_improvement_steps=0,
                num_rollouts=R,
                deconstruction_size=15,
            ),
        ),
        model=model,
        repair_type="noop",
    )
    optimizer = torch.optim.AdamW(nds_module.parameters())

    seen_epochs: list[int] = []
    Trainer(
        max_epochs=2,
        training_batches_per_epoch=2,
        batch_size=B,
        on_epoch_train_end=[lambda s: seen_epochs.append(s.epoch)],
    ).fit(module=nds_module, optimizer=optimizer, instance_sampler=sampler)

    assert seen_epochs == [0, 1]


def test_train_with_amp_enabled():
    B, R, N = 4, 10, 20
    sampler = CVRPSampler(CVRPConfig.for_n(N))
    model = AutoregressiveNDSModel(problem_name="cvrp")

    nds_module = NDSModule(
        config=TrainConfig(
            trainer=TrainerParams(
                num_main_improvement_steps=2,
            ),
            search=SearchParams(
                num_improvement_steps=0,
                num_rollouts=R,
                deconstruction_size=15,
            ),
            amp=AmpConfig(enabled=True, dtype="bfloat16"),
        ),
        model=model,
        repair_type="noop",
    )
    optimizer = torch.optim.AdamW(nds_module.parameters())

    Trainer(
        max_epochs=1,
        training_batches_per_epoch=1,
        batch_size=B,
    ).fit(module=nds_module, optimizer=optimizer, instance_sampler=sampler)
