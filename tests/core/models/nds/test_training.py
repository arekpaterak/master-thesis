import torch

from core.lns.config import SearchParams, StartSolutionParams
from core.models.nds.baseline import GreedyRolloutBaselineConfig
from core.models.nds.config import (
    AmpConfig,
    JointModelParams,
    TrainConfig,
    TrainerParams,
    ValidationConfig,
)
from core.models.nds.decoders.config import SimpleOnePassDecoderConfig
from core.models.nds.models.joint import JointNDSModel
from core.models.nds.module import NDSModule
from core.models.nds.policy_loss import BestRolloutLossConfig
from core.models.trainer import Trainer
from core.problems.cvrp import CVRP, CVRPSampler


def test_train():
    B, R, N = 4, 10, 20
    sampler = CVRPSampler(CVRP.for_n(N))
    model = JointNDSModel(problem_name="cvrp")

    nds_module = NDSModule(
        config=TrainConfig(
            trainer=TrainerParams(
                num_main_improvement_steps=3,
            ),
            search=SearchParams(
                repair_type="greedy_insert",
                num_improvement_steps=0,
                num_rollouts=R,
                deconstruction_size=15,
                start_solution_params=StartSolutionParams(num_improvement_steps=0),
            ),
        ),
        model=model,
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
    sampler = CVRPSampler(CVRP.for_n(N))
    model = JointNDSModel(problem_name="cvrp")

    nds_module = NDSModule(
        config=TrainConfig(
            trainer=TrainerParams(
                num_main_improvement_steps=2,
            ),
            search=SearchParams(
                repair_type="greedy_insert",
                num_improvement_steps=0,
                num_rollouts=R,
                deconstruction_size=15,
                start_solution_params=StartSolutionParams(num_improvement_steps=0),
            ),
            amp=AmpConfig(enabled=True, dtype="bfloat16"),
        ),
        model=model,
    )
    optimizer = torch.optim.AdamW(nds_module.parameters())

    Trainer(
        max_epochs=1,
        training_batches_per_epoch=1,
        batch_size=B,
    ).fit(module=nds_module, optimizer=optimizer, instance_sampler=sampler)


def test_train_greedy_rollout_baseline_single_rollout():
    B, N = 4, 20
    sampler = CVRPSampler(CVRP.for_n(N))
    # The greedy rollout baseline requires a PolyNet-free model (with_poly=False).
    decoder = SimpleOnePassDecoderConfig(with_poly=False)
    model = JointNDSModel(problem_name="cvrp", decoder=decoder)

    small_search = SearchParams(
        repair_type="greedy_insert",
        num_improvement_steps=0,
        num_rollouts=1,
        deconstruction_size=15,
        start_solution_params=StartSolutionParams(num_improvement_steps=0),
    )
    nds_module = NDSModule(
        config=TrainConfig(
            model=JointModelParams(decoder=decoder),
            loss=BestRolloutLossConfig(),
            baseline=GreedyRolloutBaselineConfig(
                num_instances=B, batch_size=B, sync_every_n_epochs=1
            ),
            trainer=TrainerParams(num_main_improvement_steps=2),
            search=small_search,
            validation=ValidationConfig(
                search=small_search.model_copy(update={"num_improvement_steps": 2})
            ),
        ),
        model=model,
        baseline_eval_sampler=sampler,
    )
    optimizer = torch.optim.AdamW(nds_module.parameters())

    # The baseline model is a frozen copy and must not collect gradients.
    assert all(not p.requires_grad for p in nds_module.baseline._baseline_model.parameters())

    seen_epochs: list[int] = []
    Trainer(
        max_epochs=1,
        training_batches_per_epoch=1,
        batch_size=B,
        on_epoch_train_end=[lambda s: seen_epochs.append(s.epoch)],
    ).fit(module=nds_module, optimizer=optimizer, instance_sampler=sampler)

    # Smoke: a full epoch (training step + end-of-epoch t-test sync) runs at num_rollouts=1.
    assert seen_epochs == [0]
