import torch

from core.lns.config import SearchParams, StartSolutionParams
from core.models.nds.config import SplitModelConfig, TrainConfig, TrainerParams
from core.models.nds.decoders.config import SimpleOnePassDecoderConfig
from core.models.nds.models.joint import JointNDSModel
from core.models.nds.module import NDSModule
from core.models.runtime import AmpConfig
from core.models.trainer import Trainer
from core.problems.cvrp import CVRPDistribution, CVRPSampler
from core.problems.cvrp.lns.components import LNS
from core.problems.cvrp.nds.components import NDS as CVRP_NDS
from core.problems.registry import build_nds_model


def test_train():
    B, R, N = 4, 10, 20
    sampler = CVRPSampler(CVRPDistribution.for_n(N))
    model = JointNDSModel(components=CVRP_NDS)

    nds_module = NDSModule(
        lns=LNS,
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


def test_train_with_numba_apply_and_repair_backend():
    """End-to-end training step with the numba apply-and-repair backend."""
    B, R, N = 4, 10, 20
    sampler = CVRPSampler(CVRPDistribution.for_n(N))
    model = JointNDSModel(components=CVRP_NDS)

    nds_module = NDSModule(
        lns=LNS,
        config=TrainConfig(
            trainer=TrainerParams(
                num_main_improvement_steps=3,
            ),
            search=SearchParams(
                repair_type="greedy_insert",
                num_improvement_steps=0,
                num_rollouts=R,
                deconstruction_size=15,
                apply_and_repair_backend="numba",
                start_solution_params=StartSolutionParams(num_improvement_steps=0),
            ),
        ),
        model=model,
    )
    optimizer = torch.optim.AdamW(nds_module.parameters())

    summary = nds_module.training_step(optimizer=optimizer, instance=sampler(B))

    assert summary.loss == summary.loss  # not NaN
    assert summary.objective_after > 0


def test_train_with_amp_enabled():
    B, R, N = 4, 10, 20
    sampler = CVRPSampler(CVRPDistribution.for_n(N))
    model = JointNDSModel(components=CVRP_NDS)

    nds_module = NDSModule(
        lns=LNS,
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


def test_encode_once_matches_recompute():
    """encode_once_in_training must be a pure speed switch: identical loss and
    gradients to recomputing the encoder every step (the encoder is deterministic)."""
    B, R, N = 2, 8, 20

    def build_config(encode_once: bool) -> TrainConfig:
        return TrainConfig(
            model=SplitModelConfig(
                problem_name="cvrp",
                decoder=SimpleOnePassDecoderConfig(),
                encode_once_in_training=encode_once,
            ),
            trainer=TrainerParams(num_main_improvement_steps=3),
            search=SearchParams(
                repair_type="greedy_insert",
                num_improvement_steps=0,
                num_rollouts=R,
                deconstruction_size=8,
                start_solution_params=StartSolutionParams(num_improvement_steps=0),
            ),
        )

    model_baseline = build_nds_model(build_config(False).model)
    model_once = build_nds_model(build_config(True).model)
    model_once.load_state_dict(model_baseline.state_dict())

    module_baseline = NDSModule(config=build_config(False), model=model_baseline, lns=LNS)
    module_once = NDSModule(config=build_config(True), model=model_once, lns=LNS)

    instance = CVRPSampler(CVRPDistribution.for_n(N))(B)

    torch.manual_seed(0)
    summary_baseline = module_baseline.training_step(
        optimizer=torch.optim.AdamW(module_baseline.parameters()), instance=instance
    )
    torch.manual_seed(0)
    summary_once = module_once.training_step(
        optimizer=torch.optim.AdamW(module_once.parameters()), instance=instance
    )

    assert summary_once.loss == summary_baseline.loss
    torch.testing.assert_close(
        summary_once.grad_norm, summary_baseline.grad_norm, rtol=1e-4, atol=1e-5
    )

    grads_baseline = dict(model_baseline.named_parameters())
    for name, param_once in model_once.named_parameters():
        param_baseline = grads_baseline[name]
        if param_baseline.grad is None:
            assert param_once.grad is None, name
            continue
        torch.testing.assert_close(
            param_once.grad, param_baseline.grad, rtol=1e-4, atol=1e-5, msg=name
        )
