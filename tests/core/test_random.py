import pytest
import torch

from core.lns.acceptance_criterion import SimulatedAnnealingAcceptance
from core.models.attention_model.model import AttentionModel
from core.models.nds.config import JointModelConfig, SplitModelConfig
from core.models.nds.decoders.config import (
    AutoregressiveDecoderConfig,
    SimpleOnePassDecoderConfig,
    SlotDecoderConfig,
)
from core.problems import make_problem_config, registry
from core.random import SeededGenerator, sample_categorical, sample_gumbel


def test_same_seed_gives_same_stream() -> None:
    a, b = SeededGenerator(7).on("cpu"), SeededGenerator(7).on("cpu")
    assert torch.equal(torch.rand(5, generator=a), torch.rand(5, generator=b))


def test_generator_is_created_once_per_device() -> None:
    rng = SeededGenerator(7)
    assert rng.on("cpu") is rng.on(torch.device("cpu"))


def test_sample_categorical_matches_torch_categorical() -> None:
    logits = torch.randn(3, 4, 6)
    torch.manual_seed(1)
    expected = torch.distributions.Categorical(logits=logits).sample()
    assert torch.equal(sample_categorical(logits, generator=SeededGenerator(1).on("cpu")), expected)


def test_sample_gumbel_matches_torch_gumbel() -> None:
    like = torch.zeros(3, 5)
    torch.manual_seed(1)
    expected = torch.distributions.Gumbel(torch.zeros_like(like), torch.ones_like(like)).sample()
    assert torch.equal(sample_gumbel(like, generator=SeededGenerator(1).on("cpu")), expected)


def _leaves_global_rng_untouched(run) -> None:
    state = torch.random.get_rng_state()
    run()
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize("model_config", [JointModelConfig, SplitModelConfig])
@pytest.mark.parametrize(
    "decoder",
    [AutoregressiveDecoderConfig(), SimpleOnePassDecoderConfig(), SlotDecoderConfig(num_slots=5)],
)
def test_nds_rollout_is_reproducible_and_leaves_global_rng_untouched(model_config, decoder):
    model = registry.build_nds_model(model_config(problem_name="cvrp", decoder=decoder)).eval()
    instance = registry.build_sampler(make_problem_config(name="cvrp", size=10), seed=0)(2)
    solution = registry.problem("cvrp").initial_solution(instance)

    def rollout(seed: int):
        return model.rollout(
            solution, rollouts_per_solution=3, deconstruction_size=5, rng=SeededGenerator(seed)
        )

    _leaves_global_rng_untouched(lambda: rollout(0))
    assert torch.equal(rollout(0).actions, rollout(0).actions)


def test_am_sampling_needs_an_rng_and_leaves_global_rng_untouched() -> None:
    model = AttentionModel(problem_name="tsp", hidden_dim=8, num_heads=2, num_layers=1).eval()
    instance = registry.build_sampler(make_problem_config(name="tsp", size=8), seed=0)(2)

    with pytest.raises(ValueError, match="needs an rng"):
        model.rollout(instance, greedy=False)
    _leaves_global_rng_untouched(lambda: model.rollout(instance, rng=SeededGenerator(0)))


def test_simulated_annealing_leaves_global_rng_untouched() -> None:
    cvrp = registry.problem("cvrp")
    sampler = registry.build_sampler(make_problem_config(name="cvrp", size=10), seed=0)
    current, candidate = cvrp.initial_solution(sampler(16)), cvrp.initial_solution(sampler(16))
    criterion = SimulatedAnnealingAcceptance(temperature=1.0, decay=0.9, seed=0)

    _leaves_global_rng_untouched(lambda: criterion(current, candidate))
