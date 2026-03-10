import torch

from core.models.attention.model import AttentionModel


def test_attention_model_rollout():
    batch_size = 2
    n = 5
    d_x = 2
    d_h = 16

    model = AttentionModel(d_x=d_x, d_h=d_h, num_heads=2, num_layers=1)
    instances = torch.rand((batch_size, n, d_x))

    result_greedy = model(instances, greedy=True)
    assert result_greedy.tour.shape == (batch_size, n), "Expected tour shape (batch_size, n)"
    assert result_greedy.log_p.shape == (batch_size, n), "Expected log_p shape (batch_size, n)"
    assert isinstance(result_greedy.entropy, float), "Expected entropy to be float"
    assert isinstance(result_greedy.entropy_norm, float), "Expected entropy_norm to be float"

    result_sample = model(instances, greedy=False)
    assert result_sample.tour.shape == (batch_size, n), "Expected tour shape (batch_size, n)"
    assert result_sample.log_p.shape == (batch_size, n), "Expected log_p shape (batch_size, n)"
