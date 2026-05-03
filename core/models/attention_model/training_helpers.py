import copy

from core.models.attention_model import AttentionModel
from core.models.utils import freeze_parameters_


def sync_baseline_(baseline: AttentionModel, model: AttentionModel) -> None:
    baseline.load_state_dict(copy.deepcopy(model.state_dict()))
    baseline.eval()
    freeze_parameters_(baseline)


def prepare_baseline(model: AttentionModel) -> AttentionModel:
    baseline = copy.deepcopy(model)
    sync_baseline_(baseline, model)
    return baseline
