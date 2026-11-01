from __future__ import annotations

from typing import Annotated

from pydantic import Field

from core.methods.construction import ConstructionMethodConfig
from core.methods.nds_pretrained import NDSPretrainedConfig
from core.methods.nds_trained import NDSTrainedConfig
from core.methods.random_destroy import RandomDestroyMethodConfig

MethodConfig = Annotated[
    NDSTrainedConfig | NDSPretrainedConfig | RandomDestroyMethodConfig | ConstructionMethodConfig,
    Field(discriminator="type"),
]
