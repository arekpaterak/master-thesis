from core.models.nds.models.ar import AutoregressiveNDSModel
from core.models.nds.models.one_pass import OnePassNDSModel

NDSModel = AutoregressiveNDSModel | OnePassNDSModel
