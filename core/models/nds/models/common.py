from core.models.nds.models.ar import AutoregressiveNDSModel
from core.models.nds.models.lns_native import LNSNativeModel
from core.models.nds.models.one_pass import OnePassNDSModel

NDSModel = AutoregressiveNDSModel | OnePassNDSModel | LNSNativeModel
