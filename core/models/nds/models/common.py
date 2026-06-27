from core.models.nds.models.joint import JointNDSModel
from core.models.nds.models.split import SplitNDSModel

NDSModel = JointNDSModel | SplitNDSModel
