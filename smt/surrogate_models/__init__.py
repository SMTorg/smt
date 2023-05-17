from .ls import LS
from .qp import QP
from .krg import KRG
from .kpls import KPLS
from .gekpls import GEKPLS
from .kplsk import KPLSK
from .genn import GENN
from .mgp import MGP

from .krg_based import MixIntKernelType
from smt.utils.design_space import (
    DesignSpace,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
    CategoricalVariable,
)
from smt.utils.kriging import MixHrcKernelType

try:
    from .idw import IDW
    from .rbf import RBF
    from .rmtc import RMTC
    from .rmtb import RMTB
except:
    pass
