from .ls import LS
from .qp import QP
from .krg import KRG
from .kpls import KPLS
from .gekpls import GEKPLS
from .kplsk import KPLSK
from .gpx import GPX
from .genn import GENN
from .mgp import MGP
from .sgp import SGP

from .krg_based import MixIntKernelType
from smt.utils.design_space import (
    DesignSpace,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
    CategoricalVariable,
)
from smt.utils.kriging import MixHrcKernelType

__all__ = [
    "LS",
    "QP",
    "KRG",
    "KPLS",
    "GEKPLS",
    "KPLSK",
    "GPX",
    "GENN",
    "MGP",
    "SGP",
    "MixIntKernelType",
    "DesignSpace",
    "FloatVariable",
    "IntegerVariable",
    "OrdinalVariable",
    "CategoricalVariable",
    "MixHrcKernelType",
]

try:
    from .idw import IDW
    from .rbf import RBF
    from .rmtc import RMTC
    from .rmtb import RMTB

    __all__ = __all__ + ["IDW", "RBF", "RMTC", "RMTB"]

except ImportError:
    pass
