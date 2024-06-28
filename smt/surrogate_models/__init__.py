from smt.utils.design_space import (
    CategoricalVariable,
    DesignSpace,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
)
from smt.utils.kriging import MixHrcKernelType

from .gekpls import GEKPLS
from .genn import GENN
from .gpx import GPX
from .kpls import KPLS
from .kplsk import KPLSK
from .krg import KRG
from .krg_based import MixIntKernelType
from .ls import LS
from .mgp import MGP
from .qp import QP
from .sgp import SGP

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
    from .rmtb import RMTB
    from .rmtc import RMTC

    __all__ = __all__ + ["IDW", "RBF", "RMTC", "RMTB"]

except ImportError:
    pass
