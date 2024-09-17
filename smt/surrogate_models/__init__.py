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
from smt.utils.design_space import HAS_SMTDesignSpace

if HAS_SMTDesignSpace:
    from SMTDesignSpace import design_space as ds
    from SMTDesignSpace.design_space import (
        HAS_CONFIG_SPACE,
        CategoricalVariable,
        BaseDesignSpace,
        DesignSpace,
        FloatVariable,
        IntegerVariable,
        OrdinalVariable,
        ensure_design_space,
    )
else:
    from smt.utils import design_space as ds
    from smt.utils.design_space import (
        HAS_CONFIG_SPACE,
        CategoricalVariable,
        DesignSpace,
        BaseDesignSpace,
        FloatVariable,
        IntegerVariable,
        OrdinalVariable,
        ensure_design_space,
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
