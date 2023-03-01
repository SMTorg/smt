from .ls import LS
from .qp import QP
from .krg import KRG
from .kpls import KPLS
from .gekpls import GEKPLS
from .kplsk import KPLSK
from .genn import GENN
from .mgp import MGP

from .krg_based import (
    CONT_RELAX_KERNEL,
    GOWER_KERNEL,
    HOMO_HSPHERE_KERNEL,
    EXP_HOMO_HSPHERE_KERNEL,
)
from smt.utils.mixed_integer import (
    FLOAT_TYPE,
    ORD_TYPE,
    ENUM_TYPE,
)
from smt.utils.kriging import (
    NEUTRAL_ROLE,
    META_ROLE,
    DECREED_ROLE,
)

from smt.utils.kriging import XSpecs

try:
    from .idw import IDW
    from .rbf import RBF
    from .rmtc import RMTC
    from .rmtb import RMTB
except:
    pass
