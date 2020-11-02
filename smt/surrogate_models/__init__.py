from .ls import LS
from .qp import QP
from .kpls import KPLS
from .krg import KRG
from .gekpls import GEKPLS
from .kplsk import KPLSK
from .genn import GENN
from .krg_mgp import MGP

try:
    from .idw import IDW
    from .rbf import RBF
    from .rmtc import RMTC
    from .rmtb import RMTB
except:
    pass
