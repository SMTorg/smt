from __future__ import print_function, division
import numpy as np
import unittest

from smt.problems.carre import Carre
from smt.problems.tensor_product import TensorProduct
from smt.sampling.lhs import lhs_center
from smt.sampling.random import random
from smt.ls import LS
from smt.pa2 import PA2
from smt.kpls import KPLS
from smt.idw import IDW
from smt.rmts import RMTS
from smt.mbr import MBR


class Test(unittest.TestCase):

    def test1(self):
        ndim = 3
        prob = Carre(ndim=ndim)
        prob = TensorProduct(ndim=ndim, func='tanh')
        sampling = lhs_center

        sm = RMTS({'name':'RMTS','num_elem':[8]*ndim, 'smoothness':[1.0]*ndim,
            'xlimits':prob.xlimits, 'mode': 'approx', 'approx_norm': 4,
            'reg_dv': 1e-8, 'reg_cons': 1e-12, 'save_solution': False,
            'mg_factors': [2, 2, 2], 'solver': 'krylov', 'max_nln_iter': 20,
            'line_search': 'backtracking',
        }, {})

        # sm = KPLS({'name':'KRG','n_comp':ndim,'theta0': [1e-2]*ndim},{})

        nt = 5000 * ndim
        ne = 100 * ndim

        np.random.seed(0)
        xt = sampling(prob.xlimits, nt)
        yt = prob(xt)

        np.random.seed(1)
        xe = sampling(prob.xlimits, ne)
        ye = prob(xe)

        sm.add_training_pts('exact', xt, yt)
        sm.train()

        yt2 = sm.predict(xt)
        error = np.linalg.norm(yt2 - yt) / np.linalg.norm(yt)
        print(error)

        ye2 = sm.predict(xe)
        error = np.linalg.norm(ye2 - ye) / np.linalg.norm(ye)
        print(error)


if __name__ == '__main__':
    unittest.main()
