from __future__ import print_function, division
import numpy as np
import unittest

from smt.problems.carre import Carre
from smt.problems.tensor_product import TensorProduct
from smt.sampling.lhs import lhs_center
from smt.sampling.random import random
from smt.utils.testing import SMTestCase

from smt.ls import LS
from smt.pa2 import PA2
from smt.kpls import KPLS
try:
    from smt.idw import IDW
    from smt.rmts import RMTS
    from smt.mbr import MBR
    compiled_available = True
except:
    compiled_available = False


class Test(SMTestCase):

    def run_model(self, sm, prob, nt=100, ne=1000):
        sampling = lhs_center

        np.random.seed(0)
        xt = sampling(prob.xlimits, nt)
        yt = prob(xt)

        np.random.seed(1)
        xe = sampling(prob.xlimits, ne)
        ye = prob(xe)

        sm.training_pts = {'exact': {}}
        sm.add_training_pts('exact', xt, yt)
        sm.train()

        t_error = sm.compute_rms_error()
        e_error = sm.compute_rms_error(xe, ye)
        print('%6s %8s %18.9e %18.9e'
              % (sm.sm_options['name'], prob.__class__.__name__[:6], t_error, e_error))

        return t_error, e_error

    def test_LS(self):
        ndim = 3
        sm = LS({'name':'LS'}, {'global':False})
        print()

        prob = Carre(ndim=ndim)
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1.0)
        self.assert_error(e_error, 0., 1.0)

        prob = TensorProduct(ndim=ndim, func='exp')
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1.0)
        self.assert_error(e_error, 0., 1.0)

        prob = TensorProduct(ndim=ndim, func='cos')
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1.0)
        self.assert_error(e_error, 0., 10000.0)

        prob = TensorProduct(ndim=ndim, func='tanh')
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1.0)
        self.assert_error(e_error, 0., 10000.0)

    def test_PA2(self):
        ndim = 3
        sm = PA2({'name':'PA2'}, {'global':False})
        print()

        prob = Carre(ndim=ndim)
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1e-14)
        self.assert_error(e_error, 0., 1e-14)

        prob = TensorProduct(ndim=ndim, func='exp')
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1.0)
        self.assert_error(e_error, 0., 1.0)

        prob = TensorProduct(ndim=ndim, func='cos')
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1.0)
        self.assert_error(e_error, 0., 10000.0)

        prob = TensorProduct(ndim=ndim, func='tanh')
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1.0)
        self.assert_error(e_error, 0., 10000.0)

    def test_IDW(self):
        if not compiled_available:
            return

        ndim = 3
        sm = IDW({'name':'IDW'}, {'global':False})
        print()

        prob = Carre(ndim=ndim)
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1e-15)
        self.assert_error(e_error, 0., 1.0)

        prob = TensorProduct(ndim=ndim, func='exp')
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1e-15)
        self.assert_error(e_error, 0., 1.0)

        prob = TensorProduct(ndim=ndim, func='cos')
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1e-15)
        self.assert_error(e_error, 0., 1.0)

        prob = TensorProduct(ndim=ndim, func='tanh')
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1e-15)
        self.assert_error(e_error, 0., 1.0)

    def test_KRG(self):
        if not compiled_available:
            return

        ndim = 3
        sm = KPLS({'name':'KRG','n_comp':ndim,'theta0': [1e-2]*ndim}, {'global':False})
        print()

        prob = Carre(ndim=ndim)
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1e-5)
        self.assert_error(e_error, 0., 1.0)

        prob = TensorProduct(ndim=ndim, func='exp')
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1e-1)
        self.assert_error(e_error, 0., 1.0)

        prob = TensorProduct(ndim=ndim, func='cos')
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 2.0)
        self.assert_error(e_error, 0., 5.0)

        prob = TensorProduct(ndim=ndim, func='tanh')
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 2.0)
        self.assert_error(e_error, 0., 5.0)

    def test_RMTS(self):
        if not compiled_available:
            return

        ndim = 3
        print()

        prob = Carre(ndim=ndim)
        sm = RMTS({'name':'RMTS', 'num_elem':[4]*ndim, 'solver': 'krylov-lu',
            'xlimits':prob.xlimits, 'save_solution': False}, {'global': False})
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1e-5)
        self.assert_error(e_error, 0., 1.0)

        prob = TensorProduct(ndim=ndim, func='exp')
        sm = RMTS({'name':'RMTS', 'num_elem':[4]*ndim, 'solver': 'krylov-lu',
            'xlimits':prob.xlimits, 'save_solution': False}, {'global': False})
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1e-5)
        self.assert_error(e_error, 0., 1.0)

        prob = TensorProduct(ndim=ndim, func='cos')
        sm = RMTS({'name':'RMTS', 'num_elem':[4]*ndim, 'solver': 'krylov-lu',
            'xlimits':prob.xlimits, 'save_solution': False}, {'global': False})
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1e-5)
        self.assert_error(e_error, 0., 5.0)

        prob = TensorProduct(ndim=ndim, func='tanh')
        sm = RMTS({'name':'RMTS', 'num_elem':[4]*ndim, 'solver': 'krylov-lu',
            'xlimits':prob.xlimits, 'save_solution': False}, {'global': False})
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1e-5)
        self.assert_error(e_error, 0., 5.0)

    def test_MBR(self):
        if not compiled_available:
            return

        ndim = 3
        print()

        prob = Carre(ndim=ndim)
        sm = MBR({'name':'MBR', 'order':[4]*ndim, 'num_ctrl_pts':[8]*ndim,
            'xlimits':prob.xlimits, 'save_solution': False}, {'global': False})
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1e-5)
        self.assert_error(e_error, 0., 1.0)

        prob = TensorProduct(ndim=ndim, func='exp')
        sm = MBR({'name':'MBR', 'order':[4]*ndim, 'num_ctrl_pts':[8]*ndim,
            'xlimits':prob.xlimits, 'save_solution': False}, {'global': False})
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1e-5)
        self.assert_error(e_error, 0., 1.0)

        prob = TensorProduct(ndim=ndim, func='cos')
        sm = MBR({'name':'MBR', 'order':[4]*ndim, 'num_ctrl_pts':[8]*ndim,
            'xlimits':prob.xlimits, 'save_solution': False}, {'global': False})
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1e-5)
        self.assert_error(e_error, 0., 5.0)

        prob = TensorProduct(ndim=ndim, func='tanh')
        sm = MBR({'name':'MBR', 'order':[4]*ndim, 'num_ctrl_pts':[8]*ndim,
            'xlimits':prob.xlimits, 'save_solution': False}, {'global': False})
        t_error, e_error = self.run_model(sm, prob)
        self.assert_error(t_error, 0., 1e-5)
        self.assert_error(e_error, 0., 5.0)

    # def test1(self):
    #     ndim = 3
    #     prob = Carre(ndim=ndim)
    #     prob = TensorProduct(ndim=ndim, func='tanh')
    #     sampling = lhs_center
    #
    #     sm = RMTS({'name':'RMTS','num_elem':[8]*ndim, 'smoothness':[1.0]*ndim,
    #         'xlimits':prob.xlimits, 'mode': 'approx', 'approx_norm': 4,
    #         'reg_dv': 1e-8, 'reg_cons': 1e-12, 'save_solution': False,
    #         'mg_factors': [2, 2, 2], 'solver': 'krylov', 'solver_nln_iter': 20,
    #         'line_search': 'backtracking',
    #     }, {})
    #
    #     # sm = KPLS({'name':'KRG','n_comp':ndim,'theta0': [1e-2]*ndim},{})
    #
    #     nt = 5000 * ndim
    #     ne = 100 * ndim
    #
    #     np.random.seed(0)
    #     xt = sampling(prob.xlimits, nt)
    #     yt = prob(xt)
    #
    #     np.random.seed(1)
    #     xe = sampling(prob.xlimits, ne)
    #     ye = prob(xe)
    #
    #     sm.add_training_pts('exact', xt, yt)
    #     sm.train()
    #
    #     print(sm.compute_rms_error())
    #     print(sm.compute_rms_error(xe, ye))


if __name__ == '__main__':
    print('%6s %8s %18s %18s'
          % ('SM', 'Problem', 'Train. pt. error', 'Test pt. error'))
    unittest.main()
