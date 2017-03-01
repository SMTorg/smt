from __future__ import print_function, division
import numpy as np
import unittest
import inspect

from six import iteritems
from collections import OrderedDict

from smt.problems.carre import Carre
from smt.problems.tensor_product import TensorProduct
from smt.sampling.lhs import lhs_center
from smt.sampling.random import random
from smt.utils.sm_test_case import SMTestCase
from smt.utils.silence import Silence

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


print_output = False

class Test(SMTestCase):

    def setUp(self):
        ndim = 3
        nt = 500
        ne = 100
        sampling = lhs_center

        problems = OrderedDict()
        problems['carre'] = Carre(ndim=ndim)

        sms = OrderedDict()
        if compiled_available:
            sms['RMTS'] = RMTS({'name':'RMTS', 'num_elem':[6]*ndim, 'solver':'krylov-lu'})
            sms['MBR'] = MBR({'name':'MBR', 'order':[6]*ndim, 'num_ctrl_pts':[8]*ndim})

        self.nt = nt
        self.ne = ne
        self.sampling = sampling
        self.problems = problems
        self.sms = sms

    def run_test(self, sname, extrap_train=False, extrap_predict=False):
        if sname in ['IDW', 'RMTS', 'MBR'] and not compiled_available:
            return

        prob = self.problems['carre']

        np.random.seed(0)
        xt = self.sampling(prob.xlimits, self.nt)
        yt = prob(xt)

        sm0 = self.sms[sname]

        sm = sm0.__class__()
        sm.sm_options = dict(sm0.sm_options)
        sm.printf_options = dict(sm0.printf_options)
        sm.sm_options['xlimits'] = prob.xlimits
        sm.printf_options['global'] = False

        x = np.zeros((1, xt.shape[1]))
        x[0, :] = prob.xlimits[:, 1] + 1.0
        y = prob(x)

        sm.training_pts = {'exact': {}}
        sm.add_training_pts('exact', xt, yt)
        if extrap_train:
            sm.add_training_pts('exact', x, y)

        with Silence():
            sm.train()

        if extrap_predict:
            sm.predict(x)

    def test_rmts(self):
        self.run_test('RMTS', False, False)

    def test_rmts_train(self):
        if compiled_available:
            with self.assertRaises(Exception) as context:
                self.run_test('RMTS', True, False)
            self.assertEqual(str(context.exception),
                             'Training pts above max for 0')

    def test_rmts_predict(self):
        self.run_test('RMTS', False, True)

    def test_mbr(self):
        self.run_test('MBR', False, False)

    def test_mbr_train(self):
        if compiled_available:
            with self.assertRaises(Exception) as context:
                self.run_test('MBR', True, False)
            self.assertEqual(str(context.exception),
                             'Training pts above max for 0')

    def test_mbr_predict(self):
        self.run_test('MBR', False, True)


if __name__ == '__main__':
    unittest.main()
