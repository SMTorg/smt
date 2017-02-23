from __future__ import print_function, division
import numpy as np
import unittest

from six import iteritems
from collections import OrderedDict

from smt.problems.carre import Carre
from smt.problems.tensor_product import TensorProduct
from smt.sampling.lhs import lhs_center
from smt.sampling.random import random
from smt.utils.testing import SMTestCase
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


class Test(SMTestCase):

    def test_all(self):
        ndim = 10
        nt = 500
        ne = 100
        sampling = lhs_center

        problems = OrderedDict()
        problems['carre'] = Carre(ndim=ndim)
        problems['tp-exp'] = TensorProduct(ndim=ndim, func='exp')
        problems['tp-tanh'] = TensorProduct(ndim=ndim, func='tanh')
        problems['tp-cos'] = TensorProduct(ndim=ndim, func='cos')

        sms = OrderedDict()
        sms['LS'] = LS()
        sms['PA2'] = PA2()
        sms['KRG'] = KPLS({'name':'KRG','n_comp':ndim, 'theta0': [40e-2]*ndim})
        if compiled_available:
            sms['IDW'] = IDW()

        t_errors = {}
        t_errors['LS'] = 1.0
        t_errors['PA2'] = 1.0
        t_errors['KRG'] = 1e-6
        t_errors['IDW'] = 1e-15

        e_errors = {}
        e_errors['LS'] = 1.5
        e_errors['PA2'] = 1.5
        e_errors['KRG'] = 1e-3
        e_errors['IDW'] = 1.0

        for pname, prob in iteritems(problems):

            np.random.seed(0)
            xt = sampling(prob.xlimits, nt)
            yt = prob(xt)

            np.random.seed(1)
            xe = sampling(prob.xlimits, ne)
            ye = prob(xe)

            print()
            for sname, sm0 in iteritems(sms):

                sm = sm0.__class__()
                sm.sm_options = dict(sm0.sm_options)
                sm.printf_options = dict(sm0.printf_options)
                sm.sm_options['xlimits'] = prob.xlimits
                sm.printf_options['global'] = False

                sm.training_pts = {'exact': {}}
                sm.add_training_pts('exact', xt, yt)
                with Silence():
                    sm.train()

                t_error = sm.compute_rms_error()
                e_error = sm.compute_rms_error(xe, ye)
                print('%8s %6s %18.9e %18.9e'
                      % (pname[:6], sname, t_error, e_error))
                self.assert_error(t_error, 0., t_errors[sname])
                # self.assert_error(e_error, 0., e_errors[sname])


if __name__ == '__main__':
    print('%6s %8s %18s %18s'
          % ('SM', 'Problem', 'Train. pt. error', 'Test pt. error'))
    unittest.main()
