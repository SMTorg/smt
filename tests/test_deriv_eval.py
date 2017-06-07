from __future__ import print_function, division
import numpy as np
import unittest
import inspect

from six import iteritems
from collections import OrderedDict

from smt.problems import Carre, TensorProduct
from smt.sampling import LHS

from smt.utils.sm_test_case import SMTestCase
from smt.utils.silence import Silence

from smt import LS, PA2, KPLS
try:
    from smt import IDW, RBF, RMTC, RMTB
    compiled_available = True
except:
    compiled_available = False


print_output = False

class Test(SMTestCase):

    def setUp(self):
        ndim = 2
        nt = 5000
        ne = 100

        problems = OrderedDict()
        problems['carre'] = Carre(ndim=ndim)

        sms = OrderedDict()
        if compiled_available:
            sms['RBF'] = RBF()
            sms['RMTC'] = RMTC()
            sms['RMTB'] = RMTB()

        self.nt = nt
        self.ne = ne
        self.problems = problems
        self.sms = sms

    def run_test(self):
        method_name = inspect.stack()[1][3]
        pname = method_name.split('_')[1]
        sname = method_name.split('_')[2]

        prob = self.problems[pname]
        sampling = LHS(xlimits=prob.xlimits)

        np.random.seed(0)
        xt = sampling(self.nt)
        yt = prob(xt)
        dyt = {}
        for kx in range(prob.xlimits.shape[0]):
            dyt[kx] = prob(xt, kx=kx)

        np.random.seed(1)
        xe = sampling(self.ne)
        ye = prob(xe)
        dye = {}
        for kx in range(prob.xlimits.shape[0]):
            dye[kx] = prob(xe, kx=kx)

        sm0 = self.sms[sname]

        sm = sm0.__class__()
        sm.options = sm0.options.clone()
        if sm.options.is_declared('xlimits'):
            sm.options['xlimits'] = prob.xlimits
        sm.options['print_global'] = False

        sm.training_points = {'exact': {}}
        sm.add_training_points('exact', xt, yt)

        with Silence():
            sm.train()

        t_error = sm.compute_rms_error()
        e_error = sm.compute_rms_error(xe, ye)
        e_error0 = sm.compute_rms_error(xe, dye[0], 0)
        e_error1 = sm.compute_rms_error(xe, dye[1], 1)

        if print_output:
            print('%8s %6s %18.9e %18.9e %18.9e %18.9e'
                  % (pname[:6], sname, t_error, e_error, e_error0, e_error1))

        self.assert_error(e_error0, 0., 1e-1)
        self.assert_error(e_error1, 0., 1e-1)

    # --------------------------------------------------------------------
    # Function: carre

    @unittest.skipIf(not compiled_available, 'Compiled Fortran libraries not available')
    def test_carre_RBF(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, 'Compiled Fortran libraries not available')
    def test_carre_RMTC(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, 'Compiled Fortran libraries not available')
    def test_carre_RMTB(self):
        self.run_test()


if __name__ == '__main__':
    print_output = True
    print('%6s %8s %18s %18s %18s %18s'
          % ('SM', 'Problem', 'Train. pt. error', 'Test pt. error', 'Deriv 0 error', 'Deriv 1 error'))
    unittest.main()
