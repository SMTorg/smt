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

from smt.ls import LS
from smt.pa2 import PA2
from smt.kpls import KPLS

try:
    from smt.idw import IDW
    from smt.rmts import RMTS
    from smt.rmtb import RMTB
    compiled_available = True
except:
    compiled_available = False


print_output = False

class Test(SMTestCase):

    def setUp(self):
        ndim = 3
        nt = 500
        ne = 100

        problems = OrderedDict()
        problems['carre'] = Carre(ndim=ndim)
        problems['exp'] = TensorProduct(ndim=ndim, func='exp')
        problems['tanh'] = TensorProduct(ndim=ndim, func='tanh')
        problems['cos'] = TensorProduct(ndim=ndim, func='cos')

        sms = OrderedDict()
        if compiled_available:
            sms['RMTS'] = RMTS({'name':'RMTS', 'num_elem':[6]*ndim, 'solver':'krylov-lu'})
            sms['RMTB'] = RMTB({'name':'RMTB', 'order':[6]*ndim, 'num_ctrl_pts':[8]*ndim,
                'solver':'krylov-lu'})

        t_errors = {}
        t_errors['RMTS'] = 1e-6
        t_errors['RMTB'] = 1e-2

        e_errors = {}
        e_errors['RMTS'] = 1e-1
        e_errors['RMTB'] = 1e-1

        ge_t_errors = {}
        ge_t_errors['RMTS'] = 1e-4
        ge_t_errors['RMTB'] = 2e-2

        ge_e_errors = {}
        ge_e_errors['RMTS'] = 4e-1
        ge_e_errors['RMTB'] = 1e-1

        self.nt = nt
        self.ne = ne
        self.problems = problems
        self.sms = sms
        self.t_errors = t_errors
        self.e_errors = e_errors
        self.ge_t_errors = ge_t_errors
        self.ge_e_errors = ge_e_errors

    def run_test(self):
        method_name = inspect.stack()[1][3]
        pname = method_name.split('_')[1]
        sname = method_name.split('_')[2]

        if sname in ['IDW', 'RMTS', 'RMTB'] and not compiled_available:
            return

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

        sm = sm0.__class__()
        sm.sm_options = dict(sm0.sm_options)
        sm.printf_options = dict(sm0.printf_options)
        sm.sm_options['xlimits'] = prob.xlimits
        sm.printf_options['global'] = False

        sm.training_pts = {'exact': {}}
        sm.add_training_pts('exact', xt, yt)
        for kx in range(prob.xlimits.shape[0]):
            sm.add_training_pts('exact', xt, dyt[kx], kx)

        with Silence():
            sm.train()

        ge_t_error = sm.compute_rms_error()
        ge_e_error = sm.compute_rms_error(xe, ye)

        if print_output:
            print('%8s %6s %18.9e %18.9e %18.9e %18.9e'
                  % (pname[:6], sname, t_error, e_error, ge_t_error, ge_e_error))

        self.assert_error(t_error, 0., self.t_errors[sname])
        self.assert_error(e_error, 0., self.e_errors[sname])
        self.assert_error(ge_t_error, 0., self.ge_t_errors[sname])
        self.assert_error(ge_e_error, 0., self.ge_e_errors[sname])

    # --------------------------------------------------------------------
    # Function: carre

    def test_carre_RMTS(self):
        self.run_test()

    def test_carre_RMTB(self):
        self.run_test()

    # --------------------------------------------------------------------
    # Function: exp

    def test_exp_RMTS(self):
        self.run_test()

    def test_exp_RMTB(self):
        self.run_test()

    # --------------------------------------------------------------------
    # Function: tanh

    def test_tanh_RMTS(self):
        self.run_test()

    def test_tanh_RMTB(self):
        self.run_test()

    # --------------------------------------------------------------------
    # Function: cos

    def test_cos_RMTS(self):
        self.run_test()

    def test_cos_RMTB(self):
        self.run_test()


if __name__ == '__main__':
    print_output = True
    print('%6s %8s %18s %18s %18s %18s'
          % ('SM', 'Problem', 'Train. pt. error', 'Test pt. error', 'GE tr. pt. error', 'GE test pt. error'))
    unittest.main()
