"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

import numpy as np
import unittest
import inspect

from collections import OrderedDict

from smt.problems import Sphere, TensorProduct
from smt.sampling_methods import LHS
from smt.utils.design_space import DesignSpace

from smt.utils.sm_test_case import SMTestCase
from smt.utils.silence import Silence
from smt.utils import compute_rms_error
from smt.surrogate_models import LS, QP, KPLS, KRG

try:
    from smt.surrogate_models import IDW, RBF, RMTC, RMTB

    compiled_available = True
except:
    compiled_available = False


print_output = False


class Test(SMTestCase):
    def setUp(self):
        ndim = 2
        nt = 10000
        ne = 1000

        problems = OrderedDict()
        problems["sphere"] = Sphere(ndim=ndim)
        problems["exp"] = TensorProduct(ndim=ndim, func="exp", width=5)
        problems["tanh"] = TensorProduct(ndim=ndim, func="tanh", width=5)
        problems["cos"] = TensorProduct(ndim=ndim, func="cos", width=5)

        sms = OrderedDict()
        sms["LS"] = LS()
        sms["QP"] = QP()
        if compiled_available:
            sms["RMTC"] = RMTC(num_elements=20, energy_weight=1e-10)
            sms["RMTB"] = RMTB(num_ctrl_pts=40, energy_weight=1e-10)

        t_errors = {}
        t_errors["LS"] = 1.0
        t_errors["QP"] = 1.0
        t_errors["RMTC"] = 1.0
        t_errors["RMTB"] = 1.0

        e_errors = {}
        e_errors["LS"] = 1.5
        e_errors["QP"] = 1.5
        e_errors["RMTC"] = 1.0
        e_errors["RMTB"] = 1.0

        self.nt = nt
        self.ne = ne
        self.problems = problems
        self.sms = sms
        self.t_errors = t_errors
        self.e_errors = e_errors

    def run_test(self):
        method_name = inspect.stack()[1][3]
        pname = method_name.split("_")[1]
        sname = method_name.split("_")[2]

        prob = self.problems[pname]
        sampling = LHS(xlimits=prob.xlimits)

        np.random.seed(0)
        xt = sampling(self.nt)
        yt = prob(xt)

        np.random.seed(1)
        xe = sampling(self.ne)
        ye = prob(xe)

        sm0 = self.sms[sname]

        sm = sm0.__class__()
        sm.options = sm0.options.clone()
        if sm.options.is_declared("design_space"):
            sm.options["design_space"] = DesignSpace(prob.xlimits)
        if sm.options.is_declared("xlimits"):
            sm.options["xlimits"] = prob.xlimits
        sm.options["print_global"] = False

        sm.set_training_values(xt, yt)

        with Silence():
            sm.train()

        t_error = compute_rms_error(sm)
        e_error = compute_rms_error(sm, xe, ye)

        if print_output:
            print("%8s %6s %18.9e %18.9e" % (pname[:6], sname, t_error, e_error))

        self.assert_error(t_error, 0.0, self.t_errors[sname])
        self.assert_error(e_error, 0.0, self.e_errors[sname])

    # --------------------------------------------------------------------
    # Function: sphere

    def test_sphere_LS(self):
        self.run_test()

    def test_sphere_QP(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_sphere_RMTC(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_sphere_RMTB(self):
        self.run_test()

    # --------------------------------------------------------------------
    # Function: exp

    def test_exp_LS(self):
        self.run_test()

    def test_exp_QP(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_exp_RMTC(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_exp_RMTB(self):
        self.run_test()

    # --------------------------------------------------------------------
    # Function: tanh

    def test_tanh_LS(self):
        self.run_test()

    def test_tanh_QP(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_tanh_RMTC(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_tanh_RMTB(self):
        self.run_test()

    # --------------------------------------------------------------------
    # Function: cos

    def test_cos_LS(self):
        self.run_test()

    def test_cos_QP(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_cos_RMTC(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_cos_RMTB(self):
        self.run_test()


if __name__ == "__main__":
    print_output = True
    print("%6s %8s %18s %18s" % ("SM", "Problem", "Train. pt. error", "Test pt. error"))
    unittest.main()
