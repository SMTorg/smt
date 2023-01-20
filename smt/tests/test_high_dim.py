"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

import os
import numpy as np
import unittest
import inspect

from collections import OrderedDict

from smt.problems import Sphere, TensorProduct
from smt.sampling_methods import LHS

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
        ndim = 10
        nt = 500
        ne = 100

        problems = OrderedDict()
        problems["sphere"] = Sphere(ndim=ndim)
        problems["exp"] = TensorProduct(ndim=ndim, func="exp")
        problems["tanh"] = TensorProduct(ndim=ndim, func="tanh")
        problems["cos"] = TensorProduct(ndim=ndim, func="cos")

        sms = OrderedDict()
        sms["LS"] = LS()
        sms["QP"] = QP()
        sms["KRG"] = KRG(theta0=[4e-1] * ndim)
        sms["KPLS"] = KPLS()

        if compiled_available:
            sms["IDW"] = IDW()
            sms["RBF"] = RBF()

        t_errors = {}
        t_errors["LS"] = 1.0
        t_errors["QP"] = 1.0
        t_errors["KRG"] = 1e-4
        t_errors["IDW"] = 1e-15
        t_errors["RBF"] = 1e-2
        t_errors["KPLS"] = 1e-3

        e_errors = {}
        e_errors["LS"] = 2.5
        e_errors["QP"] = 2.0
        e_errors["KRG"] = 2.0
        e_errors["IDW"] = 4
        e_errors["RBF"] = 2
        e_errors["KPLS"] = 2.5

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
        sampling = LHS(xlimits=prob.xlimits, random_state=42)

        np.random.seed(0)
        xt = sampling(self.nt)
        yt = prob(xt)

        np.random.seed(1)
        xe = sampling(self.ne)
        ye = prob(xe)

        sm0 = self.sms[sname]

        sm = sm0.__class__()
        sm.options = sm0.options.clone()
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

        self.assert_error(t_error, 0.0, self.t_errors[sname], 1e-5)
        self.assert_error(e_error, 0.0, self.e_errors[sname], 1e-5)

    # --------------------------------------------------------------------
    # Function: sphere

    def test_sphere_LS(self):
        self.run_test()

    def test_sphere_QP(self):
        self.run_test()

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_sphere_KRG(self):
        self.run_test()

    def test_sphere_KPLS(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_sphere_IDW(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_sphere_RBF(self):
        self.run_test()

    # --------------------------------------------------------------------
    # Function: exp

    def test_exp_LS(self):
        self.run_test()

    def test_exp_QP(self):
        self.run_test()

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_exp_KRG(self):
        self.run_test()

    def test_exp_KPLS(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_exp_IDW(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_exp_RBF(self):
        self.run_test()

    # --------------------------------------------------------------------
    # Function: tanh

    def test_tanh_LS(self):
        self.run_test()

    def test_tanh_QP(self):
        self.run_test()

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_tanh_KRG(self):
        self.run_test()

    def test_tanh_KPLS(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_tanh_IDW(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_tanh_RBF(self):
        self.run_test()

    # --------------------------------------------------------------------
    # Function: cos

    def test_cos_LS(self):
        self.run_test()

    def test_cos_QP(self):
        self.run_test()

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_cos_KRG(self):
        self.run_test()

    def test_cos_KPLS(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_cos_IDW(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_cos_RBF(self):
        self.run_test()


if __name__ == "__main__":
    print_output = True
    print("%6s %8s %18s %18s" % ("SM", "Problem", "Train. pt. error", "Test pt. error"))
    unittest.main()
