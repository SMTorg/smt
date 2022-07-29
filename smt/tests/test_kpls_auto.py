"""
Author: Paul Saves 

This package is distributed under New BSD license.
"""

import numpy as np
import unittest
import inspect

from collections import OrderedDict

from smt.problems import Sphere, TensorProduct, Rosenbrock, Branin
from smt.sampling_methods import LHS

from smt.utils.sm_test_case import SMTestCase
from smt.utils.silence import Silence
from smt.utils import compute_rms_error
from smt.surrogate_models import KPLS


print_output = False


class Test(SMTestCase):
    def setUp(self):
        ndim = 10
        nt = 50
        ne = 100

        problems = OrderedDict()
        problems["Branin"] = Branin(ndim=2)
        problems["Rosenbrock"] = Rosenbrock(ndim=3)
        problems["sphere"] = Sphere(ndim=ndim)
        problems["exp"] = TensorProduct(ndim=ndim, func="exp")
        problems["tanh"] = TensorProduct(ndim=ndim, func="tanh")
        problems["cos"] = TensorProduct(ndim=ndim, func="cos")
        sms = OrderedDict()
        sms["KPLS"] = KPLS(eval_n_comp=True)

        t_errors = {}
        e_errors = {}
        t_errors["KPLS"] = 1e-3
        e_errors["KPLS"] = 2.5

        n_comp_opt = {}
        n_comp_opt["Branin"] = 2
        n_comp_opt["Rosenbrock"] = 1
        n_comp_opt["sphere"] = 1
        n_comp_opt["exp"] = 3
        n_comp_opt["tanh"] = 1
        n_comp_opt["cos"] = 1

        self.nt = nt
        self.ne = ne
        self.problems = problems
        self.sms = sms
        self.t_errors = t_errors
        self.e_errors = e_errors
        self.n_comp_opt = n_comp_opt

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

        l = sm.options["n_comp"]

        t_error = compute_rms_error(sm)
        e_error = compute_rms_error(sm, xe, ye)

        if print_output:
            print("%8s %6s %18.9e %18.9e" % (pname[:6], sname, t_error, e_error))

        self.assert_error(t_error, 0.0, self.t_errors[sname], 1e-5)
        self.assert_error(e_error, 0.0, self.e_errors[sname], 1e-5)
        self.assertEqual(l, self.n_comp_opt[pname])

    # --------------------------------------------------------------------
    # Function: sphere

    def test_Branin_KPLS(self):
        self.run_test()

    def test_Rosenbrock_KPLS(self):
        self.run_test()

    def test_sphere_KPLS(self):
        self.run_test()

    def test_exp_KPLS(self):
        self.run_test()

    def test_tanh_KPLS(self):
        self.run_test()

    def test_cos_KPLS(self):
        self.run_test()


if __name__ == "__main__":
    print_output = True
    print("%6s %8s %18s %18s" % ("SM", "Problem", "Train. pt. error", "Test pt. error"))
    unittest.main()
