"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

import unittest
from collections import OrderedDict

import numpy as np

from smt.problems import Sphere
from smt.sampling_methods import LHS
from smt.utils.design_space import DesignSpace
from smt.utils.silence import Silence
from smt.utils.sm_test_case import SMTestCase

try:
    from smt.surrogate_models import RMTB, RMTC

    COMPILED_AVAILABLE = True
except ImportError:
    COMPILED_AVAILABLE = False


print_output = False


class Test(SMTestCase):
    def setUp(self):
        ndim = 3
        nt = 500
        ne = 100

        problems = OrderedDict()
        problems["sphere"] = Sphere(ndim=ndim)

        sms = OrderedDict()
        if COMPILED_AVAILABLE:
            sms["RMTC"] = RMTC(num_elements=6, extrapolate=True)
            sms["RMTB"] = RMTB(order=4, num_ctrl_pts=10, extrapolate=True)

        self.nt = nt
        self.ne = ne
        self.problems = problems
        self.sms = sms

    def run_test(self, sname, extrap_train=False, extrap_predict=False):
        prob = self.problems["sphere"]
        sampling = LHS(xlimits=prob.xlimits)

        np.random.seed(0)
        xt = sampling(self.nt)
        yt = prob(xt)

        sm0 = self.sms[sname]

        sm = sm0.__class__()
        sm.options = sm0.options.clone()
        if sm.options.is_declared("design_space"):
            sm.options["design_space"] = DesignSpace(prob.xlimits)
        if sm.options.is_declared("xlimits"):
            sm.options["xlimits"] = prob.xlimits
        sm.options["print_global"] = False

        x = np.zeros((1, xt.shape[1]))
        x[0, :] = prob.xlimits[:, 1] + 1.0
        y = prob(x)

        sm.set_training_values(xt, yt)
        if extrap_train:
            sm.set_training_values(x, y)

        with Silence():
            sm.train()

        if extrap_predict:
            sm.predict_values(x)

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_RMTC(self):
        self.run_test("RMTC", False, False)

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_RMTC_train(self):
        with self.assertRaises(Exception) as context:
            self.run_test("RMTC", True, False)
        self.assertEqual(str(context.exception), "Training points above max for 0")

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_RMTC_predict(self):
        self.run_test("RMTC", False, True)

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_RMTB(self):
        self.run_test("RMTB", False, False)

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_RMTB_train(self):
        with self.assertRaises(Exception) as context:
            self.run_test("RMTB", True, False)
        self.assertEqual(str(context.exception), "Training points above max for 0")

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_RMTB_predict(self):
        self.run_test("RMTB", False, True)


if __name__ == "__main__":
    unittest.main()
