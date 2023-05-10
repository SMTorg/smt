"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

import numpy as np
import unittest
import inspect

from collections import OrderedDict

from smt.problems import Sphere
from smt.sampling_methods import LHS

from smt.utils.sm_test_case import SMTestCase
from smt.utils.silence import Silence
from smt.utils import compute_rms_error
from smt.utils.design_space import DesignSpace

from smt.applications import MFK

try:
    from smt.surrogate_models import IDW, RBF, RMTC, RMTB

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
        problems["sphere"] = Sphere(ndim=ndim)

        sms = OrderedDict()
        if compiled_available:
            sms["RBF"] = RBF()
            sms["RMTC"] = RMTC()
            sms["RMTB"] = RMTB()
            sms["MFK"] = MFK(theta0=[1e-2] * ndim)

        self.nt = nt
        self.ne = ne
        self.problems = problems
        self.sms = sms

    def run_test(self):
        method_name = inspect.stack()[1][3]
        pname = method_name.split("_")[1]
        sname = method_name.split("_")[2]

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
        e_error0 = compute_rms_error(sm, xe, dye[0], 0)
        e_error1 = compute_rms_error(sm, xe, dye[1], 1)

        if print_output:
            print(
                "%8s %6s %18.9e %18.9e %18.9e %18.9e"
                % (pname[:6], sname, t_error, e_error, e_error0, e_error1)
            )

        self.assert_error(e_error0, 0.0, 25e-1)
        self.assert_error(e_error1, 0.0, 25e-1)

    # --------------------------------------------------------------------
    # Function: sphere

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_sphere_RBF(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_sphere_RMTC(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_sphere_RMTB(self):
        self.run_test()


if __name__ == "__main__":
    print_output = True
    print(
        "%6s %8s %18s %18s %18s %18s"
        % (
            "SM",
            "Problem",
            "Train. pt. error",
            "Test pt. error",
            "Deriv 0 error",
            "Deriv 1 error",
        )
    )
    unittest.main()
