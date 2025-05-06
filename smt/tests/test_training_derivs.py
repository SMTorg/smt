"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

import inspect
import unittest
from collections import OrderedDict

import numpy as np

from smt.design_space import (
    DesignSpace,
)
from smt.problems import Sphere, TensorProduct
from smt.sampling_methods import FullFactorial
from smt.utils.misc import compute_relative_error
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
        nt = 5000
        ne = 500

        problems = OrderedDict()
        problems["sphere"] = Sphere(ndim=ndim)
        problems["exp"] = TensorProduct(ndim=ndim, func="exp")
        problems["tanh"] = TensorProduct(ndim=ndim, func="tanh")
        problems["cos"] = TensorProduct(ndim=ndim, func="cos")

        sms = OrderedDict()
        if COMPILED_AVAILABLE:
            sms["RMTC"] = RMTC()
            sms["RMTB"] = RMTB()

        t_errors = {}
        t_errors["RMTC"] = 1e-1
        t_errors["RMTB"] = 1e-1

        e_errors = {}
        e_errors["RMTC"] = 1e-1
        e_errors["RMTB"] = 1e-1

        ge_t_errors = {}
        ge_t_errors["RMTC"] = 1e-2
        ge_t_errors["RMTB"] = 1e-2

        ge_e_errors = {}
        ge_e_errors["RMTC"] = 1e-2
        ge_e_errors["RMTB"] = 1e-2

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
        pname = method_name.split("_")[1]
        sname = method_name.split("_")[2]

        prob = self.problems[pname]
        sampling = FullFactorial(xlimits=prob.xlimits, clip=True)

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

        t_error = compute_relative_error(sm)
        e_error = compute_relative_error(sm, xe, ye)

        sm = sm0.__class__()
        sm.options = sm0.options.clone()
        if sm.options.is_declared("design_space"):
            sm.options["design_space"] = DesignSpace(prob.xlimits)
        if sm.options.is_declared("xlimits"):
            sm.options["xlimits"] = prob.xlimits
        sm.options["print_global"] = False

        sm.set_training_values(xt, yt)
        for kx in range(prob.xlimits.shape[0]):
            sm.set_training_derivatives(xt, dyt[kx], kx)

        with Silence():
            sm.train()

        ge_t_error = compute_relative_error(sm)
        ge_e_error = compute_relative_error(sm, xe, ye)

        if print_output:
            print(
                "%8s %6s %18.9e %18.9e %18.9e %18.9e"
                % (pname[:6], sname, t_error, e_error, ge_t_error, ge_e_error)
            )

        self.assert_error(t_error, 0.0, self.t_errors[sname])
        self.assert_error(e_error, 0.0, self.e_errors[sname])
        self.assert_error(ge_t_error, 0.0, self.ge_t_errors[sname])
        self.assert_error(ge_e_error, 0.0, self.ge_e_errors[sname])

    # --------------------------------------------------------------------
    # Function: sphere

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_sphere_RMTC(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_sphere_RMTB(self):
        self.run_test()

    # --------------------------------------------------------------------
    # Function: exp

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_exp_RMTC(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_exp_RMTB(self):
        self.run_test()

    # --------------------------------------------------------------------
    # Function: tanh

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_tanh_RMTC(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_tanh_RMTB(self):
        self.run_test()

    # --------------------------------------------------------------------
    # Function: cos

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_cos_RMTC(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_cos_RMTB(self):
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
            "GE tr. pt. error",
            "GE test pt. error",
        )
    )
    unittest.main()
