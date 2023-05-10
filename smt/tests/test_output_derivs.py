"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

import numpy as np
import unittest
import inspect

from collections import OrderedDict
from smt.utils.design_space import DesignSpace

from smt.problems import Sphere
from smt.sampling_methods import FullFactorial
from smt.utils.sm_test_case import SMTestCase
from smt.utils.silence import Silence
from smt.utils import compute_rms_error

try:
    from smt.surrogate_models import IDW, RBF, RMTC, RMTB

    compiled_available = True
except:
    compiled_available = False


print_output = False


class Test(SMTestCase):
    def setUp(self):
        ndim = 2
        self.nt = 50
        self.ne = 10

        self.problem = Sphere(ndim=ndim)

        self.sms = sms = OrderedDict()
        if compiled_available:
            sms["IDW"] = IDW()
            sms["RBF"] = RBF()
            sms["RMTB"] = RMTB(
                regularization_weight=1e-8,
                nonlinear_maxiter=100,
                solver_tolerance=1e-16,
            )
            sms["RMTC"] = RMTC(
                regularization_weight=1e-8,
                nonlinear_maxiter=100,
                solver_tolerance=1e-16,
            )

    def run_test(self):
        method_name = inspect.stack()[1][3]
        sname = method_name.split("_")[1]

        prob = self.problem
        sampling = FullFactorial(xlimits=prob.xlimits, clip=False)

        np.random.seed(0)
        xt = sampling(self.nt)
        yt = prob(xt)
        # dyt = {}
        # for kx in range(prob.xlimits.shape[0]):
        #     dyt[kx] = prob(xt, kx=kx)

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

        sm.update_training_values(yt)
        with Silence():
            sm.train()
        ye0 = sm.predict_values(xe)

        h = 1e-3
        jac_fd = np.zeros((self.ne, self.nt))
        for ind in range(self.nt):
            sm.update_training_values(yt + h * np.eye(self.nt, M=1, k=-ind))
            with Silence():
                sm.train()
            ye = sm.predict_values(xe)

            jac_fd[:, ind] = (ye - ye0)[:, 0] / h

        jac_fd = jac_fd.reshape((self.ne, self.nt, 1))
        jac_an = sm.predict_output_derivatives(xe)[None]

        if print_output:
            print(np.linalg.norm(jac_fd - jac_an))

        self.assert_error(jac_fd, jac_an, rtol=5e-2)

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_IDW(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_RBF(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_RMTB(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, "Compiled Fortran libraries not available")
    def test_RMTC(self):
        self.run_test()


if __name__ == "__main__":
    print_output = True
    unittest.main()
