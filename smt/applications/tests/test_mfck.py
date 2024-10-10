# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:17:04 2024

@author: mcastano
"""
import unittest

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    NO_MATPLOTLIB = False
except ImportError:
    NO_MATPLOTLIB = True

from copy import deepcopy

from smt.applications.mfck import MFCK
from smt.problems import TensorProduct
from smt.sampling_methods import FullFactorial
from smt.utils.silence import Silence
from smt.utils.sm_test_case import SMTestCase

print_output = False


class TestMFCK(SMTestCase):
    def setUp(self):
        self.nt = 100
        self.ne = 100
        self.ndim = 3

    def test_mfck(self):
        self.problems = ["exp"]  # , "tanh", "cos"]

        for fname in self.problems:
            prob = TensorProduct(ndim=self.ndim, func=fname)
            sampling = FullFactorial(xlimits=prob.xlimits, clip=True)

            np.random.seed(0)
            xt = sampling(self.nt)
            yt = prob(xt)
            for i in range(self.ndim):
                yt = np.concatenate((yt, prob(xt, kx=i)), axis=1)

            y_lf = 2 * prob(xt) + 2
            x_lf = deepcopy(xt)
            np.random.seed(1)

            sm = MFCK(theta0=[1e-2] * self.ndim)
            if sm.options.is_declared("xlimits"):
                sm.options["xlimits"] = prob.xlimits
            sm.options["print_global"] = False

            sm.set_training_values(xt, yt[:, 0])
            sm.set_training_values(x_lf, y_lf[:, 0], name=0)

            with Silence():
                sm.train()

            mean1, cov1, mean, cov = sm.predict(xt)

            m,c = sm.predict_multi_lvl(xt)

            num1 = np.linalg.norm( m[1][:,0] - yt[:,0])
            den1 = np.linalg.norm(yt[:,0])

            t_error1 = num1 / den1

            num = np.linalg.norm( mean1[:,0] - yt[:,0])
            den = np.linalg.norm(yt[:,0])

            t_error = num / den

            self.assert_error(t_error1, 0.0, 1e-4,1e-4)
            self.assert_error(t_error, 0.0, 1e-4,1e-4)


    @staticmethod
    def run_mfck_example():
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.applications.mfck import MFCK
        from smt.applications.mfk import NestedLHS

        # low fidelity model
        def lf_function(x):
            import numpy as np

            return (
                0.5 * ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)
                + (x - 0.5) * 10.0
                - 5
            )

        # high fidelity model
        def hf_function(x):
            import numpy as np

            return ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)

        # Problem set up
        xlimits = np.array([[0.0, 1.0]])
        xdoes = NestedLHS(nlevel=2, xlimits=xlimits, random_state=0)
        xt_c, xt_e = xdoes(7)

        # Evaluate the HF and LF functions
        yt_e = hf_function(xt_e)
        yt_c = lf_function(xt_c)

        sm = MFCK(theta0=xt_e.shape[1] * [1.0], corr="squar_exp")

        # low-fidelity dataset names being integers from 0 to level-1
        sm.set_training_values(xt_c, yt_c, name=0)
        # high-fidelity dataset without name
        sm.set_training_values(xt_e, yt_e)

        # train the model
        sm.train()

        x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

        mean1,cov1,mean,cov = sm.predict(x)

        plt.figure()

        plt.plot(x, hf_function(x), label="reference")
        plt.plot(x, mean1, linestyle="-.", label="mean_gp")
        plt.scatter(xt_e, yt_e, marker="o", color="k", label="HF doe")
        plt.scatter(xt_c, yt_c, marker="*", color="g", label="LF doe")

        plt.legend(loc=0)
        plt.ylim(-10, 17)
        plt.xlim(-0.1, 1.1)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

        plt.show()

    # run scripts are used in documentation as documentation is not always rebuild
    # make a test run by pytest to test the run scripts
    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_run_mfck_example(self):
        self.run_mfck_example()


if __name__ == "__main__":
    unittest.main()
