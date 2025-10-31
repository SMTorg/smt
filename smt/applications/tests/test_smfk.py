# -*- coding: utf-8 -*-
"""
Created on March 05 2025

@author: m.castano
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

from smt.applications.smfk import SMFK
from smt.problems import TensorProduct
from smt.sampling_methods import FullFactorial
from smt.utils.silence import Silence
from smt.utils.sm_test_case import SMTestCase

print_output = False


class TestSMFK(SMTestCase):
    def setUp(self):
        self.nt = 100
        self.ne = 100
        self.ndim = 3

    def test_smfk(self):
        self.problems = ["exp"]  # , "tanh", "cos"]

        for fname in self.problems:
            prob = TensorProduct(ndim=self.ndim, func=fname)
            sampling = FullFactorial(xlimits=prob.xlimits, clip=True)

            xt = sampling(self.nt)
            yt = prob(xt)
            for i in range(self.ndim):
                yt = np.concatenate((yt, prob(xt, kx=i)), axis=1)

            y_lf = 2 * prob(xt) + 2
            x_lf = deepcopy(xt)
            xe = sampling(self.ne)
            ye = prob(xe)

            sm = SMFK(theta0=[1e-2] * self.ndim, n_inducing=xe.shape[0], seed=42)
            if sm.options.is_declared("xlimits"):
                sm.options["xlimits"] = prob.xlimits
            sm.options["print_global"] = False

            sm.set_training_values(xe, ye[:, 0])
            sm.set_training_values(x_lf, y_lf[:, 0], name=0)

            with Silence():
                sm.train()

            m = sm.predict_values(xt)

            num = np.linalg.norm(m[:, 0] - yt[:, 0])
            den = np.linalg.norm(yt[:, 0])

            t_error = num / den

            self.assert_error(t_error, 0.0, 1e-5, 1e-5)

    @staticmethod
    def run_smfk_example():
        import matplotlib.pyplot as plt
        import numpy as np
        from smt.applications.mfk import NestedLHS
        from smt.applications.smfk import SMFK

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
        xdoes = NestedLHS(nlevel=2, xlimits=xlimits, seed=0)
        xt_c, xt_e = xdoes(7)

        # Evaluate the HF and LF functions
        yt_e = hf_function(xt_e)
        yt_c = lf_function(xt_c)

        sm = SMFK(
            theta0=xt_e.shape[1] * [1.0], corr="squar_exp", n_inducing=xt_e.shape[0]
        )

        # low-fidelity dataset names being integers from 0 to level-1
        sm.set_training_values(xt_c, yt_c, name=0)
        # high-fidelity dataset without name
        sm.set_training_values(xt_e, yt_e)

        # train the model
        sm.train()

        x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

        # query the outputs
        y = sm.predict_values(x)
        varAl, _ = sm.predict_variances_all_levels(x)
        # _derivs = sm.predict_derivatives(x, kx=0)

        plt.figure()

        plt.plot(x, hf_function(x), label="reference")
        plt.plot(x, y, linestyle="-.", label="mean_gp")
        plt.scatter(xt_e, yt_e, marker="o", color="k", label="HF doe")
        plt.scatter(xt_c, yt_c, marker="*", color="g", label="LF doe")
        plt.plot(
            sm.Z,
            -9.9 * np.ones_like(sm.Z),
            "g|",
            mew=2,
            label=f"LF inducing:{sm.Z.shape[0]}",
        )

        plt.legend(loc=0)
        plt.ylim(-10, 17)
        plt.xlim(-0.1, 1.1)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

        plt.show()

    # run scripts are used in documentation as documentation is not always rebuild
    # make a test run by pytest to test the run scripts
    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_run_smfk_example(self):
        self.run_smfk_example()


if __name__ == "__main__":
    unittest.main()
