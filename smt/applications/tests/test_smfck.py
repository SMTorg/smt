# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:22:32 2025

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

from smt.applications import SMFCK
from smt.sampling_methods import LHS
from smt.problems import TensorProduct
from smt.sampling_methods import FullFactorial
from smt.utils.sm_test_case import SMTestCase

print_output = False


class TestSMFCK(SMTestCase):
    def setUp(self):
        self.nt = 100
        self.ne = 100
        self.ndim = 3

    def test_smfck(self):
        self.problems = ["exp"]  # , "tanh", "cos"]

        for fname in self.problems:
            prob = TensorProduct(ndim=self.ndim, func=fname)
            sampling = FullFactorial(xlimits=prob.xlimits, clip=True)

            noise_std = 1e-5

            np.random.seed(0)
            xt = sampling(self.nt)
            yt = prob(xt)
            for i in range(self.ndim):
                yt = np.concatenate((yt, prob(xt, kx=i)), axis=1)

            y_lf = 2 * prob(xt) + 2 + np.random.normal(0, noise_std, size=xt.shape)
            x_lf = deepcopy(xt)
            np.random.seed(1)
            xe = sampling(self.ne)
            ye = prob(xe) + +np.random.normal(0, noise_std, size=xe.shape)

            sm = SMFCK(
                hyper_opt="Cobyla",
                theta0=xe.shape[1] * [0.8],
                theta_bounds=[1e-6, 2.0],
                print_global=False,
                eval_noise=True,
                noise0=[1e-5],
                noise_bounds=np.array((1e-12, 10.0)),
                corr="squar_exp",
                n_inducing=[x_lf.shape[0] - 1, xe.shape[0] - 1],
                n_start=1,
            )
            # if sm.options.is_declared("xlimits"):
            #    sm.options["xlimits"] = prob.xlimits
            sm.options["print_global"] = False

            sm.set_training_values(xe, ye[:, 0])
            sm.set_training_values(x_lf, y_lf[:, 0], name=0)

            sm.train()

            m = sm.predict_values(xt)

            num = np.linalg.norm(m[:, 0] - yt[:, 0])
            den = np.linalg.norm(yt[:, 0])

            t_error = num / den

            self.assert_error(t_error, 0.0, 5e-2, 5e-2)

    @staticmethod
    def run_smfck_example():
        import matplotlib.pyplot as plt
        import numpy as np

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
        # Example with non-nested input data
        Obs_HF = 7  # Number of observations of HF
        Obs_LF = 14  # Number of observations of LF

        # Creation of LHS for non-nested LF data
        sampling = LHS(
            xlimits=xlimits,
            criterion="ese",
            random_state=0,
        )

        xt_e_non = sampling(Obs_HF)
        xt_c_non = sampling(Obs_LF)

        # Evaluate the HF and LF functions
        yt_e = hf_function(xt_e_non)
        yt_c = lf_function(xt_c_non)

        sm = SMFCK(
            hyper_opt="Cobyla",
            theta0=xt_e_non.shape[1] * [1.0],
            theta_bounds=[1e-6, 50.0],
            print_global=False,
            eval_noise=True,
            noise0=[1e-4],
            noise_bounds=np.array((1e-12, 100)),
            corr="squar_exp",
            n_inducing=[xt_c_non.shape[0] - 2, xt_e_non.shape[0] - 1],
        )

        # low-fidelity dataset names being integers from 0 to level-1
        sm.set_training_values(xt_c_non, yt_c, name=0)
        # high-fidelity dataset without name
        sm.set_training_values(xt_e_non, yt_e)

        # train the model
        sm.train()

        x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

        # query the outputs

        mean, cov = sm.predict_all_levels(x)

        y = mean[-1]
        # _derivs = sm.predict_derivatives(x, kx=0)

        plt.figure()

        plt.plot(x, hf_function(x), label="reference")
        plt.plot(x, y, linestyle="-.", label="mean_gp")
        plt.scatter(xt_e_non, yt_e, marker="o", color="k", label="HF doe")
        plt.scatter(xt_c_non, yt_c, marker="*", color="g", label="LF doe")
        plt.plot(
            sm.Z[0],
            -9.9 * np.ones_like(sm.Z[0]),
            "g|",
            mew=2,
            label=f"LF inducing:{sm.Z[0].shape[0]}",
        )
        plt.plot(
            sm.Z[1],
            -9.9 * np.ones_like(sm.Z[1]),
            "k|",
            mew=2,
            label=f"HF inducing:{sm.Z[1].shape[0]}",
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
    def test_run_smfck_example(self):
        self.run_smfck_example()


if __name__ == "__main__":
    unittest.main()
