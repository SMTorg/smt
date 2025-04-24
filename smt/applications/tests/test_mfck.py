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

            sm = MFCK(hyper_opt="Cobyla", eval_noise=False)
            if sm.options.is_declared("xlimits"):
                sm.options["xlimits"] = prob.xlimits
            sm.options["print_global"] = False

            sm.set_training_values(xt, yt[:, 0])
            sm.set_training_values(x_lf, y_lf[:, 0], name=0)

            with Silence():
                sm.train()

            m = sm.predict_values(xt)

            num = np.linalg.norm(m[:, 0] - yt[:, 0])
            den = np.linalg.norm(yt[:, 0])

            t_error = num / den

            self.assert_error(t_error, 0.0, 1e-5, 1e-5)

    @staticmethod
    def run_mfck_example():
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.applications.mfck import MFCK
        from smt.sampling_methods import LHS

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

        rnd_state = 1

        # Problem set up
        xlimits = np.array([[0.0, 1.0]])

        # Example with non-nested input data
        Obs_HF = 7  # Number of observations of HF
        Obs_LF = 14  # Number of observations of LF

        # Creation of LHS for non-nested LF data
        sampling = LHS(
            xlimits=xlimits,
            criterion="ese",
            random_state=rnd_state,
        )

        xt_e_non = sampling(Obs_HF)
        xt_c_non = sampling(Obs_LF)

        # Evaluate the LF function
        yt_e_non = hf_function(xt_e_non)
        yt_c_non = lf_function(xt_c_non)

        sm_non_nested = MFCK(
            theta0=xt_e_non.shape[1] * [0.5],
            theta_bounds=[1e-2, 100],
            corr="squar_exp",
            eval_noise=False,
        )
        sm_non_nested.options["lambda"] = 0.0  # Without regularization

        # low-fidelity dataset names being integers from 0 to level-1
        sm_non_nested.set_training_values(xt_c_non, yt_c_non, name=0)
        # high-fidelity dataset without name
        sm_non_nested.set_training_values(xt_e_non, yt_e_non)

        # train the model
        sm_non_nested.train()

        x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

        m_non_nested, c_non_nested = sm_non_nested.predict_all_levels(x)

        plt.figure()
        plt.title("Example with non-nested input data")
        plt.plot(x, hf_function(x), label="reference HF")
        plt.plot(x, lf_function(x), label="reference LF")
        plt.plot(x, m_non_nested[1], linestyle="-.", label="mean_gp_non_nested")
        plt.scatter(xt_e_non, yt_e_non, marker="o", color="k", label="HF doe")
        plt.scatter(
            xt_c_non, yt_c_non, marker="*", color="c", label="LF non-nested doe"
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
    def test_run_mfck_example(self):
        self.run_mfck_example()


if __name__ == "__main__":
    unittest.main()
