try:
    import matplotlib

    matplotlib.use("Agg")
    NO_MATPLOTLIB = False
except ImportError:
    NO_MATPLOTLIB = True

import unittest

import numpy as np

from smt.applications.mfk import MFK
from smt.problems import TensorProduct
from smt.sampling_methods import LHS
from smt.utils.misc import compute_relative_error
from smt.utils.silence import Silence
from smt.utils.sm_test_case import SMTestCase

print_output = False


class TestMFKOneFidelity(SMTestCase):
    def setUp(self):
        self.nt = 20
        self.ne = 50
        self.ndim = 1

    def test_mfk_1fidelity(self):
        self.problems = ["exp", "tanh", "cos"]

        for fname in self.problems:
            prob = TensorProduct(ndim=self.ndim, func=fname)
            sampling = LHS(xlimits=prob.xlimits, random_state=0)

            np.random.seed(0)
            xt = sampling(self.nt)
            yt = prob(xt)
            for i in range(self.ndim):
                yt = np.concatenate((yt, prob(xt, kx=i)), axis=1)

            sampling = LHS(xlimits=prob.xlimits, random_state=1)
            xv = sampling(self.ne)
            yv = prob(xv)

            sm = MFK(
                theta0=[1e-2] * self.ndim,
                print_global=False,
            )

            sm.set_training_values(xt, yt[:, 0])

            with Silence():
                sm.train()

            t_error = compute_relative_error(sm)
            e_error = compute_relative_error(sm, xv, yv)

            self.assert_error(t_error, 0.0, 3e-3)
            self.assert_error(e_error, 0.0, 3e-3)

    @staticmethod
    def run_mfk_example_1fidelity():
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.applications.mfk import MFK, NestedLHS

        # Consider only 1 fidelity level
        # high fidelity model
        def hf_function(x):
            import numpy as np

            return ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)

        # Problem set up
        xlimits = np.array([[0.0, 1.0]])
        xdoes = NestedLHS(nlevel=1, xlimits=xlimits, random_state=0)
        xt_e = xdoes(7)[0]

        # Evaluate the HF function
        yt_e = hf_function(xt_e)

        sm = MFK(theta0=xt_e.shape[1] * [1.0], corr="squar_exp")

        # High-fidelity dataset without name
        sm.set_training_values(xt_e, yt_e)

        # Train the model
        sm.train()

        x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

        # Query the outputs
        y = sm.predict_values(x)
        _mse = sm.predict_variances(x)
        _derivs = sm.predict_derivatives(x, kx=0)

        plt.figure()

        plt.plot(x, hf_function(x), label="reference")
        plt.plot(x, y, linestyle="-.", label="mean_gp")
        plt.scatter(xt_e, yt_e, marker="o", color="k", label="HF doe")

        plt.legend(loc=0)
        plt.ylim(-10, 17)
        plt.xlim(-0.1, 1.1)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

        plt.show()

    # run scripts are used in documentation as documentation is not always rebuild
    # make a test run by pytest to test the run scripts
    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_run_mfk_example_1fidelity(self):
        self.run_mfk_example_1fidelity()


if __name__ == "__main__":
    unittest.main()
