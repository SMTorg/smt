import matplotlib

matplotlib.use("Agg")

import unittest
import numpy as np
import unittest

from smt.problems import TensorProduct
from smt.sampling_methods import LHS

from smt.utils.sm_test_case import SMTestCase
from smt.utils.silence import Silence
from smt.utils import compute_rms_error
from smt.applications.mfk import MFK

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

            t_error = compute_rms_error(sm)
            e_error = compute_rms_error(sm, xv, yv)

            self.assert_error(t_error, 0.0, 1e-6)
            self.assert_error(e_error, 0.0, 1e-6)

    @staticmethod
    def run_mfk_example_1fidelity():
        import numpy as np
        import matplotlib.pyplot as plt
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

        sm = MFK(theta0=xt_e.shape[1] * [1.0])

        # High-fidelity dataset without name
        sm.set_training_values(xt_e, yt_e)

        # Train the model
        sm.train()

        x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

        # Query the outputs
        y = sm.predict_values(x)
        mse = sm.predict_variances(x)
        derivs = sm.predict_derivatives(x, kx=0)

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


if __name__ == "__main__":
    unittest.main()
