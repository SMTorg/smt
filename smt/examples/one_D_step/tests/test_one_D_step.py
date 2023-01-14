import unittest

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from smt.examples.one_D_step.one_D_step import get_one_d_step, plot_one_d_step

try:
    from smt.surrogate_models import RMTB, RMTC

    compiled_available = True
except:
    compiled_available = False


class Test(unittest.TestCase):
    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtb(self):
        from smt.surrogate_models import RMTB

        xt, yt, xlimits = get_one_d_step()

        interp = RMTB(
            num_ctrl_pts=100,
            xspecs={"xlimits": xlimits},
            nonlinear_maxiter=20,
            solver_tolerance=1e-16,
            energy_weight=1e-14,
            regularization_weight=0.0,
        )
        interp.set_training_values(xt, yt)
        interp.train()

        plot_one_d_step(xt, yt, xlimits, interp)

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtc(self):
        from smt.surrogate_models import RMTC

        xt, yt, xlimits = get_one_d_step()

        interp = RMTC(
            num_elements=40,
            xspecs={"xlimits": xlimits},
            nonlinear_maxiter=20,
            solver_tolerance=1e-16,
            energy_weight=1e-14,
            regularization_weight=0.0,
        )
        interp.set_training_values(xt, yt)
        interp.train()

        plot_one_d_step(xt, yt, xlimits, interp)


if __name__ == "__main__":
    Test().test_rmtb()
    plt.savefig("testRMTB.pdf")
    Test().test_rmtc()
    plt.savefig("testRMTC.pdf")
