import unittest

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

try:
    from smt.surrogate_models import RMTB, RMTC

    compiled_available = True
except:
    compiled_available = False

from smt.examples.b777_engine.b777_engine import get_b777_engine, plot_b777_engine


class Test(unittest.TestCase):
    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtb(self):
        from smt.surrogate_models import RMTB

        xt, yt, dyt_dxt, xlimits = get_b777_engine()

        interp = RMTB(
            num_ctrl_pts=15,
            xspecs={"xlimits": xlimits},
            nonlinear_maxiter=20,
            approx_order=2,
            energy_weight=0e-14,
            regularization_weight=0e-18,
            extrapolate=True,
        )
        interp.set_training_values(xt, yt)
        interp.set_training_derivatives(xt, dyt_dxt[:, :, 0], 0)
        interp.set_training_derivatives(xt, dyt_dxt[:, :, 1], 1)
        interp.set_training_derivatives(xt, dyt_dxt[:, :, 2], 2)
        interp.train()

        plot_b777_engine(xt, yt, xlimits, interp)

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtc(self):
        from smt.surrogate_models import RMTC

        xt, yt, dyt_dxt, xlimits = get_b777_engine()

        interp = RMTC(
            num_elements=6,
            xspecs={"xlimits": xlimits},
            nonlinear_maxiter=20,
            approx_order=2,
            energy_weight=0.0,
            regularization_weight=0.0,
            extrapolate=True,
        )
        interp.set_training_values(xt, yt)
        interp.set_training_derivatives(xt, dyt_dxt[:, :, 0], 0)
        interp.set_training_derivatives(xt, dyt_dxt[:, :, 1], 1)
        interp.set_training_derivatives(xt, dyt_dxt[:, :, 2], 2)
        interp.train()

        plot_b777_engine(xt, yt, xlimits, interp)


if __name__ == "__main__":
    Test().test_rmtb()
    plt.savefig("testRMTB.pdf")
    Test().test_rmtc()
    plt.savefig("testRMTC.pdf")
