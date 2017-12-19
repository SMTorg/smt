import unittest

import matplotlib
matplotlib.use('Agg')

try:
    from smt.surrogate import RMTB, RMTC
    compiled_available = True
except:
    compiled_available = False


class Test(unittest.TestCase):

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtb(self):
        from smt.surrogate import RMTB
        from smt.examples.b777_engine import get_b777_engine, plot_b777_engine

        xt, yt, dyt_dxt, xlimits = get_b777_engine()

        interp = RMTB(num_ctrl_pts=15, xlimits=xlimits, nonlinear_maxiter=20, approx_order=2,
            energy_weight=0e-14, regularization_weight=0e-18, extrapolate=True,
        )
        interp.set_training_values(xt, yt)
        interp.set_training_derivatives(xt, dyt_dxt[:, :, 0], 0)
        interp.set_training_derivatives(xt, dyt_dxt[:, :, 1], 1)
        interp.set_training_derivatives(xt, dyt_dxt[:, :, 2], 2)
        interp.train()

        plot_b777_engine(xt, yt, xlimits, interp)

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtc(self):
        from smt.surrogate import RMTC
        from smt.examples.b777_engine import get_b777_engine, plot_b777_engine

        xt, yt, dyt_dxt, xlimits = get_b777_engine()

        interp = RMTC(num_elements=6, xlimits=xlimits, nonlinear_maxiter=20, approx_order=2,
            energy_weight=0., regularization_weight=0., extrapolate=True,
        )
        interp.set_training_values(xt, yt)
        interp.set_training_derivatives(xt, dyt_dxt[:, :, 0], 0)
        interp.set_training_derivatives(xt, dyt_dxt[:, :, 1], 1)
        interp.set_training_derivatives(xt, dyt_dxt[:, :, 2], 2)
        interp.train()

        plot_b777_engine(xt, yt, xlimits, interp)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    Test().test_rmtc()
    plt.savefig('test.pdf')
