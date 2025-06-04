"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest
import numpy as np

from smt.sampling_methods import LHS
from smt.surrogate_models import GPX, KRG
from smt.surrogate_models.gpx import GPX_AVAILABLE


class TestKRG(unittest.TestCase):
    def test_predict_output_shape(self):
        d, n = (3, 10)
        sx = LHS(
            xlimits=np.repeat(np.atleast_2d([0.0, 1.0]), d, axis=0),
            criterion="m",
            random_state=42,
        )
        x = sx(n)
        # 2-dimensional output
        n_s = 2
        sy = LHS(
            xlimits=np.repeat(np.atleast_2d([0.0, 1.0]), n_s, axis=0),
            criterion="m",
            random_state=42,
        )
        y = sy(n)

        kriging = KRG(poly="linear")
        kriging.set_training_values(x, y)
        kriging.train()

        val = kriging.predict_values(x)
        self.assertEqual(y.shape, val.shape)

        var = kriging.predict_variances(x)
        self.assertEqual(y.shape, var.shape)

        for kx in range(d):
            val_deriv = kriging.predict_derivatives(x, kx)
            self.assertEqual(y.shape, val_deriv.shape)
            var_deriv = kriging.predict_variance_derivatives(x, 0)
            self.assertEqual((n, n_s), var_deriv.shape)

    @unittest.skipIf(not GPX_AVAILABLE, "GPX not available")
    def test_predict_var_derivs_GPX(self):
        def target_fun(x):
            return np.cos(5 * x)

        def pb(x):
            # sin + linear trend
            y = (
                np.atleast_2d(np.sin(x[:, 0])).T
                + np.atleast_2d(2 * x[:, 0] + 5 * x[:, 1]).T
                + 10
            )  # + linear trend
            return y

        xlimits = np.array([[-5, 10], [-5, 10]])
        sampling = LHS(xlimits=xlimits, random_state=42)
        xt = sampling(12)
        yt = pb(xt)

        sm_gpx = GPX(
            theta0=[0.01],
            print_global=False,
            poly="constant",
            corr="abs_exp",
            seed=42,
        )
        sm_krg = KRG(
            theta0=[0.01],
            print_global=False,
            poly="constant",
            corr="abs_exp",
            random_state=42,
        )
        # train
        sm_gpx.set_training_values(xt, yt)
        sm_gpx.train()
        sm_krg.set_training_values(xt, yt)
        sm_krg.train()

        # Predicted
        e = 2.5e-11
        xa = -1.3
        xb = 2.5
        x_valid = np.array(
            [[xa, xb], [xa + e, xb], [xa - e, xb], [xa, xb + e], [xa, xb - e]]
        )
        y_predicted_gpx = sm_gpx.predict_variances(x_valid)
        y_predicted_krg = sm_krg.predict_variances(x_valid)
        x = np.atleast_2d(x_valid[0])

        # diff
        diff_g_gpx = (y_predicted_gpx[1, 0] - y_predicted_gpx[2, 0]) / (2 * e)
        diff_d_gpx = (y_predicted_gpx[3, 0] - y_predicted_gpx[4, 0]) / (2 * e)
        diff_g_krg = (y_predicted_krg[1, 0] - y_predicted_krg[2, 0]) / (2 * e)
        diff_d_krg = (y_predicted_krg[3, 0] - y_predicted_krg[4, 0]) / (2 * e)
        diff_gpx = [diff_g_gpx, diff_d_gpx]
        diff_krg = [diff_g_krg, diff_d_krg]
        deriv_gpx = np.array(
            [
                sm_gpx.predict_variance_derivatives(x, 0)[0],
                sm_gpx.predict_variance_derivatives(x, 1)[0],
            ]
        ).T

        deriv_krg = np.array(
            [
                sm_krg.predict_variance_derivatives(x, 0)[0],
                sm_krg.predict_variance_derivatives(x, 1)[0],
            ]
        ).T

        self.assertTrue(
            (
                np.sum(deriv_gpx - deriv_krg)
                + np.sum(deriv_gpx - diff_gpx)
                + np.sum(deriv_krg - diff_krg)
            )
            / np.sum(np.abs(deriv_gpx))
            < 1e-3
        )


if __name__ == "__main__":
    unittest.main()
