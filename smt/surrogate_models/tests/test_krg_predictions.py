"""
Authors: Nathalie Bartoli, Paul Saves 

This package is distributed under New BSD license.
"""

import unittest
import numpy as np
from smt.surrogate_models import KRG
from smt.sampling_methods import LHS
from smt.utils.sm_test_case import SMTestCase


class Test(SMTestCase):
    def setUp(self):
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
        self.xt = sampling(12)
        self.yt = pb(self.xt)

    def test_predictions(self):
        trends = ["constant", "linear"]
        kernels = ["squar_exp", "abs_exp", "matern32", "matern52"]

        for trend in trends:
            for kernel in kernels:
                sm = KRG(
                    theta0=[0.01], print_global=False, poly=trend, corr=kernel
                )  # ,eval_noise=True)
                sm.set_training_values(self.xt, self.yt)
                sm.train()

                print(f"\n*** TREND = {trend} & kernel = {kernel} ***\n")

                # quality of the surrogate on validation points
                Test._check_prediction_variances(self, sm)
                Test._check_prediction_derivatives(self, sm)

    @staticmethod
    def _check_prediction_variances(self, sm):

        y_predicted = sm.predict_variances(self.xt)
        variance_at_training_inputs = np.sum(y_predicted**2)

        np.testing.assert_allclose(variance_at_training_inputs, 0, atol=1e-5)

    @staticmethod
    def _check_prediction_derivatives(self, sm):
        e = 5e-6
        xa = -1.3
        xb = 2.5
        x_valid = np.array(
            [[xa, xb], [xa + e, xb], [xa - e, xb], [xa, xb + e], [xa, xb - e]]
        )

        y_predicted = sm.predict_variances(x_valid)

        x = np.atleast_2d(x_valid[0])
        diff_g = (y_predicted[1, 0] - y_predicted[2, 0]) / (2 * e)
        diff_d = (y_predicted[3, 0] - y_predicted[4, 0]) / (2 * e)

        deriv = sm.predict_variance_derivatives(x)
        pred_errors = np.array(
            [
                np.abs((diff_g - deriv[0][0]) / diff_g),
                np.abs((diff_d - deriv[0][1]) / diff_d),
            ]
        )
        total_error = np.sum(pred_errors**2)
        np.testing.assert_allclose(total_error, 0, atol=5e-2)

        y_predicted = sm.predict_values(x_valid)

        x = np.atleast_2d(x_valid[0])
        diff_g = (y_predicted[1, 0] - y_predicted[2, 0]) / (2 * e)
        diff_d = (y_predicted[3, 0] - y_predicted[4, 0]) / (2 * e)

        deriv = np.array(
            [sm.predict_derivatives(x, 0)[0], sm.predict_derivatives(x, 1)[0]]
        ).T
        pred_errors = np.array(
            [
                np.abs((diff_g - deriv[0][0]) / diff_g),
                np.abs((diff_d - deriv[0][1]) / diff_d),
            ]
        )
        total_error = np.sum(pred_errors**2)

        np.testing.assert_allclose(total_error, 0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
