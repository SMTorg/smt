"""
Authors: Nathalie Bartoli, Paul Saves

This package is distributed under New BSD license.
"""

import unittest

import numpy as np

from smt.sampling_methods import LHS
from smt.surrogate_models import KRG
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

        def pb_for_sin_squar_exp(x):
            # sin + linear trend
            y = np.sin(x[:, 1]) + 2 * x[:, 1] + 5 * x[:, 0] + 10  # + linear trend
            return y

        xlimits = np.array([[-5, 10], [-5, 10]])
        self.sampling = LHS(xlimits=xlimits, random_state=42)
        self.xt = self.sampling(12)
        self.yt = pb(self.xt)
        self.yt_squar_sin_exp = pb_for_sin_squar_exp(self.xt)

    def test_predictions(self):
        trends = ["constant", "linear"]
        kernels = [
            "pow_exp",
            "squar_exp",
            "abs_exp",
            "matern32",
            "matern52",
            "squar_sin_exp",
        ]
        powers = [1.0, 1.5, 2.0]

        for trend in trends:
            for kernel in kernels:
                if kernel == "squar_sin_exp":
                    yt = self.yt_squar_sin_exp
                else:
                    yt = self.yt
                if kernel == "pow_exp":
                    for power in powers:
                        sm = KRG(
                            theta0=[0.01],
                            print_global=False,
                            poly=trend,
                            corr=kernel,
                            pow_exp_power=power,
                        )  # ,eval_noise=True)
                        sm.set_training_values(self.xt, yt)
                        sm.train()

                        print(f"\n*** TREND = {trend} & kernel = {kernel} ***\n")

                        # quality of the surrogate on validation points
                        Test._check_prediction_variances(self, sm)
                        Test._check_prediction_derivatives(self, sm)

                else:
                    sm = KRG(
                        theta0=[0.01], print_global=False, poly=trend, corr=kernel
                    )  # ,eval_noise=True)
                    sm.set_training_values(self.xt, yt)
                    sm.train()

                    print(f"\n*** TREND = {trend} & kernel = {kernel} ***\n")

                    # quality of the surrogate on validation points
                    Test._check_prediction_variances(self, sm)
                    Test._check_prediction_derivatives(self, sm)

    def test_variance_derivatives_vs_gradient(self):
        # checks that concatenation of partial derivatives wrt kx-th component
        # is equivalent to the gradients at a given point x
        sm = KRG(theta0=[0.01], print_global=False)
        sm.set_training_values(self.xt, self.yt)
        sm.train()
        x = np.array([[1, 2]])
        deriv0 = sm.predict_variance_derivatives(x, 0)
        deriv1 = sm.predict_variance_derivatives(x, 1)
        derivs = np.hstack((deriv0, deriv1))
        gradient = sm.predict_variance_gradient(x[0])
        self.assertEqual((1, 2), gradient.shape)
        gradient = sm.predict_variance_gradient(x)
        self.assertEqual((1, 2), gradient.shape)
        np.testing.assert_allclose(gradient, derivs)

    @staticmethod
    def _check_prediction_variances(self, sm):
        y_predicted = sm.predict_variances(self.xt)
        variance_at_training_inputs = np.sum(y_predicted**2)

        np.testing.assert_allclose(variance_at_training_inputs, 0, atol=1e-9)

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

        deriv = np.array(
            [
                sm.predict_variance_derivatives(x, 0)[0],
                sm.predict_variance_derivatives(x, 1)[0],
            ]
        ).T
        np.testing.assert_allclose(
            deriv, np.array([[diff_g, diff_d]]), atol=1e-2, rtol=1e-2
        )

        y_predicted = sm.predict_values(x_valid)

        x = np.atleast_2d(x_valid[0])
        diff_g = (y_predicted[1, 0] - y_predicted[2, 0]) / (2 * e)
        diff_d = (y_predicted[3, 0] - y_predicted[4, 0]) / (2 * e)

        deriv = np.array(
            [sm.predict_derivatives(x, 0)[0], sm.predict_derivatives(x, 1)[0]]
        ).T
        np.testing.assert_allclose(
            deriv, np.array([[diff_g, diff_d]]), atol=1e-2, rtol=1e-2
        )

        ### VECTORIZATION TESTS

        x_valid = np.concatenate(
            (
                x_valid,
                np.atleast_2d(np.array([x_valid[0][0] + 1.0, x_valid[0][1] + 1.0])),
            )
        )

        # test predict values & variances vectorization
        all_vals1 = np.zeros((6, 2))
        for i, x in enumerate(x_valid):
            all_vals1[i, 0] = sm.predict_values(np.atleast_2d(x)).item()
            all_vals1[i, 1] = sm.predict_variances(np.atleast_2d(x)).item()
        all_vals2x = sm.predict_values(np.atleast_2d(x_valid)).flatten()
        all_vals2y = sm.predict_variances(np.atleast_2d(x_valid)).flatten()
        total_error = np.sum(
            [
                np.power(all_vals1[:, 0] - all_vals2x, 2),
                np.power(all_vals1[:, 1] - all_vals2y, 2),
            ]
        )
        np.testing.assert_allclose(total_error, 0, atol=1e-9)

        # test predict_derivatives vectorization
        all_vals1 = np.zeros((6, 2))
        for i, x in enumerate(x_valid):
            all_vals1[i, 0] = sm.predict_derivatives(np.atleast_2d(x), 0).item()
            all_vals1[i, 1] = sm.predict_derivatives(np.atleast_2d(x), 1).item()
        all_vals2x = sm.predict_derivatives(np.atleast_2d(x_valid), 0).flatten()
        all_vals2y = sm.predict_derivatives(np.atleast_2d(x_valid), 1).flatten()
        total_error = np.sum(
            [
                np.power(all_vals1[:, 0] - all_vals2x, 2),
                np.power(all_vals1[:, 1] - all_vals2y, 2),
            ]
        )
        np.testing.assert_allclose(total_error, 0, atol=1e-9)

        # test predict_variance_derivatives vectorization
        all_vals1 = np.zeros((6, 2))
        for i, x in enumerate(x_valid):
            all_vals1[i, 0] = sm.predict_variance_derivatives(
                np.atleast_2d(x), 0
            ).item()
            all_vals1[i, 1] = sm.predict_variance_derivatives(
                np.atleast_2d(x), 1
            ).item()
        all_vals2x = sm.predict_variance_derivatives(
            np.atleast_2d(x_valid), 0
        ).flatten()
        all_vals2y = sm.predict_variance_derivatives(
            np.atleast_2d(x_valid), 1
        ).flatten()
        total_error = np.sum(
            [
                np.power(all_vals1[:, 0] - all_vals2x, 2),
                np.power(all_vals1[:, 1] - all_vals2y, 2),
            ]
        )
        np.testing.assert_allclose(total_error, 0, atol=1e-9)

    def test_fixed_theta(self):
        theta0 = np.ones(2)

        sm = KRG(theta0=theta0)
        sm.set_training_values(self.xt, self.yt)
        sm.train()
        print(sm.optimal_theta)  # [0.04373813 0.00697964]
        sm = KRG(theta0=theta0, hyper_opt="NoOp")
        sm.set_training_values(self.xt, self.yt)
        sm.train()
        np.testing.assert_allclose(sm.optimal_theta, theta0, 1e-2)


if __name__ == "__main__":
    unittest.main()
