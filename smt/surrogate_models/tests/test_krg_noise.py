"""
Author: Andres Lopez-Lopera <<andres.lopez_lopera@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest

import numpy as np

from smt.surrogate_models import KRG
from smt.utils.sm_test_case import SMTestCase


class Test(SMTestCase):
    def test_predict_output(self):
        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 1.1, 1.0])

        # Adding noisy repetitions
        np.random.seed(6)
        yt_std_rand = np.std(yt) * np.random.uniform(size=yt.shape)
        xt_full = np.array(3 * xt.tolist())
        yt_full = np.concatenate((yt, yt + 0.2 * yt_std_rand, yt - 0.2 * yt_std_rand))

        sm = KRG(
            theta0=[1.0],
            eval_noise=True,
            use_het_noise=True,
            n_start=1,
            hyper_opt="Cobyla",
        )
        sm.set_training_values(xt_full, yt_full)
        sm.train()

        yt = yt.reshape(-1, 1)
        y = sm.predict_values(xt)
        t_error = np.linalg.norm(y - yt) / np.linalg.norm(yt)
        self.assert_error(t_error, 0.0, 1e-2)

    def test_predict_variance(self):
        # defining the training data
        xt = np.array([0.0, 1.0, 2.0, 2.5, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 1.1, 1.0])

        # defining the models
        sm_noise_free = KRG()  # noise-free Kriging model
        sm_noise_fixed = KRG(
            noise0=[1e-1], print_global=False
        )  # noisy Kriging model with fixed variance
        sm_noise_estim = KRG(
            noise0=[1e-1],
            eval_noise=True,
            noise_bounds=[1e-2, 1000.0],
            print_global=False,
        )  # noisy Kriging model with estimated variance

        # training the models
        sm_noise_free.set_training_values(xt, yt)
        sm_noise_free.train()

        sm_noise_fixed.set_training_values(xt, yt)
        sm_noise_fixed.train()

        sm_noise_estim.set_training_values(xt, yt)
        sm_noise_estim.train()

        # predictions at training points
        x = xt

        # error message in case if test case got failed

        # the Variance (interpolation case without noise) must be =/ 0
        var_noise_free = sm_noise_free.predict_variances(x)  # predictive variance
        self.assert_error(np.linalg.norm(var_noise_free), 0.0, 1e-5)

        # the Variance (regression case with noise) must be =/ 0
        var_noise_fixed = sm_noise_fixed.predict_variances(x)  # predictive variance
        self.assert_error(np.linalg.norm(var_noise_fixed), 0.04768, 1e-5)
        var_noise_estim = sm_noise_estim.predict_variances(x)  # predictive variance
        self.assert_error(np.linalg.norm(var_noise_estim), 0.01135, 1e-3)

    def test_predict_variance_ri(self):
        # defining the training data
        xt = np.array([0.0, 1.0, 2.0, 2.5, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 1.1, 1.0])

        # defining the models
        sm_noisy = KRG(noise0=[1e-1], print_global=False)
        # training the models
        sm_noisy.set_training_values(xt, yt)
        sm_noisy.train()

        var_estim_free = sm_noisy.predict_variances(xt, is_ri=False)
        var_estim_noisy = sm_noisy.predict_variances(xt, is_ri=True)

        # the variances with re-interpolation should be lower than without
        for var_free, var_noisy in zip(var_estim_free, var_estim_noisy):
            self.assertTrue(
                var_noisy < var_free,
                f"Expected var_noisy < var_free but got {var_noisy} >= {var_free}",
            )


if __name__ == "__main__":
    unittest.main()
