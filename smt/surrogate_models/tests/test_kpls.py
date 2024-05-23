"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest

import numpy as np

from smt.sampling_methods import LHS
from smt.surrogate_models import KPLS
from smt.surrogate_models import KRG, KPLSK
from smt.utils.misc import compute_rms_error
import time


class TestKPLS(unittest.TestCase):
    def test_predict_output(self):
        for corr_str in [
            "pow_exp",
            "abs_exp",
            "squar_exp",
        ]:
            d, n = (3, 3)
            sx = LHS(
                xlimits=np.repeat(np.atleast_2d([0.0, 1.0]), d, axis=0),
                criterion="m",
                random_state=42,
            )
            x = sx(n)
            sy = LHS(
                xlimits=np.repeat(np.atleast_2d([0.0, 1.0]), 1, axis=0),
                criterion="m",
                random_state=42,
            )
            y = sy(n)

            kriging = KPLS(n_comp=2, corr=corr_str, print_global=False)
            kriging.set_training_values(x, y)
            kriging.train()

            x_fail_1 = np.asarray([[0, 0, 0, 0]])
            x_fail_2 = np.asarray([[0]])

            self.assertRaises(ValueError, lambda: kriging.predict_values(x_fail_1))
            self.assertRaises(ValueError, lambda: kriging.predict_values(x_fail_2))
            var = kriging.predict_variances(x)
            self.assertEqual(y.shape[0], var.shape[0])
            kriging = KPLS(n_comp=3, print_global=False)
            kriging.set_training_values(x, y)
            self.assertRaises(ValueError, lambda: kriging.train())

    def test_optim_TNC_Cobyla(self):
        for corr_str in [
            "pow_exp",
            "abs_exp",
            "squar_exp",
        ]:
            d, n = (12, 12)
            sx = LHS(
                xlimits=np.repeat(np.atleast_2d([0.0, 1.0]), d, axis=0),
                criterion="m",
                random_state=42,
            )
            x = sx(n)
            sy = LHS(
                xlimits=np.repeat(np.atleast_2d([0.0, 1.0]), 1, axis=0),
                criterion="m",
                random_state=42,
            )
            y = sy(n)

            kriging_Cobyla = KPLS(
                n_comp=5,
                corr=corr_str,
                print_global=False,
                n_start=50,
                hyper_opt="Cobyla",
            )
            kriging_TNC = KPLS(
                n_comp=5, corr=corr_str, print_global=False, n_start=50, hyper_opt="TNC"
            )
            kriging_TNC.set_training_values(x, y)
            kriging_Cobyla.set_training_values(x, y)
            kriging_TNC.train()
            kriging_Cobyla.train()
            # assert TNC better than Cobyla
            self.assertTrue(
                np.linalg.norm(
                    kriging_TNC.optimal_rlf_value - kriging_Cobyla.optimal_rlf_value
                )
                > 0
            )
            # assert small error between Cobyla and TNC
            print(
                np.linalg.norm(kriging_Cobyla.optimal_theta - kriging_TNC.optimal_theta)
            )
            self.assertTrue(
                np.linalg.norm(kriging_Cobyla.optimal_theta - kriging_TNC.optimal_theta)
                < 2
            )

    def test_optim_kplsk(self):
        # Griewank function definition
        def griewank(x):
            x = np.asarray(x)
            if x.ndim == 1 or max(x.shape) == 1:
                x = x.reshape((1, -1))
            # dim = x.shape[1]

            s, p = 0.0, 1.0
            for i, xi in enumerate(x.T):
                s += xi**2 / 4000.0
                p *= np.cos(xi / np.sqrt(i + 1))
            return s - p + 1.0

        lb = -5
        ub = 5
        n_dim = 100

        # LHS training point generation
        n_train = 30
        sx = LHS(
            xlimits=np.repeat(np.atleast_2d([0.0, 1.0]), n_dim, axis=0),
            criterion="m",
            random_state=42,
        )
        x_train = sx(n_train)
        x_train = lb + (ub - lb) * x_train  # map generated samples to design space
        y_train = griewank(x_train)
        y_train = y_train.reshape((n_train, -1))  # reshape to 2D array

        # Random test point generation
        n_test = 3000
        x_test = np.random.random_sample((n_test, n_dim))
        x_test = lb + (ub - lb) * x_test  # map generated samples to design space
        y_test = griewank(x_test)
        y_test = y_test.reshape((n_test, -1))  # reshape to 2D array

        # Surrogate model definition
        n_pls = 2
        models = [
            KRG(n_start=20, hyper_opt="Cobyla"),
            KPLSK(n_comp=n_pls, n_start=20, hyper_opt="Cobyla"),
            KPLS(n_comp=n_pls, n_start=20, hyper_opt="Cobyla"),
        ]
        rms = []
        times = []
        # Surrogate model fit & error estimation
        for model in models:
            model.set_training_values(x_train, y_train)
            intime = time.time()
            model.train()
            times.append(time.time() - intime)

            # y_pred = model.predict_values(x_test)
            error = compute_rms_error(model, x_test, y_test)
            rms.append(error)
        self.assertTrue((rms[0] <= rms[1]) and (rms[1] <= rms[2]))
        self.assertTrue((times[0] >= times[1]) and (times[1] >= times[2]))


if __name__ == "__main__":
    unittest.main()
