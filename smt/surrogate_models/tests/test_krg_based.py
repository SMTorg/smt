"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest

import numpy as np

from smt.surrogate_models import KRG
from smt.surrogate_models.krg_based import KrgBased
from smt.utils.misc import compute_rms_error
from smt.problems import Rosenbrock
from smt.sampling_methods import LHS


# defining the toy example
def target_fun(x):
    return np.cos(5 * x)


class TestKrgBased(unittest.TestCase):
    def test_theta0_default_init(self):
        krg = KrgBased()
        krg.set_training_values(np.array([[1, 2, 3]]), np.array([[1]]))
        krg._check_param()
        self.assertTrue(np.array_equal(krg.options["theta0"], [1e-2, 1e-2, 1e-2]))

    def test_theta0_one_dim_init(self):
        krg = KrgBased(theta0=[2e-2])
        krg.set_training_values(np.array([[1, 2, 3]]), np.array([[1]]))
        krg._check_param()
        self.assertTrue(np.array_equal(krg.options["theta0"], [2e-2, 2e-2, 2e-2]))

    def test_theta0_erroneous_init(self):
        krg = KrgBased(theta0=[2e-2, 1e-2])
        krg.set_training_values(np.array([[1, 2]]), np.array([[1]]))  # correct
        krg._check_param()
        krg.set_training_values(np.array([[1, 2, 3]]), np.array([[1]]))  # erroneous
        self.assertRaises(ValueError, krg._check_param)

    def test_less_almost_squar_exp(self):
        nobs = 50  # number of obsertvations
        np.random.seed(0)  # a seed for reproducibility
        xt = np.random.uniform(size=nobs)  # design points

        # adding a random noise to observations
        yt = target_fun(xt) + np.random.normal(scale=0.05, size=nobs)

        # training the model with the option eval_noise= True
        sm = KRG(eval_noise=False, corr="pow_exp", pow_exp_power=1.99)
        sm.set_training_values(xt, yt)
        sm.train()

        # predictions
        x = np.linspace(0, 1, 500).reshape(-1, 1)
        sm.predict_values(x)  # predictive mean
        sm.predict_variances(x)  # predictive variance
        sm.predict_derivatives(x, 0)  # predictive variance
        self.assertLess(
            np.abs(
                sm.predict_derivatives(x[20], 0)
                - (sm.predict_values(x[20] + 1e-6) - sm.predict_values(x[20])) / 1e-6
            ),
            1.01e-2,
        )

    def test_check_training_numerically(self):
        ndim = 2
        ndoe = 20  # int(10*ndim)
        # Define the function
        fun = Rosenbrock(ndim=ndim)

        # Construction of the DOE
        # in order to have the always same LHS points, random_state=1
        sampling = LHS(xlimits=fun.xlimits, criterion="ese", random_state=1)
        xt = sampling(ndoe)
        # Compute the outputs
        yt = fun(xt)

        # Construction of the validation points
        ntest = 200  # 500
        sampling = LHS(xlimits=fun.xlimits, criterion="ese", random_state=1)
        xtest = sampling(ntest)
        ytest = fun(xtest)

        # The variable 'theta0' is a list of length ndim.
        t = KRG(theta0=[1e-2] * ndim, print_prediction=False, corr="pow_exp")
        t.set_training_values(xt, yt[:, 0])

        t.train()

        # Prediction of the validation points
        t.predict_values(xtest)
        print("Kriging,  err: " + str(compute_rms_error(t, xtest, ytest)))
        t.check_training_numerically()
        # The variable 'theta0' is a list of length ndim.
        t = KRG(theta0=[1e-2] * ndim, print_prediction=False, corr="squar_exp")
        t.set_training_values(xt, yt[:, 0])

        t.train()

        # Prediction of the validation points
        t.predict_values(xtest)
        print("Kriging,  err: " + str(compute_rms_error(t, xtest, ytest)))
        t.check_training_numerically()


if __name__ == "__main__":
    unittest.main()
