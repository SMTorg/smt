import numpy as np
import unittest

from smt.surrogate_models import SGP
from smt.utils.sm_test_case import SMTestCase


def f_obj(x):
    return (
        np.sin(3 * np.pi * x)
        + 0.3 * np.cos(9 * np.pi * x)
        + 0.5 * np.sin(7 * np.pi * x)
    )


class TestSGP(SMTestCase):
    def setUp(self):
        rng = np.random.RandomState(1)

        # Generate training data
        N_train = 200
        self.eta = [0.01]
        gaussian_noise = rng.normal(loc=0.0, scale=np.sqrt(self.eta), size=(N_train, 1))
        self.Xtrain = 2 * rng.rand(N_train, 1) - 1
        self.Ytrain = f_obj(self.Xtrain) + gaussian_noise

        # Generate test data (noise-free)
        N_test = 50
        self.Xtest = 2 * rng.rand(N_test, 1) - 1
        self.Ytest = f_obj(self.Xtest).reshape(-1, 1)

        # Pick inducing points at random
        N_inducing = 30
        self.Z = 2 * rng.rand(N_inducing, 1) - 1

    def test_fitc_with_noise0(self):
        # Assume we know the variance eta of our noisy input data
        sgp = SGP(noise0=self.eta)
        sgp.set_training_values(self.Xtrain, self.Ytrain)
        sgp.set_inducing_inputs(Z=self.Z)
        sgp.train()

        Ypred = sgp.predict_values(self.Xtest)
        self.assert_error(Ypred, self.Ytest, atol=0.02, rtol=0.1)

    def test_vfe_with_noise0(self):
        sgp = SGP(noise0=self.eta, method="VFE")
        sgp.set_training_values(self.Xtrain, self.Ytrain)
        sgp.set_inducing_inputs(Z=self.Z)
        sgp.train()

        Ypred = sgp.predict_values(self.Xtest)
        self.assert_error(Ypred, self.Ytest, atol=0.05, rtol=0.1)

    def test_fitc_with_noise_eval(self):
        # Assume we know the variance eta of our noisy input data
        sgp = SGP()
        sgp.set_training_values(self.Xtrain, self.Ytrain)
        sgp.set_inducing_inputs(Z=self.Z)
        sgp.train()

        Ypred = sgp.predict_values(self.Xtest)
        self.assert_error(Ypred, self.Ytest, atol=0.02, rtol=0.1)
        self.assertAlmostEqual(sgp.optimal_noise, self.eta[0], delta=5e-3)

    def test_vfe_with_noise_eval(self):
        # Assume we know the variance eta of our noisy input data
        sgp = SGP(method="VFE")
        sgp.set_training_values(self.Xtrain, self.Ytrain)
        sgp.set_inducing_inputs(Z=self.Z)
        sgp.train()

        Ypred = sgp.predict_values(self.Xtest)
        self.assert_error(Ypred, self.Ytest, atol=0.05, rtol=0.1)
        self.assertAlmostEqual(sgp.optimal_noise, self.eta[0], delta=1.6e-2)


if __name__ == "__main__":
    unittest.main()
