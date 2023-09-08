import numpy as np
import matplotlib.pyplot as plt
import unittest

from smt.surrogate_models import SGP


def f_obj(x):
    return (
        np.sin(3 * np.pi * x)
        + 0.3 * np.cos(9 * np.pi * x)
        + 0.5 * np.sin(7 * np.pi * x)
    )


class TestSGP(unittest.TestCase):
    def test_1d(self):
        dir = "D:\\rlafage/workspace/sparse_GPy_SMT/"
        rng = np.random.RandomState(0)

        # Generate training data
        N_train = 200
        eta = [0.01]
        gaussian_noise = rng.normal(loc=0.0, scale=np.sqrt(eta), size=(N_train, 1))
        Xtrain = 2 * np.random.rand(N_train, 1) - 1
        # Xtrain = np.load(dir + "xtrain.npy")
        Ytrain = f_obj(Xtrain) + gaussian_noise
        # Ytrain = np.load(dir + "ytrain.npy")
        # Generate test data (noise-free)
        N_test = 50
        Xtest = 2 * np.random.rand(N_test, 1) - 1
        # Xtest = np.load(dir + "xtest.npy")
        Ytest = f_obj(Xtest).reshape(-1, 1)

        sgp = SGP(noise0=eta, theta_bounds=[1e-6, 100], n_inducing=30)
        sgp.set_training_values(Xtrain, Ytrain)
        # sgp.set_inducing_inputs(Z=np.load(dir + "inducing.npy"))
        sgp.train()

        Ypred = sgp.predict_values(Xtest)
        rmse = np.sqrt(np.mean((Ypred - Ytest) ** 2))
        print("\nRMSE : %f" % rmse)

        # plot prediction
        x = np.linspace(-1, 1, 201).reshape(-1, 1)
        y = f_obj(x)

        hat_y = sgp.predict_values(x)
        var = sgp.predict_variances(x)
        Z = sgp.Z
        plt.figure(figsize=(14, 6))

        plt.plot(x, y, "C1-", label="target function")
        plt.scatter(Xtrain, Ytrain, marker="o", s=10, label="observed data")
        plt.plot(x, hat_y, "k-", label="sparse GP")
        plt.plot(x, hat_y - 3 * np.sqrt(var), "k--")
        plt.plot(x, hat_y + 3 * np.sqrt(var), "k--", label="99% CI")
        plt.plot(Z, -2.9 * np.ones_like(Z), "r|", mew=2, label="inducing points")
        # plt.xlim([-1.1,1.1])
        plt.ylim([-3, 3])
        plt.legend(loc=0)
        plt.show()


if __name__ == "__main__":
    unittest.main()
