import unittest

import numpy as np

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
        rng = np.random.default_rng(42)

        # Generate training data
        N_train = 200
        self.eta = [0.01]
        gaussian_noise = rng.normal(loc=0.0, scale=np.sqrt(self.eta), size=(N_train, 1))
        self.Xtrain = 2 * rng.random((N_train, 1)) - 1
        self.Ytrain = f_obj(self.Xtrain) + gaussian_noise

        # Generate test data (noise-free)
        N_test = 50
        self.Xtest = 2 * rng.random((N_test, 1)) - 1
        self.Ytest = f_obj(self.Xtest).reshape(-1, 1)

        # Pick inducing points at random
        N_inducing = 30
        self.Z = 2 * rng.random((N_inducing, 1)) - 1

    def test_fitc_with_noise0(self):
        # Assume we know the variance eta of our noisy input data
        sgp = SGP(noise0=self.eta)
        sgp.set_training_values(self.Xtrain, self.Ytrain)
        sgp.set_inducing_inputs(Z=self.Z)
        sgp.train()

        Ypred = sgp.predict_values(self.Xtest)
        self.assert_error(Ypred, self.Ytest, atol=0.05, rtol=0.2)

    def test_vfe_with_noise0(self):
        sgp = SGP(noise0=self.eta, method="VFE")
        sgp.set_training_values(self.Xtrain, self.Ytrain)
        sgp.set_inducing_inputs(Z=self.Z)
        sgp.train()

        Ypred = sgp.predict_values(self.Xtest)
        self.assert_error(Ypred, self.Ytest, atol=0.05, rtol=0.2)

    def test_fitc_with_noise_eval(self):
        # Assume we know the variance eta of our noisy input data
        sgp = SGP()
        sgp.set_training_values(self.Xtrain, self.Ytrain)
        sgp.set_inducing_inputs(Z=self.Z)
        sgp.train()

        Ypred = sgp.predict_values(self.Xtest)
        self.assert_error(Ypred, self.Ytest, atol=0.05, rtol=0.2)
        self.assertAlmostEqual(sgp.optimal_noise, self.eta[0], delta=5e-3)

    def test_vfe_with_noise_eval(self):
        # Assume we know the variance eta of our noisy input data
        sgp = SGP(method="VFE")
        sgp.set_training_values(self.Xtrain, self.Ytrain)
        sgp.set_inducing_inputs(Z=self.Z)
        sgp.train()

        Ypred = sgp.predict_values(self.Xtest)
        self.assert_error(Ypred, self.Ytest, atol=0.05, rtol=0.2)
        self.assertAlmostEqual(sgp.optimal_noise, self.eta[0], delta=2.9e-2)

    # --- 1D derivative tests (f_obj) ---

    def test_predict_derivatives_fitc(self):
        sgp = SGP(noise0=self.eta)
        sgp.set_training_values(self.Xtrain, self.Ytrain)
        sgp.set_inducing_inputs(Z=self.Z)
        sgp.train()

        h = 1e-6
        x = self.Xtest
        analytical = sgp.predict_derivatives(x, 0)

        x_fwd = x.copy()
        x_fwd[:, 0] += h
        x_bwd = x.copy()
        x_bwd[:, 0] -= h
        fd = (sgp.predict_values(x_fwd) - sgp.predict_values(x_bwd)) / (2 * h)

        np.testing.assert_allclose(analytical, fd, rtol=1e-4, atol=1e-8)

    def test_predict_derivatives_vfe(self):
        sgp = SGP(noise0=self.eta, method="VFE")
        sgp.set_training_values(self.Xtrain, self.Ytrain)
        sgp.set_inducing_inputs(Z=self.Z)
        sgp.train()

        h = 1e-6
        x = self.Xtest
        analytical = sgp.predict_derivatives(x, 0)

        x_fwd = x.copy()
        x_fwd[:, 0] += h
        x_bwd = x.copy()
        x_bwd[:, 0] -= h
        fd = (sgp.predict_values(x_fwd) - sgp.predict_values(x_bwd)) / (2 * h)

        np.testing.assert_allclose(analytical, fd, rtol=1e-4, atol=1e-8)

    def test_predict_variance_derivatives_fitc(self):
        sgp = SGP(noise0=self.eta)
        sgp.set_training_values(self.Xtrain, self.Ytrain)
        sgp.set_inducing_inputs(Z=self.Z)
        sgp.train()

        h = 1e-6
        x = self.Xtest
        analytical = sgp.predict_variance_derivatives(x, 0)

        # Use kernel-level FD: compute FD of K(x, Z) then apply the formula
        # -2 * dK^T @ W_inv @ K.  This avoids catastrophic cancellation in
        # the FD of sigma2 - k^T W_inv k when the Woodbury inverse is
        # ill-conditioned and variance is small.
        x_fwd = x.copy()
        x_fwd[:, 0] += h
        x_bwd = x.copy()
        x_bwd[:, 0] -= h
        theta, sigma2 = sgp.optimal_theta, sgp.optimal_sigma2
        Kx = sgp._compute_K(x, sgp.Z, theta, sigma2)
        dKx_fd = (
            sgp._compute_K(x_fwd, sgp.Z, theta, sigma2)
            - sgp._compute_K(x_bwd, sgp.Z, theta, sigma2)
        ) / (2 * h)
        fd = -2 * np.sum(
            (dKx_fd @ sgp.woodbury_data["inv"]) * Kx, axis=1, keepdims=True
        )

        np.testing.assert_allclose(analytical, fd, rtol=1e-3, atol=5e-4)

    def test_predict_variance_derivatives_vfe(self):
        sgp = SGP(noise0=self.eta, method="VFE")
        sgp.set_training_values(self.Xtrain, self.Ytrain)
        sgp.set_inducing_inputs(Z=self.Z)
        sgp.train()

        h = 1e-6
        x = self.Xtest
        analytical = sgp.predict_variance_derivatives(x, 0)

        # Use kernel-level FD (see test_predict_variance_derivatives_fitc)
        x_fwd = x.copy()
        x_fwd[:, 0] += h
        x_bwd = x.copy()
        x_bwd[:, 0] -= h
        theta, sigma2 = sgp.optimal_theta, sgp.optimal_sigma2
        Kx = sgp._compute_K(x, sgp.Z, theta, sigma2)
        dKx_fd = (
            sgp._compute_K(x_fwd, sgp.Z, theta, sigma2)
            - sgp._compute_K(x_bwd, sgp.Z, theta, sigma2)
        ) / (2 * h)
        fd = -2 * np.sum(
            (dKx_fd @ sgp.woodbury_data["inv"]) * Kx, axis=1, keepdims=True
        )

        np.testing.assert_allclose(analytical, fd, rtol=1e-3, atol=5e-4)

    # --- 3D derivative tests (sin volume) ---

    def test_predict_derivatives_3d(self):
        rng = np.random.default_rng(42)
        N_train = 200
        Xtrain = rng.uniform([-0.5, -1.0, -2.0], [0.5, 1.0, 2.0], size=(N_train, 3))
        Ytrain = (
            np.sin(2 * np.pi * Xtrain[:, 0])
            * np.sin(np.pi * Xtrain[:, 1])
            * np.sin(np.pi * Xtrain[:, 2] / 2)
        )

        sgp = SGP(
            noise0=[1e-4],
            n_inducing=30,
            inducing_method="kmeans",
            seed=42,
            print_global=False,
        )
        sgp.set_training_values(Xtrain, Ytrain)
        sgp.train()

        N_test = 50
        Xtest = rng.uniform([-0.5, -1.0, -2.0], [0.5, 1.0, 2.0], size=(N_test, 3))

        h = 1e-6
        for kx in range(3):
            analytical = sgp.predict_derivatives(Xtest, kx)
            x_fwd = Xtest.copy()
            x_fwd[:, kx] += h
            x_bwd = Xtest.copy()
            x_bwd[:, kx] -= h
            fd = (sgp.predict_values(x_fwd) - sgp.predict_values(x_bwd)) / (2 * h)
            np.testing.assert_allclose(
                analytical,
                fd,
                rtol=1e-4,
                atol=1e-8,
                err_msg=f"predict_derivatives failed for kx={kx}",
            )

    def test_predict_variance_derivatives_3d(self):
        rng = np.random.default_rng(42)
        N_train = 200
        Xtrain = rng.uniform([-0.5, -1.0, -2.0], [0.5, 1.0, 2.0], size=(N_train, 3))
        Ytrain = (
            np.sin(2 * np.pi * Xtrain[:, 0])
            * np.sin(np.pi * Xtrain[:, 1])
            * np.sin(np.pi * Xtrain[:, 2] / 2)
        )

        sgp = SGP(
            noise0=[1e-4],
            n_inducing=30,
            inducing_method="kmeans",
            seed=42,
            print_global=False,
        )
        sgp.set_training_values(Xtrain, Ytrain)
        sgp.train()

        N_test = 50
        Xtest = rng.uniform([-0.5, -1.0, -2.0], [0.5, 1.0, 2.0], size=(N_test, 3))

        h = 1e-6
        for kx in range(3):
            analytical = sgp.predict_variance_derivatives(Xtest, kx)
            x_fwd = Xtest.copy()
            x_fwd[:, kx] += h
            x_bwd = Xtest.copy()
            x_bwd[:, kx] -= h
            fd = (sgp.predict_variances(x_fwd) - sgp.predict_variances(x_bwd)) / (2 * h)
            np.testing.assert_allclose(
                analytical,
                fd,
                rtol=1e-3,
                atol=1e-8,
                err_msg=f"predict_variance_derivatives failed for kx={kx}",
            )

    # --- Other tests ---

    def test_fitc_with_kmeans(self):
        sgp = SGP(n_inducing=30, inducing_method="kmeans")
        sgp.set_training_values(self.Xtrain, self.Ytrain)
        sgp.train()

        Ypred = sgp.predict_values(self.Xtest)
        self.assert_error(Ypred, self.Ytest, atol=0.05, rtol=0.2)

    def test_vfe_with_random(self):
        sgp = SGP(method="VFE", n_inducing=30, inducing_method="random")
        sgp.set_training_values(self.Xtrain, self.Ytrain)
        sgp.train()

        Ypred = sgp.predict_values(self.Xtest)
        self.assert_error(Ypred, self.Ytest, atol=0.05, rtol=0.3)

    def test_inducing_error(self):
        sgp = SGP()
        sgp.set_training_values(self.Xtrain, self.Ytrain)
        with self.assertRaises(
            ValueError,
            msg="Specify inducing points with set_inducing_inputs() or set inducing_method option",
        ):
            sgp.train()

    def test_sgp_reproducibility(self):
        samples = 32
        xs, ys = generate_sin_volume(samples)
        sgp1 = SGP(
            print_global=False,
            inducing_method="kmeans",
            n_inducing=samples,
            seed=42,
        )
        sgp1.set_training_values(xs, ys)
        sgp1.train()
        sgp2 = SGP(
            print_global=False,
            inducing_method="kmeans",
            n_inducing=samples,
            seed=42,
        )
        sgp2.set_training_values(xs, ys)
        sgp2.train()

        x = np.array([np.array([0.3, -0.4, 0.6])])
        y1 = sgp1.predict_values(x)[0][0]
        y2 = sgp2.predict_values(x)[0][0]
        print(f"{y1=}, {y2=}")
        assert y1 == y2, "SGP is not bitwise reproducible."


def generate_sin_volume(samples: int):
    abscissas = np.ndarray(shape=(samples, 3))
    ordinates = np.ndarray(shape=samples)
    for i in range(samples):
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-2, 2)
        abscissas[i] = np.array([x, y, z])
        ordinates[i] = np.sin(2 * np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z / 2)
    return abscissas, ordinates


if __name__ == "__main__":
    unittest.main()
