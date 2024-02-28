# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 09:30:40 2023

@author: hvalayer, rlafage

This implementation in SMT is derived from FITC/VarDTC methods from
Sparse GP implementations of GPy project. See https://github.com/SheffieldML/GPy
"""

import numpy as np
from scipy import linalg

from smt.surrogate_models.krg import KRG
from smt.utils.checks import ensure_2d_array
from smt.utils.kriging import differences
from smt.utils.misc import standardization


class SGP(KRG):
    name = "SGP"

    def _initialize(self):
        super()._initialize()

        declare = self.options.declare
        declare(
            "corr",
            "squar_exp",  # gaussian kernel only
            values=("squar_exp"),
            desc="Correlation function type",
            types=(str),
        )
        declare(
            "poly",
            "constant",  # constant mean function
            values=("constant"),
            desc="Regression function type",
            types=(str),
        )
        declare(
            "theta_bounds",
            [1e-6, 1e2],  # upper bound increased compared to kriging-based one
            types=(list, np.ndarray),
            desc="bounds for hyperparameters",
        )
        declare(
            "noise0",
            [1e-2],
            desc="Gaussian noise on observed training data",
            types=(list, np.ndarray),
        )
        declare(
            "hyper_opt",
            "Cobyla",
            values=("Cobyla"),
            desc="Optimiser for hyperparameters optimisation",
            types=str,
        )
        declare(
            "eval_noise",
            True,  # for SGP evaluate noise by default
            types=bool,
            values=(True, False),
            desc="Noise is always evaluated",
        )
        declare(
            "nugget",
            1000.0
            * np.finfo(
                np.double
            ).eps,  # slightly increased compared to kriging-based one
            types=(float),
            desc="a jitter for numerical stability",
        )
        declare(
            "method",
            "FITC",
            values=("FITC", "VFE"),
            desc="Method used by sparse GP model",
            types=(str),
        )
        declare("n_inducing", 10, desc="Number of inducing inputs", types=int)

        supports = self.supports
        supports["derivatives"] = False
        supports["variances"] = True
        supports["variance_derivatives"] = False
        supports["x_hierarchy"] = False

        self.Z = None
        self.woodbury_data = {"vec": None, "inv": None}
        self.optimal_par = {}
        self.optimal_noise = None

    def set_inducing_inputs(self, Z=None, normalize=False):
        """
        Define number of inducing inputs or set the locations manually.
        When Z is not specified then points are picked randomly amongst the inputs training set.

        Parameters
        ----------
        nz : int, optional
            Number of inducing inputs.
        Z : np.ndarray [M,ndim], optional
            Inducing inputs.
        normalize : When Z is given, whether values should be normalized
        """
        if Z is None:
            self.nz = self.options["n_inducing"]
            X = self.training_points[None][0][0]  # [nt,nx]
            random_idx = np.random.permutation(self.nt)[: self.nz]
            self.Z = X[random_idx].copy()  # [nz,nx]
        else:
            Z = ensure_2d_array(Z, "Z")
            if self.nx != Z.shape[1]:
                raise ValueError("DimensionError: Z.shape[1] != X.shape[1]")
            self.Z = Z  # [nz,nx]
            if normalize:
                X = self.training_points[None][0][0]  # [nt,nx]
                y = self.training_points[None][0][1]
                self.normalize = True
                (
                    _,
                    _,
                    X_offset,
                    _,
                    X_scale,
                    _,
                ) = standardization(X, y)
                self.Z = (self.Z - X_offset) / X_scale
            else:
                self.normalize = False
            self.nz = Z.shape[0]

    # overload kriging based implementation
    def _new_train(self):
        if self.Z is None:
            self.set_inducing_inputs()

        # make sure the latent function is scalars
        Y = self.training_points[None][0][1]
        _, output_dim = Y.shape
        if output_dim > 1:
            raise NotImplementedError("SGP does not support vector-valued function")

        # make sure the noise is not hetero
        if self.options["use_het_noise"]:
            raise NotImplementedError("SGP does not support heteroscedastic noise")

        # make sure we are using continuous variables only
        if not self.is_continuous:
            raise NotImplementedError("SGP does not support mixed-integer variables")

        # works only with COBYLA because of no likelihood gradients
        if self.options["hyper_opt"] != "Cobyla":
            raise NotImplementedError("SGP works only with COBYLA internal opimizer")

        return super()._new_train()

    # overload kriging based implementation
    def _reduced_likelihood_function(self, theta):
        X = self.training_points[None][0][0]
        Y = self.training_points[None][0][1]
        Z = self.Z

        if self.options["eval_noise"]:
            sigma2 = theta[-2]
            noise = theta[-1]
            theta = theta[0:-2]
        else:
            sigma2 = theta[-1]
            noise = self.options["noise0"]
            theta = theta[0:-1]

        nugget = self.options["nugget"]

        if self.options["method"] == "VFE":
            likelihood, w_vec, w_inv = self._vfe(X, Y, Z, theta, sigma2, noise, nugget)
        else:
            likelihood, w_vec, w_inv = self._fitc(X, Y, Z, theta, sigma2, noise, nugget)

        self.woodbury_data["vec"] = w_vec
        self.woodbury_data["inv"] = w_inv

        params = {
            "theta": theta,
            "sigma2": sigma2,
        }

        return likelihood, params

    def _reduced_likelihood_gradient(self, theta):
        raise NotImplementedError(
            "SGP gradient of reduced likelihood not implemented yet"
        )

    def _reduced_likelihood_hessian(self, theta):
        raise NotImplementedError(
            "SGP hessian of reduced likelihood not implemented yet"
        )

    def _compute_K(self, A: np.ndarray, B: np.ndarray, theta, sigma2):
        """
        Compute the covariance matrix K between A and B
        """
        # Compute pairwise componentwise L1-distances between A and B
        dx = differences(A, B)
        d = self._componentwise_distance(dx)
        # Compute the correlation vector r and matrix R
        r = self._correlation_types[self.options["corr"]](theta, d)
        R = r.reshape(A.shape[0], B.shape[0])
        # Compute the covariance matrix K
        K = sigma2 * R
        return K

    def _fitc(self, X, Y, Z, theta, sigma2, noise, nugget):
        """
        FITC method implementation.
        """

        # Compute: diag(Knn), Kmm and Kmn
        Knn = np.full(self.nt, sigma2)
        Kmm = self._compute_K(Z, Z, theta, sigma2) + np.eye(self.nz) * nugget
        Kmn = self._compute_K(Z, X, theta, sigma2)

        # Compute (lower) Cholesky decomposition: Kmm = U U^T
        U = linalg.cholesky(Kmm, lower=True)

        # Compute (upper) Cholesky decomposition: Qnn = V^T V
        Ui = linalg.inv(U)
        V = Ui @ Kmn

        # Assumption on the gaussian noise on training outputs
        eta2 = noise

        # Compute diagonal correction: nu = Knn_diag - Qnn_diag + \eta^2
        nu = Knn - np.sum(np.square(V), 0) + eta2
        # Compute beta, the effective noise precision
        beta = 1.0 / nu

        # Compute (lower) Cholesky decomposition: A = I + V diag(beta) V^T = L L^T
        A = np.eye(self.nz) + V * beta @ V.T
        L = linalg.cholesky(A, lower=True)
        Li = linalg.inv(L)

        # Compute a and b
        a = np.einsum("ij,i->ij", Y, beta)  # avoid reshape for mat-vec multiplication
        b = Li @ V @ a

        # Compute marginal log-likelihood
        likelihood = -0.5 * (
            # num_data * np.log(2.0 * np.pi)   # constant term ignored in reduced likelihood
            +np.sum(np.log(nu))
            + 2.0 * np.sum(np.log(np.diag(L)))
            + a.T @ Y
            - np.einsum("ij,ij->", b, b)
        )

        # Store Woodbury vectors for prediction step
        LiUi = Li @ Ui
        LiUiT = LiUi.T
        woodbury_vec = LiUiT @ b
        woodbury_inv = Ui.T @ Ui - LiUiT @ LiUi

        return likelihood, woodbury_vec, woodbury_inv

    def _vfe(self, X, Y, Z, theta, sigma2, noise, nugget):
        """
        VFE method implementation.
        """

        # Assume zero mean function
        mean = 0
        Y -= mean

        # Compute: diag(Knn), Kmm and Kmn
        Kmm = self._compute_K(Z, Z, theta, sigma2) + np.eye(self.nz) * nugget
        Kmn = self._compute_K(Z, X, theta, sigma2)

        # Compute (lower) Cholesky decomposition: Kmm = U U^T
        U = linalg.cholesky(Kmm, lower=True)

        # Compute (upper) Cholesky decomposition: Qnn = V^T V
        Ui = linalg.inv(U)
        V = Ui @ Kmn

        # Compute beta, the effective noise precision
        beta = 1.0 / np.fmax(noise, nugget)

        # Compute A = beta * V @ V.T
        A = beta * V @ V.T

        # Compute (lower) Cholesky decomposition: B = I + A = L L^T
        B = np.eye(self.nz) + A
        L = linalg.cholesky(B, lower=True)
        Li = linalg.inv(L)

        # Compute b
        b = beta * Li @ V @ Y

        # Compute log-marginal likelihood
        likelihood = -0.5 * (
            # self.nt * np.log(2.0 * np.pi)   # constant term ignored in reduced likelihood
            -self.nt * np.log(beta)
            + 2.0 * np.sum(np.log(np.diag(L)))
            + beta * np.einsum("ij,ij->", Y, Y)
            - b.T @ b
            + self.nt * beta * sigma2
            - np.trace(A)
        )

        # Store Woodbury vectors for prediction step
        LiUi = Li @ Ui
        Bi = np.eye(self.nz) + Li.T @ Li
        woodbury_vec = LiUi.T @ b
        woodbury_inv = Ui.T @ Bi @ Ui

        return likelihood, woodbury_vec, woodbury_inv

    # overload kriging based implementation
    def _predict_values(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        """
        Evaluates the model at a set of points using the Woodbury vector.
        """
        Kx = self._compute_K(x, self.Z, self.optimal_theta, self.optimal_sigma2)
        mu = Kx @ self.woodbury_data["vec"]
        return mu

    # overload kriging based implementation
    def _predict_variances(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        """
        Evaluates the model at a set of points using the inverse Woodbury vector.
        """
        Kx = self._compute_K(self.Z, x, self.optimal_theta, self.optimal_sigma2)
        Kxx = np.full(x.shape[0], self.optimal_sigma2)
        var = (Kxx - np.sum(np.dot(self.woodbury_data["inv"].T, Kx) * Kx, 0))[:, None]
        var = np.clip(var, 1e-15, np.inf)
        var += self.optimal_noise
        return var
