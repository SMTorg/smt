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
            "squar_exp",
            values=("squar_exp"),
            desc="Correlation function type",
            types=(str),
        )
        declare(
            "poly",
            "constant",
            values=("constant"),
            desc="Regression function type",
            types=(str),
        )
        declare(
            "noise0",
            [0.01],
            desc="Gaussian noise on observed data",
            types=(list, np.ndarray),
        )
        declare(
            "method",
            "FITC",
            values=("FITC", "VFE"),
            desc="Method for sparse GP model",
            types=(str),
        )
        declare("n_inducing", 10, desc="Number of inducing inputs", types=int)
        self.Z = None
        self.woodbury_data = {"vec": None, "inv": None}
        self.optimal_par = {}

    def compute_K(self, A: np.ndarray, B: np.ndarray, theta, sigma2):
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

    def _new_train(self):
        if self.Z is None:
            self.set_inducing_inputs()

        # Has to evaluate the noise
        self.options["eval_noise"] = True

        # make sure the latent function is scalars
        Y = self.training_points[None][0][1]
        _, output_dim = Y.shape
        if output_dim > 1:
            raise NotImplementedError("FITC does not support vector-valued function")

        # make sure the noise is not hetero
        eta2 = np.array(self.options["noise0"])  # likelihood.gaussian_variance(Y_norma)
        if eta2.size > 1:
            raise NotImplementedError("FITC does not support heteroscedastic noise")

        return super()._new_train()

    # overload kriging based imlementation
    def _reduced_likelihood_function(self, theta):
        # print("variance=", theta[-1])
        # print("lengthscale=", 1.0 / np.sqrt(2.0 * theta[0:-1]))
        # print("theta=", theta[0:-1])
        X = self.training_points[None][0][0]
        Y = self.training_points[None][0][1]
        Z = self.Z

        sigma2 = theta[-1]
        theta = theta[0:-1]

        nugget = 1e-8

        if self.options["method"] == "VFE":
            likelihood, w_vec, w_inv = self._vfe(X, Y, Z, theta, sigma2, nugget)
        else:
            likelihood, w_vec, w_inv = self._fitc(X, Y, Z, theta, sigma2, nugget)

        self.woodbury_data["vec"] = w_vec
        self.woodbury_data["inv"] = w_inv

        params = {
            "theta": theta,
            "sigma2": sigma2,
        }
        # print(">>> lkh=", likelihood)
        return likelihood, params

    def _fitc(self, X, Y, Z, theta, sigma2, nugget):
        """FITC method implementation.
        See also https://github.com/SheffieldML/GPy/blob/9ec3e50e3b96a10db58175a206ed998ec5a8e3e3/GPy/inference/latent_function_inference/fitc.py
        """

        # Compute: diag(Knn), Kmm and Kmn
        Knn = np.full(self.nt, sigma2)
        Kmm = self.compute_K(Z, Z, theta, sigma2) + np.eye(self.nz) * nugget
        Kmn = self.compute_K(Z, X, theta, sigma2)

        # Compute (lower) Cholesky decomposition: Kmm = U U^T
        U = linalg.cholesky(Kmm, lower=True)

        # Compute (upper) Cholesky decomposition: Qnn = V^T V
        Ui = linalg.inv(U)
        V = Ui @ Kmn

        # Compute diagonal correction: nu = Knn_diag - Qnn_diag + \eta^2
        nu = Knn - np.sum(np.square(V), 0) + np.array(self.options["noise0"])
        # Compute beta, the effective noise precision
        beta = 1.0 / nu

        # Compute (lower) Cholesky decomposition: A = I + V diag(beta) V^T = L L^T
        A = np.eye(self.nz) + V @ np.diag(beta) @ V.T
        L = linalg.cholesky(A, lower=True)
        Li = linalg.inv(L)

        # back substitute to get b, P, v
        a = np.einsum("ij,i->ij", Y, beta)  # avoid reshape for mat-vec multiplication
        b = Li @ V @ a

        # For prediction
        LiUi = Li @ Ui
        LiUiT = LiUi.T
        woodbury_vec = LiUiT @ b
        woodbury_inv = Ui.T @ Ui - LiUiT @ LiUi

        # Compute marginal log-likelihood
        likelihood = -0.5 * (
            # num_data * np.log(2.0 * np.pi)   # constant term ignored in reduced likelihood
            +np.sum(np.log(nu))
            + 2.0 * np.sum(np.log(np.diag(L)))
            + a.T @ Y
            - np.einsum("ij,ij->", b, b)
        )

        return likelihood, woodbury_vec, woodbury_inv

    def _vfe(self, X, Y, Z, theta, sigma2, nugget):
        """VFE method implementation.
        See also https://github.com/SheffieldML/GPy/blob/9ec3e50e3b96a10db58175a206ed998ec5a8e3e3/GPy/inference/latent_function_inference/fitc.py
        """

        # model constants
        num_data, output_dim = Y.shape

        # Assume zero mean function
        mean = 0
        # Assume Gaussian likelihood (precision is equivalent to beta in FITC)
        precision = 1.0 / np.fmax(self.options["noise0"], nugget)

        # store some matrices
        VVT_factor = precision * (Y - mean)
        trYYT = np.einsum("ij,ij->", Y - mean, Y - mean)

        # kernel computations, using BGPLVM notation (psi stats)
        Kmm = self.compute_K(Z, Z, theta, sigma2) + np.eye(self.nz) * nugget
        Lm = linalg.cholesky(Kmm, lower=True)

        psi0 = np.full(self.nt, sigma2)
        psi1 = self.compute_K(X, Z, theta, sigma2)

        tmp = psi1 * (np.sqrt(precision))
        LmInv = linalg.inv(Lm)
        tmp = LmInv @ tmp.T
        A = tmp @ tmp.T

        B = np.eye(self.nz) + A
        LB = linalg.cholesky(B, lower=True)
        LBInv = linalg.inv(LB)

        tmp = LmInv @ psi1.T
        LBi_Lmi_psi1 = LBInv @ tmp
        LBi_Lmi_psi1Vf = LBi_Lmi_psi1 @ VVT_factor
        tmp = LBInv.T @ LBi_Lmi_psi1Vf
        Cpsi1Vf = LmInv.T @ tmp

        delit = LBi_Lmi_psi1Vf @ LBi_Lmi_psi1Vf.T
        data_fit = np.trace(delit)

        beta = precision
        lik_1 = (
            -0.5 * num_data * output_dim * (np.log(2.0 * np.pi) - np.log(beta))
            - 0.5 * beta * trYYT
        )
        lik_2 = -0.5 * output_dim * (np.sum(beta * psi0) - np.trace(A))
        lik_3 = -output_dim * (np.sum(np.log(np.diag(LB))))
        lik_4 = 0.5 * data_fit
        likelihood = lik_1 + lik_2 + lik_3 + lik_4

        woodbury_vec = Cpsi1Vf

        Bi = LBInv.T @ LBInv + np.eye(LBInv.shape[0])
        woodbury_inv = LmInv.T @ Bi @ LmInv

        return likelihood, woodbury_vec, woodbury_inv

    def _predict_values(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        Kx, _ = self._compute_KxKxx(x)
        mu = Kx.T @ self.woodbury_data["vec"]
        return mu

    def _predict_variances(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        Kx, Kxx = self._compute_KxKxx(x)
        var = (Kxx - np.sum(np.dot(self.woodbury_data["inv"].T, Kx) * Kx, 0))[:, None]
        var = np.clip(var, 1e-15, np.inf)
        var += self.options["noise0"]
        return var

    def _compute_KxKxx(self, x):
        Kx = self.compute_K(
            self.Z, x, self.optimal_par["theta"], self.optimal_par["sigma2"]
        )
        Kxx = np.full(x.shape[0], self.optimal_par["sigma2"])
        return Kx, Kxx
