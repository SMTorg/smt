"""
Author: Remy Priem (remy.priem@onera.fr)

This package is distributed under New BSD license.
"""

from __future__ import division
import numpy as np
from scipy import linalg

from smt.utils.kriging_utils import differences
from smt.surrogate_models.krg_based import KrgBased
from smt.utils.kriging_utils import componentwise_distance

"""
The Active kriging class.
"""


class MGP(KrgBased):
    name = "MGP"

    def _initialize(self):
        """
        Initialized MGP

        """
        super(MGP, self)._initialize()
        declare = self.options.declare
        declare("n_comp", 1, types=int, desc="Number of active dimensions")
        declare(
            "prior",
            {"mean": [0.0], "var": 5.0 / 4.0},
            types=dict,
            desc="Parameters for Gaussian prior of the Hyperparameters",
        )
        self.options["hyper_opt"] = "TNC"
        self.options["corr"] = "act_exp"

    def _componentwise_distance(self, dx, small=False, opt=0):
        """
        Compute the componentwise distance with respect to the correlation kernel


        Parameters
        ----------
        dx : numpy.ndarray
            Distance matrix.
        small : bool, optional
            Compute the componentwise distance in small (n_components) dimension
            or in initial dimension. The default is False.
        opt : int, optional
            useless for MGP

        Returns
        -------
        d : numpy.ndarray
            Component wise distance.

        """
        if small:
            d = componentwise_distance(dx, self.options["corr"], self.options["n_comp"])
        else:
            d = componentwise_distance(dx, self.options["corr"], self.nx)
        return d

    def predict_variances(self, x, both=False):
        """
        Predict the variance of a specific point

        Parameters
        ----------
        x : numpy.ndarray
            Point to compute.
        both : bool, optional
            True if MSE and MGP-MSE wanted. The default is False.

        Raises
        ------
        ValueError
            The number fo dimension is not good.

        Returns
        -------
        numpy.nd array
            MSE or (MSE, MGP-MSE).

        """
        n_eval, n_features = x.shape

        if n_features < self.nx:
            if n_features != self.options["n_comp"]:
                raise ValueError(
                    "dim(u) should be equal to %i" % self.options["n_comp"]
                )
            u = x
            x = self.get_x_from_u(u)

            u = u * self.embedding["norm"] - self.U_mean
            x = (x - self.X_offset) / self.X_scale
        else:
            if n_features != self.nx:
                raise ValueError("dim(x) should be equal to %i" % self.X_scale.shape[0])
            u = None
            x = (x - self.X_offset) / self.X_scale

        dy = self._predict_value_derivatives_hyper(x, u)
        dMSE, MSE = self._predict_variance_derivatives_hyper(x, u)

        arg_1 = np.einsum("ij,ij->i", dy.T, linalg.solve(self.inv_sigma_R, dy).T)

        arg_2 = np.einsum("ij,ij->i", dMSE.T, linalg.solve(self.inv_sigma_R, dMSE).T)

        MGPMSE = np.zeros(x.shape[0])

        MGPMSE[MSE != 0] = (
            (4.0 / 3.0) * MSE[MSE != 0]
            + arg_1[MSE != 0]
            + (1.0 / (3.0 * MSE[MSE != 0])) * arg_2[MSE != 0]
        )

        MGPMSE[MGPMSE < 0.0] = 0.0

        if both:
            return MGPMSE, MSE
        else:
            return MGPMSE

    def predict_values(self, x):
        """
        Predict the value of the MGP for a given point

        Parameters
        ----------
        x : numpy.ndarray
            Point to compute.

        Raises
        ------
        ValueError
            The number fo dimension is not good.

        Returns
        -------
        y : numpy.ndarray
            Value of the MGP at the given point x.

        """
        n_eval, n_features = x.shape
        if n_features < self.nx:
            if n_features != self.options["n_comp"]:
                raise ValueError(
                    "dim(u) should be equal to %i" % self.options["n_comp"]
                )
            theta = np.eye(self.options["n_comp"]).reshape(
                (self.options["n_comp"] ** 2,)
            )
            # Get pairwise componentwise L1-distances to the input training set
            u = x
            x = self.get_x_from_u(u)

            u = u * self.embedding["norm"] - self.U_mean
            du = differences(u, Y=self.U_norma.copy())
            d = self._componentwise_distance(du, small=True)

            # Get an approximation of x
            x = (x - self.X_offset) / self.X_scale
            dx = differences(x, Y=self.X_norma.copy())
            d_x = self._componentwise_distance(dx)
        else:
            if n_features != self.nx:
                raise ValueError("dim(x) should be equal to %i" % self.X_scale.shape[0])
            theta = self.optimal_theta

            # Get pairwise componentwise L1-distances to the input training set
            x = (x - self.X_offset) / self.X_scale
            dx = differences(x, Y=self.X_norma.copy())
            d = self._componentwise_distance(dx)
            d_x = None

        # Compute the correlation function
        r = self._correlation_types[self.options["corr"]](theta, d, d_x=d_x).reshape(
            n_eval, self.nt
        )

        f = self._regression_types[self.options["poly"]](x)
        # Scaled predictor
        y_ = np.dot(f, self.optimal_par["beta"]) + np.dot(r, self.optimal_par["gamma"])
        # Predictor
        y = (self.y_mean + self.y_std * y_).ravel()
        return y

    def _reduced_log_prior(self, theta, grad=False, hessian=False):
        """
        Compute the reduced log prior at given hyperparameters

        Parameters
        ----------
        theta : numpy.ndarray
            Hyperparameters.
        grad : bool, optional
            True to compuyte gradient. The default is False.
        hessian : bool, optional
            True to compute hessian. The default is False.

        Returns
        -------
        res : numpy.ndarray
            Value, gradient, hessian of the reduced log prior.

        """
        nb_theta = len(theta)

        if theta.ndim < 2:
            theta = np.atleast_2d(theta).T

        mean = np.ones((nb_theta, 1)) * self.options["prior"]["mean"]
        sig_inv = np.eye(nb_theta) / self.options["prior"]["var"]

        if grad:
            sig_inv_m = np.atleast_2d(np.sum(sig_inv, axis=0)).T
            res = -2.0 * (theta - mean) * sig_inv_m
        elif hessian:
            res = -2.0 * np.atleast_2d(np.sum(sig_inv, axis=0)).T
        else:
            res = -np.dot((theta - mean).T, sig_inv.dot(theta - mean))
        return res

    def _predict_value_derivatives_hyper(self, x, u=None):
        """
        Compute the derivatives of the mean of the GP with respect to the hyperparameters

        Parameters
        ----------
        x : numpy.ndarray
            Point to compute in initial dimension.
        u : numpy.ndarray, optional
            Point to compute in small dimension. The default is None.

        Returns
        -------
        dy : numpy.ndarray
            Derivatives of the mean of the GP with respect to the hyperparameters.

        """
        # Initialization
        n_eval, _ = x.shape

        # Get pairwise componentwise L1-distances to the input training set
        dx = differences(x, Y=self.X_norma.copy())
        d_x = self._componentwise_distance(dx)
        if u is not None:
            theta = np.eye(self.options["n_comp"]).reshape(
                (self.options["n_comp"] ** 2,)
            )

            # Get pairwise componentwise L1-distances to the input training set
            du = differences(u, Y=self.U_norma.copy())
            d = self._componentwise_distance(du, small=True)
        else:
            theta = self.optimal_theta

            # Get pairwise componentwise L1-distances to the input training set
            d = d_x
            d_x = None

        # Compute the correlation function
        r = self._correlation_types[self.options["corr"]](theta, d, d_x=d_x).reshape(
            n_eval, self.nt
        )
        # Compute the regression function
        f = self._regression_types[self.options["poly"]](x)

        dy = np.zeros((len(self.optimal_theta), n_eval))

        gamma = self.optimal_par["gamma"]
        Rinv_dR_gamma = self.optimal_par["Rinv_dR_gamma"]
        Rinv_dmu = self.optimal_par["Rinv_dmu"]

        for omega in range(len(self.optimal_theta)):
            drdomega = self._correlation_types[self.options["corr"]](
                theta, d, grad_ind=omega, d_x=d_x
            ).reshape(n_eval, self.nt)

            dbetadomega = self.optimal_par["dbeta_all"][omega]

            dy_omega = (
                f.dot(dbetadomega)
                + drdomega.dot(gamma)
                - r.dot(Rinv_dR_gamma[omega] + Rinv_dmu[omega])
            )

            dy[omega, :] = dy_omega[:, 0]

        return dy

    def _predict_variance_derivatives_hyper(self, x, u=None):
        """
        Compute the derivatives of the variance of the GP with respect to the hyperparameters

        Parameters
        ----------
        x : numpy.ndarray
            Point to compute in initial dimension.
        u : numpy.ndarray, optional
            Point to compute in small dimension. The default is None.

        Returns
        -------
        dMSE : numpy.ndarrray
            derivatives of the variance of the GP with respect to the hyperparameters.
        MSE : TYPE
            Variance of the GP.

        """
        # Initialization
        n_eval, n_features_x = x.shape

        # Get pairwise componentwise L1-distances to the input training set
        dx = differences(x, Y=self.X_norma.copy())
        d_x = self._componentwise_distance(dx)
        if u is not None:
            theta = np.eye(self.options["n_comp"]).reshape(
                (self.options["n_comp"] ** 2,)
            )
            # Get pairwise componentwise L1-distances to the input training set
            du = differences(u, Y=self.U_norma.copy())
            d = self._componentwise_distance(du, small=True)
        else:
            theta = self.optimal_theta
            # Get pairwise componentwise L1-distances to the input training set
            d = d_x
            d_x = None

        # Compute the correlation function
        r = (
            self._correlation_types[self.options["corr"]](theta, d, d_x=d_x)
            .reshape(n_eval, self.nt)
            .T
        )
        f = self._regression_types[self.options["poly"]](x).T

        C = self.optimal_par["C"]
        G = self.optimal_par["G"]
        Ft = self.optimal_par["Ft"]
        sigma2 = self.optimal_par["sigma2"]

        rt = linalg.solve_triangular(C, r, lower=True)

        F_Rinv_r = np.dot(Ft.T, rt)

        u_ = linalg.solve_triangular(G.T, f - F_Rinv_r)

        MSE = self.optimal_par["sigma2"] * (
            1.0 - (rt ** 2.0).sum(axis=0) + (u_ ** 2.0).sum(axis=0)
        )
        # Mean Squared Error might be slightly negative depending on
        # machine precision: force to zero!
        MSE[MSE < 0.0] = 0.0

        Ginv_u = linalg.solve_triangular(G, u_, lower=False)
        Rinv_F = linalg.solve_triangular(C.T, Ft, lower=False)
        Rinv_r = linalg.solve_triangular(C.T, rt, lower=False)
        Rinv_F_Ginv_u = Rinv_F.dot(Ginv_u)

        dMSE = np.zeros((len(self.optimal_theta), n_eval))

        dr_all = self.optimal_par["dr"]
        dsigma = self.optimal_par["dsigma"]

        for omega in range(len(self.optimal_theta)):
            drdomega = (
                self._correlation_types[self.options["corr"]](
                    theta, d, grad_ind=omega, d_x=d_x
                )
                .reshape(n_eval, self.nt)
                .T
            )

            dRdomega = np.zeros((self.nt, self.nt))
            dRdomega[self.ij[:, 0], self.ij[:, 1]] = dr_all[omega][:, 0]
            dRdomega[self.ij[:, 1], self.ij[:, 0]] = dr_all[omega][:, 0]

            # Compute du2dtheta

            dRdomega_Rinv_F_Ginv_u = dRdomega.dot(Rinv_F_Ginv_u)
            r_Rinv_dRdomega_Rinv_F_Ginv_u = np.einsum(
                "ij,ij->i", Rinv_r.T, dRdomega_Rinv_F_Ginv_u.T
            )
            drdomega_Rinv_F_Ginv_u = np.einsum("ij,ij->i", drdomega.T, Rinv_F_Ginv_u.T)
            u_Ginv_F_Rinv_dRdomega_Rinv_F_Ginv_u = np.einsum(
                "ij,ij->i", Rinv_F_Ginv_u.T, dRdomega_Rinv_F_Ginv_u.T
            )

            du2domega = (
                2.0 * r_Rinv_dRdomega_Rinv_F_Ginv_u
                - 2.0 * drdomega_Rinv_F_Ginv_u
                + u_Ginv_F_Rinv_dRdomega_Rinv_F_Ginv_u
            )
            du2domega = np.atleast_2d(du2domega)

            # Compute drt2dtheta
            drdomega_Rinv_r = np.einsum("ij,ij->i", drdomega.T, Rinv_r.T)
            r_Rinv_dRdomega_Rinv_r = np.einsum(
                "ij,ij->i", Rinv_r.T, dRdomega.dot(Rinv_r).T
            )

            drt2domega = 2.0 * drdomega_Rinv_r - r_Rinv_dRdomega_Rinv_r
            drt2domega = np.atleast_2d(drt2domega)

            dMSE[omega] = dsigma[omega] * MSE / sigma2 + sigma2 * (
                -drt2domega + du2domega
            )

        return dMSE, MSE

    def get_x_from_u(self, u):
        """
        Compute the point in initial dimension from a point in low dimension

        Parameters
        ----------
        u : numpy.ndarray
            Point in low dimension.

        Returns
        -------
        res : numpy.ndarray
            point in initial dimension.

        """
        u = np.atleast_2d(u)

        self.embedding["Q_C"], self.embedding["R_C"]

        x_temp = np.dot(
            self.embedding["Q_C"],
            linalg.solve_triangular(self.embedding["R_C"].T, u.T, lower=True),
        ).T

        res = np.atleast_2d(x_temp)
        return res

    def get_u_from_x(self, x):
        """
        Compute the point in low dimension from a point in initial dimension

        Parameters
        ----------
        x : numpy.ndarray
            Point in initial dimension.

        Returns
        -------
        u : numpy.ndarray
             Point in low dimension.

        """
        u = x.dot(self.embedding["C"])
        return u

    def _specific_train(self):
        """
        Compute the specific training values necessary for MGP (Hessian)
        """
        # Compute covariance matrix of hyperparameters
        var_R = np.zeros((len(self.optimal_theta), len(self.optimal_theta)))
        r, r_ij, par = self._reduced_likelihood_hessian(self.optimal_theta)
        var_R[r_ij[:, 0], r_ij[:, 1]] = r[:, 0]
        var_R[r_ij[:, 1], r_ij[:, 0]] = r[:, 0]

        self.inv_sigma_R = -var_R

        # Compute normalise embedding
        self.optimal_par = par

        A = np.reshape(self.optimal_theta, (self.options["n_comp"], self.nx)).T
        B = (A.T / self.X_scale).T
        norm_B = np.linalg.norm(B)
        C = B / norm_B

        self.embedding = {}
        self.embedding["A"] = A
        self.embedding["C"] = C
        self.embedding["norm"] = norm_B
        self.embedding["Q_C"], self.embedding["R_C"] = linalg.qr(C, mode="economic")

        # Compute normalisation in embeding base
        self.U_norma = self.X_norma.dot(A)
        self.U_mean = self.X_offset.dot(C) * norm_B

        # Compute best number of Components for Active Kriging
        svd = linalg.svd(A)
        svd_cumsum = np.cumsum(svd[1])
        svd_sum = np.sum(svd[1])
        self.best_ncomp = min(np.argwhere(svd_cumsum > 0.99 * svd_sum)) + 1

    def _check_param(self):
        """
        Overrides KrgBased implementation
        This function checks some parameters of the model.
        """

        d = self.options["n_comp"] * self.nx

        if self.options["corr"] != "act_exp":
            raise ValueError("MGP must be used with act_exp correlation function")
        if self.options["hyper_opt"] != "TNC":
            raise ValueError("MGP must be used with TNC hyperparameters optimizer")

        if len(self.options["theta0"]) != d:
            if len(self.options["theta0"]) == 1:
                self.options["theta0"] *= np.ones(d)
            else:
                raise ValueError(
                    "the number of dim %s should be equal to the length of theta0 %s."
                    % (d, len(self.options["theta0"]))
                )
