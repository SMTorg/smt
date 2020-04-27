"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

from __future__ import division
import warnings
import numpy as np
from scipy import linalg, optimize, sparse
from smt.utils.kriging_utils import differences

from smt.surrogate_models.krg_based import KrgBased
from smt.utils.kriging_utils import componentwise_distance
import time

"""
The Active kriging class.
"""


class AKRG(KrgBased):
    def _initialize(self):
        super(AKRG, self)._initialize()
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
        self.options["noise"] = 0.0
        self.name = "Active Kriging"

    def _componentwise_distance(self, dx, opt=0, small=False):
        if small:
            d = componentwise_distance(dx, self.options["corr"], self.options["n_comp"])
        else:
            d = componentwise_distance(dx, self.options["corr"], self.nx)
        return d

    def predict_variances(self, x, both=False, restricted_domain=None):
        """
        Predict variances 
        
        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        both : TYPE, optional
            DESCRIPTION. The default is False.
        small : TYPE, optional
            DESCRIPTION. The default is False.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        n_eval, n_features = x.shape

        if n_features < self.nx:
            if n_features != self.options["n_comp"]:
                raise ValueError(
                    "dim(u) should be equal to %i" % self.options["n_comp"]
                )
            u = x
            x, _ = self.get_x_from_u(u, restricted_domain)

            u = u * self.embedding["norm"] - self.U_mean
            x = (x - self.X_mean) / self.X_std
        else:
            if n_features != self.nx:
                raise ValueError("dim(x) should be equal to %i" % self.X_std.shape[0])
            u = None
            x = (x - self.X_mean) / self.X_std

        dy = self._predict_value_derivatives_hyper(x, u)
        dMSE, MSE = self._predict_variance_derivatives_hyper(x, u)

        arg_1 = np.einsum("ij,ij->i", dy.T, self.sigma_R.dot(dy).T)

        arg_2 = np.einsum("ij,ij->i", dMSE.T, self.sigma_R.dot(dMSE).T)

        AMSE = np.zeros(x.shape[0])

        AMSE[MSE != 0] = (
            (4.0 / 3.0) * MSE[MSE != 0]
            + arg_1[MSE != 0]
            + (1.0 / (3.0 * MSE[MSE != 0])) * arg_2[MSE != 0]
        )

        AMSE[AMSE < 0.0] = 0.0

        if both:
            return AMSE, MSE
        else:
            return AMSE

    def predict_values(self, x, restricted_domain=None):
        """
        Predict values

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        u : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        y : TYPE
            DESCRIPTION.

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
            x, _ = self.get_x_from_u(u, restricted_domain)

            u = u * self.embedding["norm"] - self.U_mean
            du = differences(u, Y=self.U_norma.copy())
            d = self._componentwise_distance(du, small=True)

            # Get an approximation of x
            x = (x - self.X_mean) / self.X_std
            dx = differences(x, Y=self.X_norma.copy())
            d_x = self._componentwise_distance(dx)
        else:
            if n_features != self.nx:
                raise ValueError("dim(x) should be equal to %i" % self.X_std.shape[0])
            theta = self.optimal_theta

            # Get pairwise componentwise L1-distances to the input training set
            x = (x - self.X_mean) / self.X_std
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
        Compute the reduced log value, gradient or heassian of the hyperparameters prior

        Parameters
        ----------
        theta : list(n_comp), optional
            - An array containing the autocorrelation parameters at which the
              Gaussian Process model parameters should be determined.
              
        grad : boulean, optional
            True if the gradient must be computed. The default is False.
        hessian : boulean, optional
            True if the hessian must be computed. The default is False.

        Returns
        -------
        res : float or np.ndarray
            Reduced log value, gradient or hessian of the hyperparameters prior

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
        Predict the value derivatives with respect to the hyperparameters

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        u : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        dy : TYPE
            DESCRIPTION.

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
        Predict the variance derivatives with respect to the hyperparameters

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        u : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        dMSE : TYPE
            DESCRIPTION.
        MSE : TYPE
            DESCRIPTION.

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

    def get_x_from_u(self, u, restricted_domain=None):
        """
        Compute x from u

        Parameters
        ----------
        u : np.ndarray [n_evals, n_comp]
            Evaluation point input variable in embedding space

        Returns
        -------
        x : np.ndarray [n_evals, dim]
            Return point input variable values in original space

        """
        # TODO: Change to use qr decomposition
        res = []
        u = np.atleast_2d(u)

        self.embedding["Q_C"], self.embedding["R_C"]

        res = []
        x_temp = np.dot(
            self.embedding["Q_C"],
            linalg.solve_triangular(self.embedding["R_C"].T, u.T, lower=True),
        ).T
        # print(u)
        # print(x_temp)

        if restricted_domain is None:
            bounds = None
            res = x_temp
        else:
            bounds = [
                (restricted_domain[i, 0], restricted_domain[i, 1])
                for i in range(self.nx)
            ]
            for i, u_i in enumerate(u):
                con_fun = lambda x: self.get_u_from_x(x) - u_i
                con = [{"type": "eq", "fun": con_fun}]
                obj = lambda x: linalg.norm(x - x_temp[i, :])
                res_i = optimize.minimize(
                    obj,
                    np.random.rand(len(self.X_std)) * 2. - 1.,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=con,
                )
                res.append(res_i["x"])

        res = np.atleast_2d(res)
        return res, x_temp

    def get_u_from_x(self, x):
        """
        Compute u from x

        Parameters
        ----------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values in original space

        Returns
        -------
        u : np.ndarray [n_evals, n_comp]
            Return point input variable in embedding space
        """
        u = x.dot(self.embedding["C"])
        return u

    def _specific_train(self):
        """
        Specific training of Active Kriging

        Returns
        -------
        None.

        """
        # Compute covariance matrix of hyperparameters
        var_R = np.zeros((len(self.optimal_theta), len(self.optimal_theta)))
        r, r_ij, par = self._reduced_likelihood_hessian(self.optimal_theta)
        var_R[r_ij[:, 0], r_ij[:, 1]] = r[:, 0]
        var_R[r_ij[:, 1], r_ij[:, 0]] = r[:, 0]
        
        self.sigma_R = - linalg.inv(var_R)
        
        # Compute normalise embedding
        self.optimal_par = par

        A = np.reshape(self.optimal_theta, (self.options["n_comp"], self.nx)).T
        B = (A.T / self.X_std).T
        norm_B = np.linalg.norm(B)
        C = B / norm_B

        self.embedding = {}
        self.embedding["A"] = A
        self.embedding["C"] = C
        self.embedding["norm"] = norm_B
        self.embedding["Q_C"], self.embedding["R_C"] = linalg.qr(C, mode="economic")

        U = []
        for i in range(C.shape[1]):
            ub = np.sum(np.abs(C[:, i]))
            lb = -ub
            U.append((lb, ub))

        self.embedding["bounds"] = U

        # Compute normalisation in embeding base
        self.U_norma = self.X_norma.dot(A)
        self.U_mean = self.X_mean.dot(C) * norm_B

        # Compute best number of Components for Active Kriging
        svd = linalg.svd(A)
        svd_cumsum = np.cumsum(svd[1])
        svd_sum = np.sum(svd[1])
        self.best_ncomp = min(np.argwhere(svd_cumsum > 0.99 * svd_sum)) + 1
