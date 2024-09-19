"""
Author: Remy Priem (remy.priem@onera.fr)

This package is distributed under New BSD license.
"""

from __future__ import division

import numpy as np
from scipy import linalg

from smt.surrogate_models.krg_based import KrgBased
from smt.utils.checks import check_nx, check_support, ensure_2d_array
from smt.utils.kriging import componentwise_distance, differences
from copy import deepcopy
import warnings
from smt.sampling_methods import LHS
from scipy import optimize


from scipy.stats import multivariate_normal as m_norm

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
        declare(
            "hyper_opt",
            "TNC",
            values=("TNC"),
            desc="Optimiser for hyperparameters optimisation",
            types=str,
        )
        declare(
            "corr",
            "act_exp",  # active subspace kernel only
            values=("act_exp"),
            desc="Correlation function type",
            types=(str),
        )

    def _optimize_hyperparam(self, D):
        """
        This function evaluates the Gaussian Process model at x.

        Parameters
        ----------
        D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The componentwise cross-spatial-correlation-distance between the
              vectors in X.
        Returns
        -------
        best_optimal_rlf_value: real
            - The value of the reduced likelihood function associated to the
              best autocorrelation parameters theta.
        best_optimal_par: dict()
            - A dictionary containing the requested Gaussian Process model
              parameters.
        best_optimal_theta: list(n_comp) or list(dim)
            - The best hyperparameters found by the optimization.
        """
        # reinitialize optimization best values
        self.best_iteration_fail = None
        self._thetaMemory = None
        # Initialize the hyperparameter-optimization

        def minus_reduced_likelihood_function(theta):
            res = -self._reduced_likelihood_function(theta)[0]
            return res

        def grad_minus_reduced_likelihood_function(theta):
            grad = -self._reduced_likelihood_gradient(theta)[0]
            return grad

        def hessian_minus_reduced_likelihood_function(theta):
            hess = -self._reduced_likelihood_hessian(theta)[0]
            return hess

        limit, _rhobeg = max(12 * len(self.options["theta0"]), 50), 0.5
        (
            best_optimal_theta,
            best_optimal_rlf_value,
            best_optimal_par,
            constraints,
        ) = (
            [],
            [],
            [],
            [],
        )

        bounds_hyp = []
        self.theta0 = deepcopy(self.options["theta0"])
        self.corr.theta = deepcopy(self.options["theta0"])
        for i in range(len(self.theta0)):
            # In practice, in 1D and for X in [0,1], theta^{-2} in [1e-2,infty),
            # i.e. theta in (0,1e1], is a good choice to avoid overfitting.
            # By standardising X in R, X_norm = (X-X_mean)/X_std, then
            # X_norm in [-1,1] if considering one std intervals. This leads
            # to theta in (0,2e1]
            theta_bounds = self.options["theta_bounds"]
            if self.theta0[i] < theta_bounds[0] or self.theta0[i] > theta_bounds[1]:
                warnings.warn(
                    f"theta0 is out the feasible bounds ({self.theta0}[{i}] out of \
                        [{theta_bounds[0]}, {theta_bounds[1]}]). \
                            A random initialisation is used instead."
                )
                self.theta0[i] = self.random_state.rand()
                self.theta0[i] = (
                    self.theta0[i] * (theta_bounds[1] - theta_bounds[0])
                    + theta_bounds[0]
                )

            constraints.append(lambda theta, i=i: theta[i] + theta_bounds[1])
            constraints.append(lambda theta, i=i: theta_bounds[1] - theta[i])
            bounds_hyp.append((-theta_bounds[1], theta_bounds[1]))

        theta0_rand = m_norm.rvs(
            self.options["prior"]["mean"] * len(self.theta0),
            self.options["prior"]["var"],
            1,
        )
        theta0 = self.theta0

        if not (self.is_continuous):
            self.D = D
        else:
            ##from abs distance to kernel distance
            self.D = self._componentwise_distance(D)

        # Initialization
        k, incr, stop, best_optimal_rlf_value, max_retry = 0, 0, 1, -1e20, 10
        while k < stop:
            # Use specified starting point as first guess
            self.noise0 = np.array(self.options["noise0"])
            noise_bounds = self.options["noise_bounds"]

            offset = 0
            if self.options["eval_noise"] and not self.options["use_het_noise"]:
                self.noise0[self.noise0 == 0.0] = noise_bounds[0]
                for i in range(len(self.noise0)):
                    if (
                        self.noise0[i] < noise_bounds[0]
                        or self.noise0[i] > noise_bounds[1]
                    ):
                        self.noise0[i] = noise_bounds[0]
                        warnings.warn(
                            "Warning: noise0 is out the feasible bounds. The lowest possible value is used instead."
                        )

                theta0 = np.concatenate(
                    [theta0, np.log10(np.array([self.noise0]).flatten())]
                )
                theta0_rand = np.concatenate(
                    [
                        theta0_rand,
                        np.log10(np.array([self.noise0]).flatten()),
                    ]
                )

                for i in range(len(self.noise0)):
                    noise_bounds = np.log10(noise_bounds)
                    constraints.append(
                        lambda log10t, i=i: log10t[offset + i + len(self.theta0)]
                        - noise_bounds[0]
                    )
                    constraints.append(
                        lambda log10t, i=i: noise_bounds[1]
                        - log10t[offset + i + len(self.theta0)]
                    )
                    bounds_hyp.append(noise_bounds)
            theta_limits = np.repeat(
                np.log10([theta_bounds]), repeats=len(theta0), axis=0
            )
            theta_all_loops = np.vstack((theta0, theta0_rand))
            if self.options["n_start"] > 1:
                sampling = LHS(
                    xlimits=theta_limits,
                    criterion="maximin",
                    random_state=self.random_state,
                )
                theta_lhs_loops = sampling(self.options["n_start"])
                theta_all_loops = np.vstack((theta_all_loops, theta_lhs_loops))

            optimal_theta_res = {"fun": float("inf")}
            optimal_theta_res_loop = None
            try:
                if self.options["hyper_opt"] == "Cobyla":
                    for theta0_loop in theta_all_loops:
                        optimal_theta_res_loop = optimize.minimize(
                            minus_reduced_likelihood_function,
                            theta0_loop,
                            constraints=[
                                {"fun": con, "type": "ineq"} for con in constraints
                            ],
                            method="COBYLA",
                            options={
                                "rhobeg": _rhobeg,
                                "tol": 1e-4,
                                "maxiter": limit,
                            },
                        )
                        if optimal_theta_res_loop["fun"] < optimal_theta_res["fun"]:
                            optimal_theta_res = optimal_theta_res_loop

                elif self.options["hyper_opt"] == "TNC":
                    if self.options["use_het_noise"]:
                        raise ValueError("For heteroscedastic noise, please use Cobyla")
                    for theta0_loop in theta_all_loops:
                        optimal_theta_res_loop = optimize.minimize(
                            minus_reduced_likelihood_function,
                            theta0_loop,
                            method="TNC",
                            jac=grad_minus_reduced_likelihood_function,
                            ###The hessian information is available but never used
                            #
                            ####hess=hessian_minus_reduced_likelihood_function,
                            bounds=bounds_hyp,
                            options={"maxfun": limit},
                        )
                        if optimal_theta_res_loop["fun"] < optimal_theta_res["fun"]:
                            optimal_theta_res = optimal_theta_res_loop

                if "x" not in optimal_theta_res:
                    raise ValueError(
                        f"Optimizer encountered a problem: {optimal_theta_res_loop!s}"
                    )
                optimal_theta = optimal_theta_res["x"]
                optimal_rlf_value, optimal_par = self._reduced_likelihood_function(
                    theta=optimal_theta
                )
                # Compare the new optimizer to the best previous one
                if k > 0:
                    if np.isinf(optimal_rlf_value):
                        stop += 1
                        if incr != 0:
                            return
                        if stop > max_retry:
                            raise ValueError(
                                "%d attempts to train the model failed" % max_retry
                            )
                    else:
                        if optimal_rlf_value >= self.best_iteration_fail:
                            if optimal_rlf_value > best_optimal_rlf_value:
                                best_optimal_rlf_value = optimal_rlf_value
                                best_optimal_par = optimal_par
                                best_optimal_theta = optimal_theta
                            else:
                                if self.best_iteration_fail > best_optimal_rlf_value:
                                    best_optimal_theta = self._thetaMemory
                                    (
                                        best_optimal_rlf_value,
                                        best_optimal_par,
                                    ) = self._reduced_likelihood_function(
                                        theta=best_optimal_theta
                                    )
                else:
                    if np.isinf(optimal_rlf_value):
                        stop += 1
                    else:
                        best_optimal_rlf_value = optimal_rlf_value
                        best_optimal_par = optimal_par
                        best_optimal_theta = optimal_theta
                k += 1
            except ValueError as ve:
                # raise ve
                # If iteration is max when fmin_cobyla fail is not reached
                if self.nb_ill_matrix > 0:
                    self.nb_ill_matrix -= 1
                    k += 1
                    stop += 1
                    # One evaluation objectif function is done at least
                    if self.best_iteration_fail is not None:
                        if self.best_iteration_fail > best_optimal_rlf_value:
                            best_optimal_theta = self._thetaMemory
                            (
                                best_optimal_rlf_value,
                                best_optimal_par,
                            ) = self._reduced_likelihood_function(
                                theta=best_optimal_theta
                            )
                # Optimization fail
                elif np.size(best_optimal_par) == 0:
                    print("Optimization failed. Try increasing the ``nugget``")
                    raise ve
                # Break the while loop
                else:
                    k = stop + 1
                    print("fmin_cobyla failed but the best value is retained")

        return best_optimal_rlf_value, best_optimal_par, best_optimal_theta

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

    def _predict_values(self, x, is_acting=None):
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
        self.corr.theta = theta
        r = self.corr(d, d_x=d_x).reshape(n_eval, self.nt)

        f = self._regression_types[self.options["poly"]](x)
        # Scaled predictor
        y_ = np.dot(f, self.optimal_par["beta"]) + np.dot(r, self.optimal_par["gamma"])
        # Predictor
        y = (self.y_mean + self.y_std * y_).ravel()
        return y

    def _predict_mgp_variances_base(self, x):
        """Base computation of MGP MSE used by predict_variances and predict_variances_no_uq"""
        _, n_features = x.shape

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

        return MSE, dy, dMSE

    def _predict_variances(
        self, x: np.ndarray, is_acting=None, is_ri=False
    ) -> np.ndarray:
        """
        Predict the variance of a specific point

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
        numpy.nd array
            MSE.
        """
        MSE, dy, dMSE = self._predict_mgp_variances_base(x)
        arg_1 = np.einsum("ij,ij->i", dy.T, linalg.solve(self.inv_sigma_R, dy).T)
        arg_2 = np.einsum("ij,ij->i", dMSE.T, linalg.solve(self.inv_sigma_R, dMSE).T)

        MGPMSE = np.zeros(x.shape[0])
        MGPMSE[MSE != 0] = (
            (4.0 / 3.0) * MSE[MSE != 0]
            + arg_1[MSE != 0]
            + (1.0 / (3.0 * MSE[MSE != 0])) * arg_2[MSE != 0]
        )
        MGPMSE[MGPMSE < 0.0] = 0.0
        return MGPMSE

    def predict_variances_no_uq(self, x):
        """Like predict_variances but without taking account hyperparameters uncertainty"""
        check_support(self, "variances")
        x = ensure_2d_array(x, "x")
        self._check_xdim(x)
        n = x.shape[0]
        _x2 = np.copy(x)
        s2, _, _ = self._predict_mgp_variances_base(x)
        s2[s2 < 0.0] = 0.0
        return s2.reshape((n, self.ny))

    def _check_xdim(self, x):
        _, n_features = x.shape
        nx = self.nx
        if n_features < self.nx:
            nx = self.options["n_comp"]
        check_nx(nx, x)

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
        self.corr.theta = theta
        r = self.corr(d, d_x=d_x).reshape(n_eval, self.nt)
        # Compute the regression function
        f = self._regression_types[self.options["poly"]](x)

        dy = np.zeros((len(self.optimal_theta), n_eval))

        gamma = self.optimal_par["gamma"]
        Rinv_dR_gamma = self.optimal_par["Rinv_dR_gamma"]
        Rinv_dmu = self.optimal_par["Rinv_dmu"]

        for omega in range(len(self.optimal_theta)):
            drdomega = self.corr(d, grad_ind=omega, d_x=d_x).reshape(n_eval, self.nt)

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
        self.corr.theta = theta
        r = self.corr(d, d_x=d_x).reshape(n_eval, self.nt).T
        f = self._regression_types[self.options["poly"]](x).T

        C = self.optimal_par["C"]
        G = self.optimal_par["G"]
        Ft = self.optimal_par["Ft"]
        sigma2 = self.optimal_par["sigma2"]

        rt = linalg.solve_triangular(C, r, lower=True)

        F_Rinv_r = np.dot(Ft.T, rt)

        u_ = linalg.solve_triangular(G.T, f - F_Rinv_r)

        MSE = self.optimal_par["sigma2"] * (
            1.0 - (rt**2.0).sum(axis=0) + (u_**2.0).sum(axis=0)
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
            drdomega = self.corr(d, grad_ind=omega, d_x=d_x).reshape(n_eval, self.nt).T

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
