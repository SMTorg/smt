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
from copy import deepcopy
import warnings
from smt.sampling_methods import LHS
from scipy import optimize


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

    def _optimize_hyperparam(self, D=None):
        """
        This function evaluates the Gaussian Process model at x.

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
        def minus_reduced_likelihood_function(log10t):
            return -self._reduced_likelihood_function(theta=10.0**log10t)[0]

        def grad_minus_reduced_likelihood_function(log10t):
            log10t_2d = np.atleast_2d(log10t).T
            res = (
                -np.log(10.0)
                * (10.0**log10t_2d)
                * (self._reduced_likelihood_gradient(10.0**log10t_2d)[0])
            )
            return res

        def hessian_minus_reduced_likelihood_function(log10t):
            log10t_2d = np.atleast_2d(log10t).T
            res = (
                -np.log(10.0)
                * (10.0**log10t_2d)
                * (self._reduced_likelihood_hessian(10.0**log10t_2d)[0])
            )
            return res

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

            log10t_bounds = np.log10(theta_bounds)
            constraints.append(lambda log10t, i=i: log10t[i] - log10t_bounds[0])
            constraints.append(lambda log10t, i=i: log10t_bounds[1] - log10t[i])
            bounds_hyp.append(log10t_bounds)

        theta_bounds = self.options["theta_bounds"]
        log10t_bounds = np.log10(theta_bounds)
        theta0_rand = self.random_state.rand(len(self.theta0))
        theta0_rand = (
            theta0_rand * (log10t_bounds[1] - log10t_bounds[0]) + log10t_bounds[0]
        )
        theta0 = np.log10(self.theta0)

        # Initialization
        k, incr, stop, best_optimal_rlf_value, max_retry = 0, 0, 1, -1e20, 10
        while k < stop:
            # Use specified starting point as first guess
            self.noise0 = np.array(self.options["noise0"])
            noise_bounds = self.options["noise_bounds"]

            # SGP: GP variance is optimized too
            offset = 0
            sigma2_0 = np.log10(np.array([self.y_std[0] ** 2]))
            theta0_sigma2 = np.concatenate([theta0, sigma2_0])
            sigma2_bounds = np.log10(np.array([1e-12, (3.0 * self.y_std[0]) ** 2]))
            constraints.append(
                lambda log10t: log10t[len(self.theta0)] - sigma2_bounds[0]
            )
            constraints.append(
                lambda log10t: sigma2_bounds[1] - log10t[len(self.theta0)]
            )
            bounds_hyp.append(sigma2_bounds)
            offset = 1
            theta0 = theta0_sigma2
            theta0_rand = np.concatenate([theta0_rand, sigma2_0])

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

                optimal_theta = 10**optimal_theta

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
        self.corr.theta = theta
        r = self.corr(d)
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
