# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 2024

@author: Lisa Pretsch <<lisa.pretsch@tum.de>>

Based on KRG from SMT by Dr. Mohamed Amine Bouhlel

With cooperative components Kriging model fit inspired by
distributed multi-disciplinary design optimization (MDO) approaches and
Zhan, D., Wu, J., Xing, H. et al.
A cooperative approach to efficient global optimization.
J Glob Optim 88, 327–357 (2024). https://doi.org/10.1007/s10898-023-01316-6
"""

from copy import deepcopy

import numpy as np
from scipy import optimize
from scipy.stats import multivariate_normal as m_norm

from smt.sampling_methods import LHS
from smt.surrogate_models.krg_based import KrgBased
from smt.utils.kriging import componentwise_distance


class CoopCompKRG(KrgBased):
    name = "Cooperative Components Kriging"

    """
    # Example with random components
    # (use physical components if available)
    n_comp = 3

    # Random design variable to component allocation
    comps = [*range(n_comp)]
    vars = [*range(n_dim)]
    random.shuffle(vars)
    comp_var = np.full((n_dim, n_comp), False)
    for c in comps:
        comp_size = int(n_dim/n_comp)
        start = c*comp_size
        end = (c+1)*comp_size
        if c+1 == n_comp:
            end = max((c+1)*comp_size, n_dim)
        comp_var[vars[start:end],c] = True

    # Cooperative components Kriging model fit
    model = CoopCompKRG(hyper_opt='Cobyla')
    for active_coop_comp in comps:
        model.set_training_values(x_train, y_train)
        model.train(active_coop_comp, comp_var)

    # Prediction as for ordinary Kriging
    y_pred = model.predict_values(x_test)
    """

    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        declare(
            "corr",
            "squar_exp",
            values=(
                "pow_exp",
                "abs_exp",
                "squar_exp",
                "squar_sin_exp",
                "matern52",
                "matern32",
            ),
            desc="Correlation function type",
            types=(str),
        )
        declare(
            "hyper_opt",
            "Cobyla",
            values=("Cobyla",),
            desc="Correlation function type",
            types=(str),
        )
        supports = self.supports
        supports["variances"] = True
        supports["derivatives"] = False
        supports["variance_derivatives"] = False
        supports["x_hierarchy"] = False

    def _componentwise_distance(self, dx, opt=0, theta=None, return_derivative=False):
        d = componentwise_distance(
            dx,
            self.options["corr"],
            self.nx,
            self.options["pow_exp_power"],
            theta=theta,
            return_derivative=return_derivative,
        )
        return d

    def _compute_pls(self, X, y):
        """
        Workaround for function call in _new_train()
        """
        return X, y

    def train(self, active_coop_comp, comp_var) -> None:
        """
        Train the model
        Overrides SurrogateModel implementation with additional parameters
        """

        # Set allocations of design variables to components
        if not isinstance(comp_var, np.ndarray) or comp_var.shape[0] != self.nx:
            raise ValueError("comp_var has the wrong data type or shape.")
        self.comp_var = [var for var in comp_var.T]

        # Set context vector to current best hyperparameter (used for inactive components)
        try:
            self.coop_theta = self.optimal_theta
        except AttributeError:
            self.coop_theta = self.options["theta0"] * np.ones((self.nx))

        # Hyperparameter optimization for active components
        self.active_comp_var = self.comp_var[active_coop_comp]
        self.num_active = np.count_nonzero(self.active_comp_var)
        super().train()

    def _optimize_hyperparam(self, D):
        """
        This function evaluates the Gaussian Process model at x.
        Overrides KrgBased implementation for component-wise cooperative
        hyperparameter optimization.

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
        if self.name in ["MGP"]:

            def minus_cooperative_reduced_likelihood_function(active_theta):
                theta = self.coop_theta
                theta[self.active_comp_var] = active_theta
                res = -self._reduced_likelihood_function(theta)[0]
                return res

            def grad_minus_cooperative_reduced_likelihood_function(active_theta):
                theta = self.coop_theta
                theta[self.active_comp_var] = active_theta
                grad = -self._reduced_likelihood_gradient(theta)[0]
                return grad

        else:

            def minus_cooperative_reduced_likelihood_function(active_log10t):
                theta = self.coop_theta
                theta[self.active_comp_var] = 10.0**active_log10t
                return -self._reduced_likelihood_function(theta)[0]

            def grad_minus_cooperative_reduced_likelihood_function(active_log10t):
                theta = self.coop_theta
                theta[self.active_comp_var] = 10.0**active_log10t
                theta_2d = np.atleast_2d(theta).T
                res = (
                    -np.log(10.0)
                    * (theta_2d)
                    * (self._reduced_likelihood_gradient(theta_2d)[0])
                )
                return res

        limit, _rhobeg = 10 * self.num_active, 0.5  # consider only active dimensions
        exit_function = False
        if "KPLSK" in self.name:
            n_iter = 1
        else:
            n_iter = 0

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

        for ii in range(n_iter, -1, -1):
            bounds_hyp = []

            self.theta0 = deepcopy(self.options["theta0"])
            for i in range(
                self.num_active
            ):  # range(len(self.theta0)): # set only active constraints
                # In practice, in 1D and for X in [0,1], theta^{-2} in [1e-2,infty),
                # i.e. theta in (0,1e1], is a good choice to avoid overfitting.
                # By standardising X in R, X_norm = (X-X_mean)/X_std, then
                # X_norm in [-1,1] if considering one std intervals. This leads
                # to theta in (0,2e1]
                theta_bounds = self.options["theta_bounds"]
                if self.theta0[i] < theta_bounds[0] or self.theta0[i] > theta_bounds[1]:
                    self.theta0[i] = np.random.rand()
                    self.theta0[i] = (
                        self.theta0[i] * (theta_bounds[1] - theta_bounds[0])
                        + theta_bounds[0]
                    )
                    print(
                        "Warning: theta0 is out the feasible bounds. A random initialisation is used instead."
                    )

                if self.name in ["MGP"]:  # to be discussed with R. Priem
                    constraints.append(lambda theta, i=i: theta[i] + theta_bounds[1])
                    constraints.append(lambda theta, i=i: theta_bounds[1] - theta[i])
                    bounds_hyp.append((-theta_bounds[1], theta_bounds[1]))
                else:
                    log10t_bounds = np.log10(theta_bounds)
                    constraints.append(lambda log10t, i=i: log10t[i] - log10t_bounds[0])
                    constraints.append(lambda log10t, i=i: log10t_bounds[1] - log10t[i])
                    bounds_hyp.append(log10t_bounds)

            if self.name in ["MGP"]:
                theta0_rand = m_norm.rvs(
                    self.options["prior"]["mean"] * len(self.theta0),
                    self.options["prior"]["var"],
                    1,
                )
                theta0 = self.theta0
            else:
                theta0_rand = np.random.rand(len(self.theta0))
                theta0_rand = (
                    theta0_rand * (log10t_bounds[1] - log10t_bounds[0])
                    + log10t_bounds[0]
                )
                theta0 = np.log10(self.theta0)
            ##from abs distance to kernel distance
            self.D = self._componentwise_distance(D, opt=ii)

            # Initialization
            k, stop, best_optimal_rlf_value = 0, 1, -1e20
            while k < stop:
                # Use specified starting point as first guess
                self.noise0 = np.array(self.options["noise0"])
                noise_bounds = self.options["noise_bounds"]
                if self.options["eval_noise"] and not self.options["use_het_noise"]:
                    self.noise0[self.noise0 == 0.0] = noise_bounds[0]
                    for i in range(len(self.noise0)):
                        if (
                            self.noise0[i] < noise_bounds[0]
                            or self.noise0[i] > noise_bounds[1]
                        ):
                            self.noise0[i] = noise_bounds[0]
                            print(
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
                            lambda log10t: log10t[i + len(self.theta0)]
                            - noise_bounds[0]
                        )
                        constraints.append(
                            lambda log10t: noise_bounds[1]
                            - log10t[i + len(self.theta0)]
                        )
                        bounds_hyp.append(noise_bounds)
                theta_limits = np.repeat(
                    np.log10([theta_bounds]), repeats=len(theta0), axis=0
                )
                theta_all_loops = np.vstack((theta0, theta0_rand))

                if self.options["n_start"] > 1:
                    sampling = LHS(
                        xlimits=theta_limits, criterion="maximin", random_state=41
                    )
                    theta_lhs_loops = sampling(self.options["n_start"])
                    theta_all_loops = np.vstack((theta_all_loops, theta_lhs_loops))

                optimal_theta_res = {"fun": float("inf")}
                try:
                    if (
                        self.options["hyper_opt"] == "Cobyla"
                        and self.options["n_start"] >= 1
                    ):
                        for theta0_loop in theta_all_loops:
                            optimal_theta_res_loop = optimize.minimize(
                                minus_cooperative_reduced_likelihood_function,  # adjusted likelihood function
                                theta0_loop[
                                    self.active_comp_var
                                ],  # only optimize part of theta
                                constraints=[  # only consider part of the constraints
                                    {"fun": con, "type": "ineq"} for con in constraints
                                ],
                                method="COBYLA",
                                options={
                                    "rhobeg": _rhobeg,
                                    "tol": 1e-4,
                                    "maxiter": limit,
                                },
                            )
                            if self.name not in ["MGP"]:
                                optimal_coop_theta = np.log10(self.coop_theta)
                            else:
                                optimal_coop_theta = self.coop_theta
                            optimal_coop_theta[self.active_comp_var] = (
                                optimal_theta_res_loop["x"]
                            )
                            # augment "x" in optimal_theta_res_loop with theta_best of other indices
                            optimal_theta_res_loop["x"] = optimal_coop_theta
                            if optimal_theta_res_loop["fun"] < optimal_theta_res["fun"]:
                                optimal_theta_res = optimal_theta_res_loop
                        optimal_theta = optimal_theta_res["x"]
                    elif (
                        self.options["hyper_opt"] == "TNC"
                        and self.options["n_start"] >= 1
                    ):
                        theta_all_loops = 10**theta_all_loops
                        for theta0_loop in theta_all_loops:
                            optimal_theta_res_loop = optimize.minimize(
                                minus_cooperative_reduced_likelihood_function,  # adjusted likelihood function
                                theta0_loop[
                                    self.active_comp_var
                                ],  # only optimize part of theta
                                method="TNC",
                                jac=grad_minus_cooperative_reduced_likelihood_function,
                                bounds=bounds_hyp,
                                options={"maxiter": 100},
                            )
                            if self.name not in ["MGP"]:
                                optimal_coop_theta = np.log10(self.coop_theta)
                            else:
                                optimal_coop_theta = self.coop_theta
                            optimal_coop_theta[self.active_comp_var] = (
                                optimal_theta_res_loop["x"]
                            )
                            optimal_theta_res_loop["x"] = optimal_coop_theta
                            # augment "x" in optimal_theta_res_loop with theta_best of other indices
                            if optimal_theta_res_loop["fun"] < optimal_theta_res["fun"]:
                                optimal_theta_res = optimal_theta_res_loop
                        optimal_theta = optimal_theta_res["x"]
                    else:
                        optimal_theta = theta0

                    if self.name not in ["MGP"]:
                        optimal_theta = 10**optimal_theta
                    optimal_rlf_value, optimal_par = self._reduced_likelihood_function(
                        theta=optimal_theta
                    )

                    # Compare the new optimizer to the best previous one
                    if k > 0:
                        if np.isinf(optimal_rlf_value):
                            raise RuntimeError(
                                "Cannot train the model: infinite likelihood found"
                            )
                        else:
                            if optimal_rlf_value >= self.best_iteration_fail:
                                if optimal_rlf_value > best_optimal_rlf_value:
                                    best_optimal_rlf_value = optimal_rlf_value
                                    best_optimal_par = optimal_par
                                    best_optimal_theta = optimal_theta
                                else:
                                    if (
                                        self.best_iteration_fail
                                        > best_optimal_rlf_value
                                    ):
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
                    if self.retry > 0:
                        self.retry -= 1
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
                    elif best_optimal_par == []:
                        print("Optimization failed. Try increasing the ``nugget``")
                        raise ve
                    # Break the while loop
                    else:
                        k = stop + 1
                        print("fmin_cobyla failed but the best value is retained")

            if "KPLSK" in self.name:
                if self.options["eval_noise"]:
                    # best_optimal_theta contains [theta, noise] if eval_noise = True
                    theta = best_optimal_theta[:-1]
                else:
                    # best_optimal_theta contains [theta] if eval_noise = False
                    theta = best_optimal_theta

                if exit_function:
                    return best_optimal_rlf_value, best_optimal_par, best_optimal_theta

                if self.options["corr"] == "squar_exp":
                    self.options["theta0"] = (theta * self.coeff_pls**2).sum(1)
                else:
                    self.options["theta0"] = (theta * np.abs(self.coeff_pls)).sum(1)

                self.options["n_comp"] = int(self.nx)
                limit = 10 * self.options["n_comp"]
                self.best_iteration_fail = None
                exit_function = True
        return best_optimal_rlf_value, best_optimal_par, best_optimal_theta

    def _predict_derivatives(self, x, kx):
        """
        Not implemented yet.
        Evaluates the derivatives at a set of points.
        """
        raise NotImplementedError(
            "Derivative prediction is not available for cooperative Kriging."
        )

    def _predict_variance_derivatives(self, x):
        """
        Not implemented yet.
        Provide the derivative of the variance of the model at a set of points.
        """
        raise NotImplementedError(
            "Derivative prediction is not available for cooperative Kriging."
        )
