"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

from smt.surrogate_models import KPLS
from smt.utils.kriging import componentwise_distance, componentwise_distance_PLS
import numpy as np
from copy import deepcopy
import warnings
from smt.sampling_methods import LHS
from scipy import optimize

from smt.utils.kriging import (
    compute_n_param,
)


class KPLSK(KPLS):
    name = "KPLSK"

    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        # KPLSK used only with "squar_exp" correlations
        declare(
            "corr",
            "squar_exp",
            values=("squar_exp", "abs_exp"),
            desc="Correlation function type",
            types=(str),
        )

    def _componentwise_distance(self, dx, opt=1, theta=None, return_derivative=False):
        if opt == 1:
            # Kriging step
            d = componentwise_distance(
                dx,
                self.options["corr"],
                self.nx,
                power=self.options["pow_exp_power"],
                theta=theta,
                return_derivative=return_derivative,
            )
        else:
            # KPLS step
            d = componentwise_distance_PLS(
                dx,
                self.options["corr"],
                self.n_comp,
                self.coeff_pls,
                power=self.options["pow_exp_power"],
                theta=theta,
                return_derivative=return_derivative,
            )
        return d

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
        if self.kplsk_second_loop is None:
            self.kplsk_second_loop = False
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
        self.theta0 = deepcopy(self.options["theta0"])
        self.n_comp = deepcopy(self.options["n_comp"])
        for ii in range(2):
            bounds_hyp = []
            self.kplsk_second_loop = ii == 1 or self.kplsk_second_loop
            self.corr.theta = deepcopy(self.theta0)
            for i in range(len(self.theta0)):
                # In practice, in 1D and for X in [0,1], theta^{-2} in [1e-2,infty),
                # i.e. theta in (0,1e1], is a good choice to avoid overfitting.
                # By standardising X in R, X_norm = (X-X_mean)/X_std, then
                # X_norm in [-1,1] if considering one std intervals. This leads
                # to theta in (0,2e1]
                theta_bounds = self.options["theta_bounds"]
                if self.theta0[i] < theta_bounds[0] or self.theta0[i] > theta_bounds[1]:
                    if self.theta0[i] - theta_bounds[1] > 0:
                        self.theta0[i] = theta_bounds[1] - 1e-10
                    else:
                        self.theta0[i] = theta_bounds[0] + 1e-10

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

            if not (self.is_continuous):
                self.D = D
            else:
                ##from abs distance to kernel distance
                self.D = self._componentwise_distance(D, opt=ii)

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
                if ii == 0:
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
                            raise ValueError(
                                "For heteroscedastic noise, please use Cobyla"
                            )
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
            if self.options["eval_noise"]:
                # best_optimal_theta contains [theta, noise] if eval_noise = True
                theta = best_optimal_theta[:-1]
            else:
                # best_optimal_theta contains [theta] if eval_noise = False
                theta = best_optimal_theta

            if self.kplsk_second_loop:
                return best_optimal_rlf_value, best_optimal_par, best_optimal_theta

            if self.options["corr"] == "squar_exp":
                self.theta0 = (theta * self.coeff_pls**2).sum(1)

            else:
                self.theta0 = (theta * np.abs(self.coeff_pls)).sum(1)
            self.n_param = compute_n_param(
                self.design_space,
                self.options["categorical_kernel"],
                self.nx,
                None,
                None,
            )
            self.n_comp = int(self.n_param)
            limit = 10 * self.options["n_comp"]
            self.best_iteration_fail = None
        return best_optimal_rlf_value, best_optimal_par, best_optimal_theta
