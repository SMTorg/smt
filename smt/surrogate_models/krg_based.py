"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>

Some functions are copied from gaussian_process submodule (Scikit-learn 0.14)

This package is distributed under New BSD license.

TODO:
- fail_iteration and nb_iter_max to remove from options
- define outputs['sol'] = self.sol
"""

from __future__ import division

# import warnings
# warnings.filterwarnings("ignore")

import numpy as np
from scipy import linalg, optimize
from types import FunctionType
from smt.utils.caching import cached_operation

from smt.surrogate_models.surrogate_model import SurrogateModel
from sklearn.metrics.pairwise import manhattan_distances
from smt.utils.kriging_utils import constant, linear, quadratic
from smt.utils.kriging_utils import (
    abs_exp,
    squar_exp,
    standardization,
    l1_cross_distances,
)

from scipy.optimize import minimize

"""
The kriging class.
"""


class KrgBased(SurrogateModel):

    _regression_types = {"constant": constant, "linear": linear, "quadratic": quadratic}

    _correlation_types = {"abs_exp": abs_exp, "squar_exp": squar_exp}

    def _initialize(self):
        super(KrgBased, self)._initialize()
        declare = self.options.declare
        supports = self.supports
        declare(
            "poly",
            "constant",
            values=("constant", "linear", "quadratic"),
            desc="Regression function type",
        )
        declare(
            "corr",
            "squar_exp",
            values=("abs_exp", "squar_exp"),
            desc="Correlation function type",
        )
        declare(
            "data_dir",
            types=str,
            desc="Directory for loading / saving cached data; None means do not save or load",
        )
        declare(
            "theta0", [1e-2], types=(list, np.ndarray), desc="Initial hyperparameters"
        )
        self.name = "KrigingBased"
        self.best_iteration_fail = None
        self.nb_ill_matrix = 5
        supports["derivatives"] = True
        supports["variances"] = True

    ############################################################################
    # Model functions
    ############################################################################

    def _new_train(self):

        """
        Train the model
        """
        self._check_param()

        # Sampling points X and y
        X = self.training_points[None][0][0]
        y = self.training_points[None][0][1]

        # Compute PLS-coefficients (attr of self) and modified X and y (if GEKPLS is used)
        if self.name != "Kriging":
            X, y = self._compute_pls(X.copy(), y.copy())

        # Center and scale X and y
        self.X_norma, self.y_norma, self.X_mean, self.y_mean, self.X_std, self.y_std = standardization(
            X, y
        )

        # Calculate matrix of distances D between samples
        D, self.ij = l1_cross_distances(self.X_norma)
        if np.min(np.sum(D, axis=1)) == 0.0:
            raise Exception("Multiple input features cannot have the same value.")

        # Regression matrix and parameters
        self.F = self._regression_types[self.options["poly"]](self.X_norma)
        n_samples_F = self.F.shape[0]
        if self.F.ndim > 1:
            p = self.F.shape[1]
        else:
            p = 1
        self._check_F(n_samples_F, p)

        # Optimization
        self.optimal_rlf_value, self.optimal_par, self.optimal_theta = self._optimize_hyperparam(
            D
        )
        if self.name == "MFK":
            if self.options["eval_noise"]:
                self.optimal_theta = self.optimal_theta[:-1]
        del self.y_norma, self.D

    def _train(self):
        """
        Train the model
        """
        inputs = {"self": self}
        with cached_operation(inputs, self.options["data_dir"]) as outputs:
            if outputs:
                self.sol = outputs["sol"]
            else:
                self._new_train()
                # outputs['sol'] = self.sol

    def _reduced_likelihood_function(self, theta):

        """
        This function determines the BLUP parameters and evaluates the reduced
        likelihood function for the given autocorrelation parameters theta.

        Maximizing this function wrt the autocorrelation parameters theta is
        equivalent to maximizing the likelihood of the assumed joint Gaussian
        distribution of the observations y evaluated onto the design of
        experiments X.

        Parameters
        ----------
        theta: list(n_comp), optional
            - An array containing the autocorrelation parameters at which the
              Gaussian Process model parameters should be determined.

        Returns
        -------
        reduced_likelihood_function_value: real
            - The value of the reduced likelihood function associated to the
              given autocorrelation parameters theta.

        par: dict()
            - A dictionary containing the requested Gaussian Process model
              parameters:

            sigma2
            Gaussian Process variance.
            beta
            Generalized least-squares regression weights for
            Universal Kriging or for Ordinary Kriging.
            gamma
            Gaussian Process weights.
            C
            Cholesky decomposition of the correlation matrix [R].
            Ft
            Solution of the linear equation system : [R] x Ft = F
            G
            QR decomposition of the matrix Ft.
        """
        # Initialize output
        reduced_likelihood_function_value = -np.inf
        par = {}
        # Set up R
        MACHINE_EPSILON = np.finfo(np.double).eps
        nugget = 10.0 * MACHINE_EPSILON
        if self.name == "MFK":
            if self._lvl != self.nlvl:
                # in the case of multi-fidelity optimization
                # it is very probable that lower-fidelity correlation matrix
                # becomes ill-conditionned
                nugget = 10.0 * nugget
        noise = 0.0
        tmp_var = theta
        if self.name == "MFK":
            if self.options["eval_noise"]:
                theta = tmp_var[:-1]
                noise = tmp_var[-1]

        r = self._correlation_types[self.options["corr"]](theta, self.D).reshape(-1, 1)

        R = np.eye(self.nt) * (1.0 + nugget + noise)
        R[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
        R[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]

        # Cholesky decomposition of R
        try:
            C = linalg.cholesky(R, lower=True)
        except (linalg.LinAlgError, ValueError) as e:
            print("exception : ", e)
            return reduced_likelihood_function_value, par

        # Get generalized least squares solution
        Ft = linalg.solve_triangular(C, self.F, lower=True)
        Q, G = linalg.qr(Ft, mode="economic")
        sv = linalg.svd(G, compute_uv=False)
        rcondG = sv[-1] / sv[0]
        if rcondG < 1e-10:
            # Check F
            sv = linalg.svd(self.F, compute_uv=False)
            condF = sv[0] / sv[-1]
            if condF > 1e15:
                raise Exception(
                    "F is too ill conditioned. Poor combination "
                    "of regression model and observations."
                )

            else:
                # Ft is too ill conditioned, get out (try different theta)
                return reduced_likelihood_function_value, par

        Yt = linalg.solve_triangular(C, self.y_norma, lower=True)
        beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
        rho = Yt - np.dot(Ft, beta)

        # The determinant of R is equal to the squared product of the diagonal
        # elements of its Cholesky decomposition C
        detR = (np.diag(C) ** (2.0 / self.nt)).prod()

        # Compute/Organize output
        if self.name == "MFK":
            n_samples = self.nt
            p = self.p
            q = self.q
            sigma2 = (rho ** 2.0).sum(axis=0) / (n_samples - p - q)
            reduced_likelihood_function_value = -(n_samples - p - q) * np.log10(
                sigma2
            ) - n_samples * np.log10(detR)
        else:
            sigma2 = (rho ** 2.0).sum(axis=0) / (self.nt)
            reduced_likelihood_function_value = -sigma2.sum() * detR
        par["sigma2"] = sigma2 * self.y_std ** 2.0
        par["beta"] = beta
        par["gamma"] = linalg.solve_triangular(C.T, rho)
        par["C"] = C
        par["Ft"] = Ft
        par["G"] = G

        # A particular case when f_min_cobyla fail
        if (self.best_iteration_fail is not None) and (
            not np.isinf(reduced_likelihood_function_value)
        ):

            if reduced_likelihood_function_value > self.best_iteration_fail:
                self.best_iteration_fail = reduced_likelihood_function_value
                self._thetaMemory = np.array(tmp_var)

        elif (self.best_iteration_fail is None) and (
            not np.isinf(reduced_likelihood_function_value)
        ):
            self.best_iteration_fail = reduced_likelihood_function_value
            self._thetaMemory = np.array(tmp_var)

        return reduced_likelihood_function_value, par

    def _predict_values(self, x):
        """
        Evaluates the model at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        y : np.ndarray
            Evaluation point output variable values
        """
        # Initialization
        n_eval, n_features_x = x.shape
        x = (x - self.X_mean) / self.X_std
        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(x, Y=self.X_norma.copy(), sum_over_features=False)
        d = self._componentwise_distance(dx)
        # Compute the correlation function
        r = self._correlation_types[self.options["corr"]](
            self.optimal_theta, d
        ).reshape(n_eval, self.nt)
        y = np.zeros(n_eval)
        # Compute the regression function
        f = self._regression_types[self.options["poly"]](x)
        # Scaled predictor
        y_ = np.dot(f, self.optimal_par["beta"]) + np.dot(r, self.optimal_par["gamma"])
        # Predictor
        y = (self.y_mean + self.y_std * y_).ravel()

        return y

    def _predict_derivatives(self, x, kx):
        """
        Evaluates the derivatives at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values
        kx : int
            The 0-based index of the input variable with respect to which derivatives are desired.

        Returns
        -------
        y : np.ndarray
            Derivative values.
        """
        kx += 1

        # Initialization
        n_eval, n_features_x = x.shape
        x = (x - self.X_mean) / self.X_std
        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(x, Y=self.X_norma.copy(), sum_over_features=False)
        d = self._componentwise_distance(dx)
        # Compute the correlation function
        r = self._correlation_types[self.options["corr"]](
            self.optimal_theta, d
        ).reshape(n_eval, self.nt)

        if self.options["corr"] != "squar_exp":
            raise ValueError(
                "The derivative is only available for square exponential kernel"
            )
        if self.options["poly"] == "constant":
            df = np.array([0])
        elif self.options["poly"] == "linear":
            df = np.zeros((self.nx + 1, self.nx))
            df[1:, :] = 1
        else:
            raise ValueError(
                "The derivative is only available for ordinary kriging or "
                + "universal kriging using a linear trend"
            )

        # Beta and gamma = R^-1(y-FBeta)
        beta = self.optimal_par["beta"]
        gamma = self.optimal_par["gamma"]
        df_dx = np.dot(df.T, beta)
        d_dx = x[:, kx - 1].reshape((n_eval, 1)) - self.X_norma[:, kx - 1].reshape(
            (1, self.nt)
        )
        if self.name != "Kriging" and self.name != "KPLSK":
            theta = np.sum(self.optimal_theta * self.coeff_pls ** 2, axis=1)
        else:
            theta = self.optimal_theta
        y = (
            (df_dx[0] - 2 * theta[kx - 1] * np.dot(d_dx * r, gamma))
            * self.y_std
            / self.X_std[kx - 1]
        )

        return y

    def _predict_variances(self, x):

        # Initialization
        n_eval, n_features_x = x.shape
        x = (x - self.X_mean) / self.X_std
        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(x, Y=self.X_norma.copy(), sum_over_features=False)
        d = self._componentwise_distance(dx)

        # Compute the correlation function
        r = self._correlation_types[self.options["corr"]](self.optimal_theta, d).reshape(n_eval, self.nt)

        C = self.optimal_par["C"]
        rt = linalg.solve_triangular(C, r.T, lower=True)

        u = linalg.solve_triangular(
            self.optimal_par["G"].T,
            np.dot(self.optimal_par["Ft"].T, rt) - self._regression_types[self.options["poly"]](x).T,
        )

        A = self.optimal_par["sigma2"]
        B = 1.0 - (rt ** 2.0).sum(axis=0) + (u ** 2.0).sum(axis=0)
        MSE = np.einsum('i,j -> ji', A, B)

        # Mean Squared Error might be slightly negative depending on
        # machine precision: force to zero!
        MSE[MSE < 0.0] = 0.0
        return MSE

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
            return -self._reduced_likelihood_function(theta=10.0 ** log10t)[0]

        limit, _rhobeg = 10 * len(self.options["theta0"]), 0.5
        exit_function = False
        if self.name == "KPLSK":
            n_iter = 1
        else:
            n_iter = 0

        for ii in range(n_iter, -1, -1):
            best_optimal_theta, best_optimal_rlf_value, best_optimal_par, constraints = (
                [],
                [],
                [],
                [],
            )

            for i in range(len(self.options["theta0"])):
                constraints.append(lambda log10t, i=i: log10t[i] - np.log10(1e-6))
                constraints.append(lambda log10t, i=i: np.log10(100) - log10t[i])

            self.D = self._componentwise_distance(D, opt=ii)

            # Initialization
            k, incr, stop, best_optimal_rlf_value = 0, 0, 1, -1e20
            while k < stop:
                # Use specified starting point as first guess
                theta0 = self.options["theta0"]
                if self.name == "MFK":
                    if self.options["eval_noise"]:
                        theta0 = np.concatenate(
                            [theta0, np.array([self.options["noise0"]])]
                        )
                        constraints.append(lambda log10t: log10t[-1] + 16)
                        constraints.append(lambda log10t: 10 - log10t[-1])
                try:
                    optimal_theta = 10.0 ** optimize.fmin_cobyla(
                        minus_reduced_likelihood_function,
                        np.log10(theta0),
                        constraints,
                        rhobeg=_rhobeg,
                        rhoend=1e-4,
                        maxfun=limit,
                    )
                    optimal_rlf_value, optimal_par = self._reduced_likelihood_function(
                        theta=optimal_theta
                    )
                    # Compare the new optimizer to the best previous one
                    if k > 0:
                        if np.isinf(optimal_rlf_value):
                            stop += 1
                            if incr != 0:
                                return
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
                                        best_optimal_rlf_value, best_optimal_par = self._reduced_likelihood_function(
                                            theta=best_optimal_theta
                                        )
                    else:
                        if np.isinf(optimal_rlf_value):
                            stop += 1
                        else:
                            if optimal_rlf_value >= self.best_iteration_fail:
                                if optimal_rlf_value > best_optimal_rlf_value:
                                    best_optimal_rlf_value = optimal_rlf_value
                                    best_optimal_par = optimal_par
                                    best_optimal_theta = optimal_theta

                            else:
                                if self.best_iteration_fail > best_optimal_rlf_value:
                                    best_optimal_theta = self._thetaMemory.copy()
                                    best_optimal_rlf_value, best_optimal_par = self._reduced_likelihood_function(
                                        theta=best_optimal_theta
                                    )
                    k += 1
                except ValueError as ve:
                    # If iteration is max when fmin_cobyla fail is not reached
                    if self.nb_ill_matrix > 0:
                        self.nb_ill_matrix -= 1
                        k += 1
                        stop += 1
                        # One evaluation objectif function is done at least
                        if self.best_iteration_fail is not None:
                            if self.best_iteration_fail > best_optimal_rlf_value:
                                best_optimal_theta = self._thetaMemory
                                best_optimal_rlf_value, best_optimal_par = self._reduced_likelihood_function(
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

            if self.name == "KPLSK":
                if exit_function:
                    return best_optimal_rlf_value, best_optimal_par, best_optimal_theta

                if self.options["corr"] == "squar_exp":
                    self.options["theta0"] = (
                        best_optimal_theta * self.coeff_pls ** 2
                    ).sum(1)
                else:
                    self.options["theta0"] = (
                        best_optimal_theta * np.abs(self.coeff_pls)
                    ).sum(1)
                self.options["n_comp"] = int(self.nx)
                limit = 10 * self.options["n_comp"]
                self.best_iteration_fail = None
                exit_function = True

        return best_optimal_rlf_value, best_optimal_par, best_optimal_theta

    def _check_param(self):
        """
        This function check some parameters of the model.
        """
        # FIXME: _check_param should be overriden in corresponding subclasses
        if self.name in ["KPLS", "KPLSK", "GEKPLS"]:
            d = self.options["n_comp"]
        else:
            d = self.nx

        if len(self.options["theta0"]) != d:
            if len(self.options["theta0"]) == 1:
                self.options["theta0"] *= np.ones(d)
            else:
                raise ValueError(
                    "the number of dim %s should be equal to the length of theta0 %s."
                    % (d, len(self.options["theta0"]))
                )

        if self.supports["training_derivatives"]:
            if not (1 in self.training_points[None]):
                raise Exception(
                    "Derivative values are needed for using the GEKPLS model."
                )

    def _check_F(self, n_samples_F, p):
        """
        This function check the F-parameters of the model.
        """

        if n_samples_F != self.nt:
            raise Exception(
                "Number of rows in F and X do not match. Most "
                "likely something is going wrong with the "
                "regression model."
            )
        if p > n_samples_F:
            raise Exception(
                (
                    "Ordinary least squares problem is undetermined "
                    "n_samples=%d must be greater than the "
                    "regression model size p=%d."
                )
                % (self.nt, p)
            )
