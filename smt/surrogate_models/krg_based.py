"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>

Some functions are copied from gaussian_process submodule (Scikit-learn 0.14)

This package is distributed under New BSD license.
"""
import numpy as np
from scipy import linalg, optimize

from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.kriging_utils import differences
from smt.utils.kriging_utils import constant, linear, quadratic
from smt.utils.kriging_utils import (
    squar_exp,
    abs_exp,
    act_exp,
    standardization,
    cross_distances,
    matern52,
    matern32,
)

from scipy.stats import multivariate_normal as m_norm


# TODO : compute variance derivatives


class KrgBased(SurrogateModel):

    _regression_types = {"constant": constant, "linear": linear, "quadratic": quadratic}

    _correlation_types = {
        "abs_exp": abs_exp,
        "squar_exp": squar_exp,
        "act_exp": act_exp,
        "matern52": matern52,
        "matern32": matern32,
    }

    def _initialize(self):
        super(KrgBased, self)._initialize()
        declare = self.options.declare
        supports = self.supports
        declare(
            "poly",
            "constant",
            values=("constant", "linear", "quadratic"),
            desc="Regression function type",
            types=(str),
        )
        declare(
            "corr",
            "squar_exp",
            values=("abs_exp", "squar_exp", "act_exp", "matern52", "matern32"),
            desc="Correlation function type",
            types=(str),
        )
        declare(
            "theta0", [1e-2], types=(list, np.ndarray), desc="Initial hyperparameters"
        )
        declare(
            "hyper_opt",
            "Cobyla",
            values=("Cobyla", "TNC"),
            desc="Optimiser for hyperparameters optimisation",
            types=(str),
        )
        declare(
            "eval_noise",
            False,
            types=bool,
            values=(True, False),
            desc="noise evaluation flag",
        )
        declare(
            "noise0", 1e-6, types=(float, list), desc="Initial noise hyperparameter"
        )
        self.name = "KrigingBased"
        self.best_iteration_fail = None
        self.nb_ill_matrix = 5
        supports["derivatives"] = True
        supports["variances"] = True

    def _new_train(self):
        self._check_param()

        # Sampling points X and y
        X = self.training_points[None][0][0]
        y = self.training_points[None][0][1]

        # Compute PLS-coefficients (attr of self) and modified X and y (if GEKPLS is used)
        if self.name not in ["Kriging", "MGP"]:
            X, y = self._compute_pls(X.copy(), y.copy())

        # Center and scale X and y
        (
            self.X_norma,
            self.y_norma,
            self.X_offset,
            self.y_mean,
            self.X_scale,
            self.y_std,
        ) = standardization(X, y)

        # Calculate matrix of distances D between samples
        D, self.ij = cross_distances(self.X_norma)
        if np.min(np.sum(np.abs(D), axis=1)) == 0.0:
            print(
                "Warning: multiple x input features have the same value (at least same row twice)."
            )
        ####
        # Regression matrix and parameters
        self.F = self._regression_types[self.options["poly"]](self.X_norma)
        n_samples_F = self.F.shape[0]
        if self.F.ndim > 1:
            p = self.F.shape[1]
        else:
            p = 1
        self._check_F(n_samples_F, p)

        # Optimization
        (
            self.optimal_rlf_value,
            self.optimal_par,
            self.optimal_theta,
        ) = self._optimize_hyperparam(D)
        if self.name in ["MGP"]:
            self._specific_train()
        else:
            if self.options["eval_noise"]:
                self.optimal_theta = self.optimal_theta[:-1]
        # if self.name != "MGP":
        #     del self.y_norma, self.D

    def _train(self):
        """
        Train the model
        """
        # outputs['sol'] = self.sol

        self._new_train()

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
            Q, G
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
        elif self.name in ["MGP"]:
            nugget = 100.0 * nugget
        noise = 0
        tmp_var = theta
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
            # raise e
            return reduced_likelihood_function_value, par

        # Get generalized least squared solution
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
        p = 0
        q = 0
        if self.name in ["MFK", "MFKPLS", "MFKPLSK"]:
            p = self.p
            q = self.q
        sigma2 = (rho ** 2.0).sum(axis=0) / (self.nt - p - q)
        reduced_likelihood_function_value = -(self.nt - p - q) * np.log10(
            sigma2.sum()
        ) - self.nt * np.log10(detR)
        par["sigma2"] = sigma2 * self.y_std ** 2.0
        par["beta"] = beta
        par["gamma"] = linalg.solve_triangular(C.T, rho)
        par["C"] = C
        par["Ft"] = Ft
        par["G"] = G
        par["Q"] = Q

        if self.name in ["MGP"]:
            reduced_likelihood_function_value += self._reduced_log_prior(theta)

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

    def _reduced_likelihood_gradient(self, theta):
        """
        Evaluates the reduced_likelihood_gradient at a set of hyperparameters.

        Arguments
        ---------
        theta : list(n_comp), optional
            - An array containing the autocorrelation parameters at which the
              Gaussian Process model parameters should be determined.

        Returns
        -------
        grad_red : np.ndarray (dim,1)
            Derivative of the reduced_likelihood

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
            Q, G
            QR decomposition of the matrix Ft.
            dr
            List of all the correlation matrix derivative
            tr
            List of all the trace part in the reduce likelihood derivatives
            dmu
            List of all the mean derivatives
            arg
            List of all minus_Cinv_dRdomega_gamma
            dsigma
            List of all sigma derivatives
        """
        red, par = self._reduced_likelihood_function(theta)

        C = par["C"]
        gamma = par["gamma"]
        Q = par["Q"]
        G = par["G"]
        sigma_2 = par["sigma2"]

        nb_theta = len(theta)
        grad_red = np.zeros(nb_theta)

        dr_all = []
        tr_all = []
        dmu_all = []
        arg_all = []
        dsigma_all = []
        dbeta_all = []

        for i_der in range(nb_theta):
            # Compute R derivatives
            dr = self._correlation_types[self.options["corr"]](
                theta, self.D, grad_ind=i_der
            )

            dr_all.append(dr)

            dR = np.zeros((self.nt, self.nt))
            dR[self.ij[:, 0], self.ij[:, 1]] = dr[:, 0]
            dR[self.ij[:, 1], self.ij[:, 0]] = dr[:, 0]

            # Compute beta derivatives
            Cinv_dR_gamma = linalg.solve_triangular(C, np.dot(dR, gamma), lower=True)
            dbeta = -linalg.solve_triangular(G, np.dot(Q.T, Cinv_dR_gamma))
            arg_all.append(Cinv_dR_gamma)

            dbeta_all.append(dbeta)

            # Compute mu derivatives
            dmu = np.dot(self.F, dbeta)
            dmu_all.append(dmu)

            # Compute log(detR) derivatives
            tr_1 = linalg.solve_triangular(C, dR, lower=True)
            tr = linalg.solve_triangular(C.T, tr_1)
            tr_all.append(tr)

            # Compute Sigma2 Derivatives
            dsigma_2 = (
                (1 / self.nt)
                * (
                    -dmu.T.dot(gamma)
                    - gamma.T.dot(dmu)
                    - np.dot(gamma.T, dR.dot(gamma))
                )
                * self.y_std ** 2.0
            )
            dsigma_all.append(dsigma_2)

            # Compute reduced log likelihood derivatives
            grad_red[i_der] = (
                -self.nt / np.log(10) * (dsigma_2 / sigma_2 + np.trace(tr) / self.nt)
            )

        par["dr"] = dr_all
        par["tr"] = tr_all
        par["dmu"] = dmu_all
        par["arg"] = arg_all
        par["dsigma"] = dsigma_all
        par["dbeta_all"] = dbeta_all

        grad_red = np.atleast_2d(grad_red).T

        if self.name in ["MGP"]:
            grad_red += self._reduced_log_prior(theta, grad=True)
        return grad_red, par

    def _reduced_likelihood_hessian(self, theta):
        """
        Evaluates the reduced_likelihood_gradient at a set of hyperparameters.

        Arguments
        ---------
        theta : list(n_comp), optional
            - An array containing the autocorrelation parameters at which the
              Gaussian Process model parameters should be determined.

        Returns
        -------
        hess : np.ndarray
            Hessian values.

        hess_ij: np.ndarray [nb_theta * (nb_theta + 1) / 2, 2]
            - The indices i and j of the vectors in theta associated to the hessian in hess.

        par: dict()
            - A dictionary containing the requested Gaussian Process model
              parameters:

            sigma2
            Gaussian Process variance.
            beta
            Generalized least-squared regression weights for
            Universal Kriging or for Ordinary Kriging.
            gamma
            Gaussian Process weights.
            C
            Cholesky decomposition of the correlation matrix [R].
            Ft
            Solution of the linear equation system : [R] x Ft = F
            Q, G
            QR decomposition of the matrix Ft.
            dr
            List of all the correlation matrix derivative
            tr
            List of all the trace part in the reduce likelihood derivatives
            dmu
            List of all the mean derivatives
            arg
            List of all minus_Cinv_dRdomega_gamma
            dsigma
            List of all sigma derivatives
        """
        dred, par = self._reduced_likelihood_gradient(theta)

        C = par["C"]
        gamma = par["gamma"]
        Q = par["Q"]
        G = par["G"]
        sigma_2 = par["sigma2"]

        nb_theta = len(theta)

        dr_all = par["dr"]
        tr_all = par["tr"]
        dmu_all = par["dmu"]
        arg_all = par["arg"]
        dsigma = par["dsigma"]
        Rinv_dRdomega_gamma_all = []
        Rinv_dmudomega_all = []

        n_val_hess = nb_theta * (nb_theta + 1) // 2
        hess_ij = np.zeros((n_val_hess, 2), dtype=np.int)
        hess = np.zeros((n_val_hess, 1))
        ind_1 = 0

        if self.name in ["MGP"]:
            log_prior = self._reduced_log_prior(theta, hessian=True)

        for omega in range(nb_theta):
            ind_0 = ind_1
            ind_1 = ind_0 + nb_theta - omega
            hess_ij[ind_0:ind_1, 0] = omega
            hess_ij[ind_0:ind_1, 1] = np.arange(omega, nb_theta)

            dRdomega = np.zeros((self.nt, self.nt))
            dRdomega[self.ij[:, 0], self.ij[:, 1]] = dr_all[omega][:, 0]
            dRdomega[self.ij[:, 1], self.ij[:, 0]] = dr_all[omega][:, 0]

            dmudomega = dmu_all[omega]
            Cinv_dmudomega = linalg.solve_triangular(C, dmudomega, lower=True)
            Rinv_dmudomega = linalg.solve_triangular(C.T, Cinv_dmudomega)
            Rinv_dmudomega_all.append(Rinv_dmudomega)
            Rinv_dRdomega_gamma = linalg.solve_triangular(C.T, arg_all[omega])
            Rinv_dRdomega_gamma_all.append(Rinv_dRdomega_gamma)

            for i, eta in enumerate(hess_ij[ind_0:ind_1, 1]):
                dRdeta = np.zeros((self.nt, self.nt))
                dRdeta[self.ij[:, 0], self.ij[:, 1]] = dr_all[eta][:, 0]
                dRdeta[self.ij[:, 1], self.ij[:, 0]] = dr_all[eta][:, 0]

                dr_eta_omega = self._correlation_types[self.options["corr"]](
                    theta, self.D, grad_ind=omega, hess_ind=eta
                )
                dRdetadomega = np.zeros((self.nt, self.nt))
                dRdetadomega[self.ij[:, 0], self.ij[:, 1]] = dr_eta_omega[:, 0]
                dRdetadomega[self.ij[:, 1], self.ij[:, 0]] = dr_eta_omega[:, 0]

                # Compute beta second derivatives
                dRdeta_Rinv_dmudomega = np.dot(dRdeta, Rinv_dmudomega)

                dmudeta = dmu_all[eta]
                Cinv_dmudeta = linalg.solve_triangular(C, dmudeta, lower=True)
                Rinv_dmudeta = linalg.solve_triangular(C.T, Cinv_dmudeta)
                dRdomega_Rinv_dmudeta = np.dot(dRdomega, Rinv_dmudeta)

                dRdeta_Rinv_dRdomega_gamma = np.dot(dRdeta, Rinv_dRdomega_gamma)

                Rinv_dRdeta_gamma = linalg.solve_triangular(C.T, arg_all[eta])
                dRdomega_Rinv_dRdeta_gamma = np.dot(dRdomega, Rinv_dRdeta_gamma)

                dRdetadomega_gamma = np.dot(dRdetadomega, gamma)

                beta_sum = (
                    dRdeta_Rinv_dmudomega
                    + dRdomega_Rinv_dmudeta
                    + dRdeta_Rinv_dRdomega_gamma
                    + dRdomega_Rinv_dRdeta_gamma
                    - dRdetadomega_gamma
                )

                Qt_Cinv_beta_sum = np.dot(
                    Q.T, linalg.solve_triangular(C, beta_sum, lower=True)
                )
                dbetadetadomega = linalg.solve_triangular(G, Qt_Cinv_beta_sum)

                # Compute mu second derivatives
                dmudetadomega = np.dot(self.F, dbetadetadomega)

                # Compute sigma2 second derivatives
                sigma_arg_1 = (
                    -np.dot(dmudetadomega.T, gamma)
                    + np.dot(dmudomega.T, Rinv_dRdeta_gamma)
                    + np.dot(dmudeta.T, Rinv_dRdomega_gamma)
                )

                sigma_arg_2 = (
                    -np.dot(gamma.T, dmudetadomega)
                    + np.dot(gamma.T, dRdeta_Rinv_dmudomega)
                    + np.dot(gamma.T, dRdomega_Rinv_dmudeta)
                )

                sigma_arg_3 = np.dot(dmudeta.T, Rinv_dmudomega) + np.dot(
                    dmudomega.T, Rinv_dmudeta
                )

                sigma_arg_4_in = (
                    -dRdetadomega_gamma
                    + dRdeta_Rinv_dRdomega_gamma
                    + dRdomega_Rinv_dRdeta_gamma
                )
                sigma_arg_4 = np.dot(gamma.T, sigma_arg_4_in)

                dsigma2detadomega = (
                    (1 / self.nt)
                    * (sigma_arg_1 + sigma_arg_2 + sigma_arg_3 + sigma_arg_4)
                    * self.y_std ** 2.0
                )

                # Compute Hessian
                dreddetadomega_tr_1 = np.trace(np.dot(tr_all[eta], tr_all[omega]))

                dreddetadomega_tr_2 = np.trace(
                    linalg.solve_triangular(
                        C.T, linalg.solve_triangular(C, dRdetadomega, lower=True)
                    )
                )

                dreddetadomega_arg1 = (self.nt / sigma_2) * (
                    dsigma2detadomega - (1 / sigma_2) * dsigma[omega] * dsigma[eta]
                )
                dreddetadomega = (
                    -(dreddetadomega_arg1 - dreddetadomega_tr_1 + dreddetadomega_tr_2)
                    / self.nt
                )

                hess[ind_0 + i, 0] = self.nt / np.log(10) * dreddetadomega

                if self.name in ["MGP"] and eta == omega:
                    hess[ind_0 + i, 0] += log_prior[eta]
            par["Rinv_dR_gamma"] = Rinv_dRdomega_gamma_all
            par["Rinv_dmu"] = Rinv_dmudomega_all
        return hess, hess_ij, par

    def _predict_values(self, x):
        """
        Evaluates the model at a set of points.

        Parameters
        ----------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        y : np.ndarray
            Evaluation point output variable values
        """
        # Initialization
        n_eval, n_features_x = x.shape
        x = (x - self.X_offset) / self.X_scale
        # Get pairwise componentwise L1-distances to the input training set
        dx = differences(x, Y=self.X_norma.copy())
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

        Parameters
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
        # Initialization
        n_eval, n_features_x = x.shape
        x = (x - self.X_offset) / self.X_scale
        # Get pairwise componentwise L1-distances to the input training set
        dx = differences(x, Y=self.X_norma.copy())
        d = self._componentwise_distance(dx)
        # Compute the correlation function
        r = self._correlation_types[self.options["corr"]](
            self.optimal_theta, d
        ).reshape(n_eval, self.nt)

        if self.options["corr"] != "squar_exp":
            raise ValueError(
                "The derivative is only available for squared exponential kernel"
            )
        if self.options["poly"] == "constant":
            df = np.zeros((1, self.nx))
        elif self.options["poly"] == "linear":
            df = np.zeros((self.nx + 1, self.nx))
            df[1:, :] = np.eye(self.nx)
        else:
            raise ValueError(
                "The derivative is only available for ordinary kriging or "
                + "universal kriging using a linear trend"
            )

        # Beta and gamma = R^-1(y-FBeta)
        beta = self.optimal_par["beta"]
        gamma = self.optimal_par["gamma"]
        df_dx = np.dot(df.T, beta)
        d_dx = x[:, kx].reshape((n_eval, 1)) - self.X_norma[:, kx].reshape((1, self.nt))
        if self.name != "Kriging" and "KPLSK" not in self.name:
            theta = np.sum(self.optimal_theta * self.coeff_pls ** 2, axis=1)
        else:
            theta = self.optimal_theta
        y = (
            (df_dx[kx] - 2 * theta[kx] * np.dot(d_dx * r, gamma))
            * self.y_std
            / self.X_scale[kx]
        )
        return y

    def _predict_variances(self, x):
        """
        Provide uncertainty of the model at a set of points

        Parameters
        ----------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        MSE : np.ndarray
            Evaluation point output variable MSE
        """
        # Initialization
        n_eval, n_features_x = x.shape
        x = (x - self.X_offset) / self.X_scale
        # Get pairwise componentwise L1-distances to the input training set
        dx = differences(x, Y=self.X_norma.copy())
        d = self._componentwise_distance(dx)

        # Compute the correlation function
        r = self._correlation_types[self.options["corr"]](
            self.optimal_theta, d
        ).reshape(n_eval, self.nt)

        C = self.optimal_par["C"]
        rt = linalg.solve_triangular(C, r.T, lower=True)

        u = linalg.solve_triangular(
            self.optimal_par["G"].T,
            np.dot(self.optimal_par["Ft"].T, rt)
            - self._regression_types[self.options["poly"]](x).T,
        )

        A = self.optimal_par["sigma2"]
        B = 1.0 - (rt ** 2.0).sum(axis=0) + (u ** 2.0).sum(axis=0)
        MSE = np.einsum("i,j -> ji", A, B)

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
        if self.name in ["MGP"]:

            def minus_reduced_likelihood_function(theta):
                res = -self._reduced_likelihood_function(theta)[0]
                return res

            def grad_minus_reduced_likelihood_function(theta):
                grad = -self._reduced_likelihood_gradient(theta)[0]
                return grad

        else:

            def minus_reduced_likelihood_function(log10t):
                return -self._reduced_likelihood_function(theta=10.0 ** log10t)[0]

            def grad_minus_reduced_likelihood_function(log10t):
                log10t_2d = np.atleast_2d(log10t).T
                res = (
                    -np.log(10.0)
                    * (10.0 ** log10t_2d)
                    * (self._reduced_likelihood_gradient(10.0 ** log10t_2d)[0])
                )
                return res

        limit, _rhobeg = 10 * len(self.options["theta0"]), 0.5
        exit_function = False
        if "KPLSK" in self.name:
            n_iter = 1
        else:
            n_iter = 0

        for ii in range(n_iter, -1, -1):
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

            for i in range(len(self.options["theta0"])):
                # In practice, in 1D and for X in [0,1], theta^{-2} in [1e-2,infty),
                # i.e. theta in (0,1e1], is a good choice to avoid overfitting.
                # By standardising X in R, X_norm = (X-X_mean)/X_std, then
                # X_norm in [-1,1] if considering one std intervals. This leads
                # to theta in (0,2e1]
                theta_max = 2e1
                if self.name in ["MGP"]:
                    constraints.append(lambda theta, i=i: theta[i] + theta_max)
                    constraints.append(lambda theta, i=i: theta_max - theta[i])
                    bounds_hyp.append((-theta_max, theta_max))
                else:
                    constraints.append(lambda log10t, i=i: log10t[i] - np.log10(1e-6))
                    constraints.append(
                        lambda log10t, i=i: np.log10(theta_max) - log10t[i]
                    )
                    bounds_hyp.append((np.log10(1e-6), np.log10(theta_max)))

            if self.name in ["MGP"]:
                theta0_rand = m_norm.rvs(
                    self.options["prior"]["mean"] * len(self.options["theta0"]),
                    self.options["prior"]["var"],
                    1,
                )
                theta0 = self.options["theta0"]
            else:
                theta0_rand = np.random.rand(len(self.options["theta0"]))
                theta0_rand = theta0_rand * 8.0 - 6.0
                theta0 = np.log10(self.options["theta0"])

            self.D = self._componentwise_distance(D, opt=ii)

            # Initialization
            k, incr, stop, best_optimal_rlf_value, max_retry = 0, 0, 1, -1e20, 10
            while k < stop:
                # Use specified starting point as first guess
                if self.options["eval_noise"]:
                    theta0 = np.concatenate(
                        [theta0, np.log10(np.array([self.options["noise0"]]))]
                    )
                    theta0_rand = np.concatenate(
                        [theta0_rand, np.log10(np.array([self.options["noise0"]]))]
                    )

                    constraints.append(lambda log10t: log10t[-1] + 16)
                    constraints.append(lambda log10t: 10 - log10t[-1])

                    bounds_hyp.append((10, 16))
                try:

                    if self.options["hyper_opt"] == "Cobyla":
                        optimal_theta_res = optimize.minimize(
                            minus_reduced_likelihood_function,
                            theta0,
                            constraints=[
                                {"fun": con, "type": "ineq"} for con in constraints
                            ],
                            method="COBYLA",
                            options={"rhobeg": _rhobeg, "tol": 1e-4, "maxiter": limit},
                        )

                        optimal_theta_res_2 = optimal_theta_res

                        # optimal_theta_res_2 = optimal_theta = optimize.minimize(
                        #     minus_reduced_likelihood_function,
                        #     theta0_rand,
                        #     constraints=[
                        #         {"fun": con, "type": "ineq"} for con in constraints
                        #     ],
                        #     method="COBYLA",
                        #     options={"rhobeg": _rhobeg, "tol": 1e-4, "maxiter": limit},
                        # )

                    elif self.options["hyper_opt"] == "TNC":

                        optimal_theta_res = optimize.minimize(
                            minus_reduced_likelihood_function,
                            theta0,
                            method="TNC",
                            jac=grad_minus_reduced_likelihood_function,
                            bounds=bounds_hyp,
                            options={"maxiter": 100},
                        )

                        optimal_theta_res_2 = optimize.minimize(
                            minus_reduced_likelihood_function,
                            theta0_rand,
                            method="TNC",
                            jac=grad_minus_reduced_likelihood_function,
                            bounds=bounds_hyp,
                            options={"maxiter": 100},
                        )

                    if optimal_theta_res["fun"] > optimal_theta_res_2["fun"]:
                        optimal_theta_res = optimal_theta_res_2

                    optimal_theta = optimal_theta_res["x"]

                    if self.name not in ["MGP"]:
                        optimal_theta = 10 ** optimal_theta

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
                            if optimal_rlf_value >= self.best_iteration_fail:
                                if optimal_rlf_value > best_optimal_rlf_value:
                                    best_optimal_rlf_value = optimal_rlf_value
                                    best_optimal_par = optimal_par
                                    best_optimal_theta = optimal_theta

                            else:
                                if self.best_iteration_fail > best_optimal_rlf_value:
                                    best_optimal_theta = self._thetaMemory.copy()
                                    (
                                        best_optimal_rlf_value,
                                        best_optimal_par,
                                    ) = self._reduced_likelihood_function(
                                        theta=best_optimal_theta
                                    )
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
                    self.options["theta0"] = (theta * self.coeff_pls ** 2).sum(1)
                else:
                    self.options["theta0"] = (theta * np.abs(self.coeff_pls)).sum(1)

                self.options["n_comp"] = int(self.nx)
                limit = 10 * self.options["n_comp"]
                self.best_iteration_fail = None
                exit_function = True

        return best_optimal_rlf_value, best_optimal_par, best_optimal_theta

    def _check_param(self):
        """
        This function checks some parameters of the model.
        """

        # FIXME: _check_param should be overriden in corresponding subclasses
        if self.name in ["KPLS", "KPLSK", "GEKPLS", "MFKPLS", "MFKPLSK"]:

            d = self.options["n_comp"]
        elif self.name in ["MGP"]:
            d = self.options["n_comp"] * self.nx
        else:
            d = self.nx

        if self.name in ["MGP"]:
            if self.options["corr"] != "act_exp":
                raise ValueError("MGP must be used with act_exp correlation function")
            if self.options["hyper_opt"] != "TNC":
                raise ValueError("MGP must be used with TNC hyperparameters optimizer")
        else:
            if self.options["corr"] == "act_exp":
                raise ValueError("act_exp correlation function must be used with MGP")

        if (
            self.name in ["MFK"]
            and isinstance(self.options["theta0"], np.ndarray)
            and len(self.options["theta0"].shape) > 1
        ):
            if self.options["theta0"].shape != (self.nlvl, d):
                raise ValueError(
                    "the number of dim %s should coincide to the dimensions of theta0 %s."
                    % ((d, self.nlvl), self.options["theta0"].shape)
                )
        elif self.name in ["MFK"] and len(self.options["theta0"]) == self.nlvl:
            pass
        else:
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
        if self.name in ["KPLS"]:
            if self.options["corr"] not in ["squar_exp", "abs_exp"]:
                raise ValueError(
                    "KPLS only works with a squared exponential or an absolute exponential kernel"
                )

        if self.name in ["KPLSK"]:
            if self.options["corr"] not in ["squar_exp"]:
                raise ValueError(
                    "KPLSK only works with a squared exponential kernel (until we prove the contrary)"
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
