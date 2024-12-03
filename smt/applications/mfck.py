# -*- coding: utf-8 -*-
"""
Created on Sat May 04 10:10:12 2024

@author: Mauricio Castano Aguirre <mauricio.castano_aguirre@onera.fr>
Multi-Fidelity co-Kriging model construction for non-nested experimental
design sets.
-------
[1] Loic Le Gratiet (2013). Multi-fidelity Gaussian process modelling
[Doctoral Thesis, Université Paris-Sud].
[2] Edwin V. Bonilla, Kian Ming A. Chai, and Christopher K. I. Williams
(2007). Multi-task Gaussian Process prediction. In International
Conference on Neural Information Processing Systems.
"""

# import warnings
import numpy as np
from scipy.linalg import solve_triangular
from scipy import optimize
from smt.sampling_methods import LHS
from smt.utils.kriging import differences, componentwise_distance
from smt.surrogate_models.krg_based import KrgBased


class MFCK(KrgBased):
    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        self.name = "MFCK"

        declare(
            "rho0",
            1.0,
            types=(float),
            desc="Initial rho for the autoregressive model , \
                  (scalar factor between two consecutive fidelities, \
                    e.g., Y_HF = (Rho) * Y_LF + Gamma",
        )
        declare(
            "rho_bounds",
            [-5.0, 5.0],
            types=(list, np.ndarray),
            desc="Bounds for the rho parameter used in the autoregressive model",
        )
        declare(
            "sigma0",
            1.0,
            types=(float),
            desc="Initial variance parameter",
        )
        declare(
            "sigma_bounds",
            [1e-1, 100],
            types=(list, np.ndarray),
            desc="Bounds for the variance parameter",
        )

        self.options["nugget"] = (
            1e-9  # Incresing the nugget for numerical stability reasons
        )
        self.options["hyper_opt"] = (
            "Cobyla"  # MFCK doesn't support gradient-based optimizers
        )

    def train(self):
        """
        Overrides MFK implementation
        Trains the Multi-Fidelity co-Kriging model
        Returns
        -------
        None.
        """
        xt = []
        yt = []
        i = 0
        while self.training_points.get(i, None) is not None:
            xt.append(self.training_points[i][0][0])
            yt.append(self.training_points[i][0][1])
            i = i + 1
        xt.append(self.training_points[None][0][0])
        yt.append(self.training_points[None][0][1])
        self.lvl = i + 1
        self.X = xt
        self.y = np.vstack(yt)
        self._check_param()
        self.nx = 1  # Forcing this in order to consider isotropic kernels (i.e., only one lengthscale)
        if self.lvl == 1:
            # For a single level, initialize theta_ini, lower_bounds, and
            # upper_bounds with consistent shapes
            theta_ini = np.hstack(
                (self.options["sigma0"], self.options["theta0"][0])
            )  # Kernel variance + theta0
            lower_bounds = np.hstack(
                (self.options["sigma_bounds"][0], self.options["theta_bounds"][0])
            )
            upper_bounds = np.hstack(
                (self.options["sigma_bounds"][1], self.options["theta_bounds"][1])
            )
            theta_ini = np.log10(theta_ini)
            lower_bounds = np.log10(lower_bounds)
            upper_bounds = np.log10(upper_bounds)
            x_opt = theta_ini
        else:
            for lvl in range(self.lvl):
                if lvl == 0:
                    # Initialize theta_ini for level 0
                    theta_ini = np.hstack(
                        (self.options["sigma0"], self.options["theta0"][0])
                    )  # Variance + initial theta values
                    lower_bounds = np.hstack(
                        (
                            self.options["sigma_bounds"][0],
                            np.full(self.nx, self.options["theta_bounds"][0]),
                        )
                    )
                    upper_bounds = np.hstack(
                        (
                            self.options["sigma_bounds"][1],
                            np.full(self.nx, self.options["theta_bounds"][1]),
                        )
                    )
                    # Apply log10 to theta_ini and bounds
                    nb_params = len(self.options["theta0"])
                    theta_ini[: nb_params + 1] = np.log10(theta_ini[: nb_params + 1])
                    lower_bounds[: nb_params + 1] = np.log10(
                        lower_bounds[: nb_params + 1]
                    )
                    upper_bounds[: nb_params + 1] = np.log10(
                        upper_bounds[: nb_params + 1]
                    )

                elif lvl > 0:
                    # For additional levels, append to theta_ini, lower_bounds, and upper_bounds
                    thetat = np.hstack(
                        (self.options["sigma0"], self.options["theta0"][0])
                    )
                    lower_boundst = np.hstack(
                        (
                            self.options["sigma_bounds"][0],
                            np.full(self.nx, self.options["theta_bounds"][0]),
                        )
                    )
                    upper_boundst = np.hstack(
                        (
                            self.options["sigma_bounds"][1],
                            np.full(self.nx, self.options["theta_bounds"][1]),
                        )
                    )
                    # Apply log10 to the newly added values
                    thetat = np.log10(thetat)
                    lower_boundst = np.log10(lower_boundst)
                    upper_boundst = np.log10(upper_boundst)
                    # Append to theta_ini, lower_bounds, and upper_bounds
                    theta_ini = np.hstack([theta_ini, thetat, self.options["rho0"]])
                    lower_bounds = np.hstack([lower_bounds, lower_boundst])
                    upper_bounds = np.hstack([upper_bounds, upper_boundst])
                    # Finally, append the rho bounds
                    lower_bounds = np.hstack(
                        [lower_bounds, self.options["rho_bounds"][0]]
                    )
                    upper_bounds = np.hstack(
                        [upper_bounds, self.options["rho_bounds"][1]]
                    )

        theta_ini = theta_ini[:].T
        x_opt = theta_ini

        if self.options["hyper_opt"] == "Cobyla":
            if self.options["n_start"] > 1:
                sampling = LHS(
                    xlimits=np.stack((lower_bounds, upper_bounds), axis=1),
                    criterion="ese",
                    random_state=0,
                )
                theta_lhs_loops = sampling(self.options["n_start"])
                theta0 = np.vstack((theta_ini, theta_lhs_loops))

            constraints = []

            for i in range(len(theta_ini)):
                constraints.append(lambda theta0, i=i: theta0[i] - lower_bounds[i])
                constraints.append(lambda theta0, i=i: upper_bounds[i] - theta0[i])

            for j in range(self.options["n_start"]):
                optimal_theta_res_loop = optimize.minimize(
                    self.neg_log_likelihood_scipy,
                    theta0[j, :],
                    method="COBYLA",
                    constraints=[{"fun": con, "type": "ineq"} for con in constraints],
                    options={
                        "rhobeg": 0.1,
                        "tol": 1e-4,
                        "maxiter": 200,
                    },
                )
                x_opt_iter = optimal_theta_res_loop.x

                if j == 0:
                    x_opt = x_opt_iter
                    nll = optimal_theta_res_loop["fun"]
                else:
                    if optimal_theta_res_loop["fun"] < nll:
                        x_opt = x_opt_iter
                        nll = optimal_theta_res_loop["fun"]

        elif self.options["hyper_opt"] == "Cobyla-nlopt":
            try:
                import nlopt
            except ImportError:
                print("nlopt library is not installed or available on this system")

            opt = nlopt.opt(nlopt.LN_COBYLA, theta_ini.shape[0])
            opt.set_lower_bounds(lower_bounds)  # Lower bounds for each dimension
            opt.set_upper_bounds(upper_bounds)  # Upper bounds for each dimension
            opt.set_min_objective(self.neg_log_likelihood_nlopt)
            opt.set_maxeval(5000)
            opt.set_xtol_rel(1e-6)
            x0 = np.copy(theta_ini)
            x_opt = opt.optimize(x0)
        else:
            raise ValueError(
                f"The optimizer {self.options['hyper_opt']} is not available"
            )

        x_opt[0:2] = 10 ** (x_opt[0:2])
        x_opt[2::3] = 10 ** (x_opt[2::3])
        x_opt[3::3] = 10 ** (x_opt[3::3])
        self.optimal_theta = x_opt

    def eta(self, j, jp, rho):
        """Compute eta_{j,l} based on the given rho values."""
        if j < jp:
            return np.prod(rho[j:jp])  # Product of rho[j+1] to rho[l]
        elif j == jp:
            return 1
        else:
            raise ValueError(
                f"The iterative variable j={j} cannot be greater than j'={jp}"
            )

    # Covariance between y_l(x) and y_l'(x')
    def compute_cross_K(self, x, xp, L, Lp, param):
        """
        Calculation Cov(y_l(x), y_{l'}(x')) using the autoregressive formulation.
        Parmeters:
        - x: First input for the covariannce (np.ndarray)
        - xp: Second input for the covariannce (np.ndarray)
        - L: Level index of the first output (scalar)
        - Lp: Level index of the second output (scalar)
        - param: Set of Hyper-parameters (vector)
        Returns:
        - Covariance matrix cov(y_l(x), y_{l'}(x')) (np.ndarray)
        """
        cov_value = 0.0

        param0 = param[0:2]
        sigmas_gamma = param[2::3]
        ls_gamma = param[3::3]
        rho_values = param[4::3]

        # Sum of j=0 until l_^prime
        for j in range(Lp + 1):
            eta_j_l = self.eta(j, L, rho_values)
            eta_j_lp = self.eta(j, Lp, rho_values)

            if j == 0:
                # Cov(γ_j(x), γ_j(x')) using the kernel for K_00
                cov_gamma_j = self._compute_K(x, xp, param0)
            else:
                # Cov(γ_j(x), γ_j(x')) using the kernel
                cov_gamma_j = self._compute_K(
                    x, xp, [sigmas_gamma[j - 1], ls_gamma[j - 1]]
                )
            # Add to the value of the covariance
            cov_value += eta_j_l * eta_j_lp * cov_gamma_j

        return cov_value

    def predict_all_levels(self, x):
        """
        Generalized prediction function for the multi-fidelity co-Kriging
        Parameters
        ----------
        x : np.ndarray
            Array with the inputs for make the prediction.
        Returns
        -------
        means : (list, np.array)
            Returns the conditional means per level.
        covariances: (list, np.array)
            Returns the conditional covariance matrixes per level.
        """
        self.K = self.compute_blockwise_K(self.optimal_theta)
        means = []
        covariances = []
        if self.lvl == 1:
            k_XX = self._compute_K(self.X[0], self.X[0], self.optimal_theta[0:2])
            k_xX = self._compute_K(x, self.X[0], self.optimal_theta[0:2])
            k_xx = self._compute_K(x, x, self.optimal_theta[0:2])
            # To be adapted using the Cholesky decomposition
            k_XX_inv = np.linalg.inv(
                k_XX + self.options["nugget"] * np.eye(k_XX.shape[0])
            )
            means.append(np.dot(k_xX, np.matmul(k_XX_inv, self.y)))
            covariances.append(
                k_xx - np.matmul(k_xX, np.matmul(k_XX_inv, k_xX.transpose()))
            )

        else:
            L = np.linalg.cholesky(
                self.K + self.options["nugget"] * np.eye(self.K.shape[0])
            )
            k_xX = []
            for ind in range(self.lvl):
                k_xx = self.compute_cross_K(x, x, ind, ind, self.optimal_theta)
                for j in range(self.lvl):
                    if ind >= j:
                        k_xX.append(
                            self.compute_cross_K(
                                self.X[j], x, ind, j, self.optimal_theta
                            )
                        )
                    else:
                        k_xX.append(
                            self.compute_cross_K(
                                self.X[j], x, j, ind, self.optimal_theta
                            )
                        )

                beta1 = solve_triangular(L, np.vstack(k_xX), lower=True)
                alpha1 = solve_triangular(L, self.y, lower=True)
                means.append(np.dot(beta1.T, alpha1))
                covariances.append(k_xx - np.dot(beta1.T, beta1))
                k_xX.clear()

        return means, covariances

    def _predict(self, x):
        """
        Prediction function for the highest fidelity level
        Parameters
        ----------
        x : array
            Array with the inputs for make the prediction.
        Returns
        -------
        mean : np.array
            Returns the conditional means per level.
        covariance: np.ndarray
            Returns the conditional covariance matrixes per level.
        """
        means, covariances = self.predict_all_levels(x)

        return means[self.lvl - 1], covariances[self.lvl - 1]

    def neg_log_likelihood(self, param, grad=None):
        if self.lvl == 1:
            self.K = self._compute_K(self.X[0], self.X[0], param[0:2])
        else:
            self.K = self.compute_blockwise_K(param)

        L = np.linalg.cholesky(
            self.K + self.options["nugget"] * np.eye(self.K.shape[0])
        )
        beta = solve_triangular(L, self.y, lower=True)
        NMLL = 1 / 2 * (2 * np.sum(np.log(np.diag(L))) + np.dot(beta.T, beta))
        (nmll,) = NMLL[0]
        return nmll

    def neg_log_likelihood_scipy(self, param):
        """
        Likelihood for Cobyla-scipy (SMT) optimizer
        """
        param = np.array(param, copy=True)
        param[0:2] = 10 ** (param[0:2])
        param[2::3] = 10 ** (param[2::3])
        param[3::3] = 10 ** (param[3::3])
        return self.neg_log_likelihood(param)

    def neg_log_likelihood_nlopt(self, param, grad=None):
        """
        Likelihood for nlopt optimizers
        """
        param = np.array(param, copy=True)
        param[0:2] = 10 ** (param[0:2])
        param[2::3] = 10 ** (param[2::3])
        param[3::3] = 10 ** (param[3::3])
        return self.neg_log_likelihood(param, grad)

    def compute_blockwise_K(self, param):
        K_block = {}
        n = self.y.shape[0]
        for jp in range(self.lvl):
            for j in range(self.lvl):
                if jp >= j:
                    K_block[str(jp) + str(j)] = self.compute_cross_K(
                        self.X[j], self.X[jp], jp, j, param
                    )

        K = np.zeros((n, n))
        row_init, col_init = 0, 0
        for j in range(self.lvl):
            col_init = row_init
            for jp in range(j, self.lvl):
                r, c = K_block[str(jp) + str(j)].shape
                K[row_init : row_init + r, col_init : col_init + c] = K_block[
                    str(jp) + str(j)
                ]
                if j != jp:
                    K[col_init : col_init + c, row_init : row_init + r] = K_block[
                        str(jp) + str(j)
                    ].T
                col_init += c
            row_init += r
        return K

    def _compute_K(self, A: np.ndarray, B: np.ndarray, param):
        """
        Compute the covariance matrix K between A and B
            Modified for MFCK initial test (Same theta for each dimmension)
        """
        # Compute pairwise componentwise L1-distances between A and B
        dx = differences(A, B)
        d = componentwise_distance(
            dx,
            self.options["corr"],
            self.X[0].shape[1],
            power=self.options["pow_exp_power"],
        )
        self.corr.theta = np.full(self.X[0].shape[1], param[1])
        r = self.corr(d)
        R = r.reshape(A.shape[0], B.shape[0])
        K = param[0] * R
        return K
