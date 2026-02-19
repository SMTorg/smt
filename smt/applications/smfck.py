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
from scipy import optimize
from smt.sampling_methods import LHS
from smt.utils.kriging import differences, componentwise_distance
from smt.surrogate_models.krg_based import KrgBased
from smt.utils.misc import standardization
from scipy.cluster.vq import kmeans


class SMFCK(KrgBased):
    "This class should inherit from the class MFCK"

    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        self.name = "SMFCK"

        declare(
            "eval_noise",
            False,
            types=bool,
            values=(True, False),
            desc="If True, the model evaluates noise variance, can be homoscedastic or heteroscedastic",
        )
        declare(
            "use_het_noise",
            False,
            types=bool,
            values=(True, False),
            desc="If True, the model considers Heteroscedastic noise, array with the same size of y(x) is expected",
        )
        declare(
            "predict_with_noise",
            False,
            types=bool,
            values=(True, False),
            desc="if use_het_noise is true, then the prediction of the noise variance over the test set will given",
        )

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
            [1e-6, 100],
            types=(list, np.ndarray),
            desc="Bounds for the variance parameter",
        )
        declare(
            "lambda",
            0.0,
            types=(float),
            desc="Regularization parameter",
        )
        declare(
            "hyper_opt",
            "Cobyla",
            values=("Cobyla", "Cobyla-nlopt"),
            desc="Optimiser for hyperparameters optimisation",
        )
        declare(
            "n_inducing",
            [6, 5],
            types=(list, np.ndarray),
            desc="Number of inducing points per fidelity level",
        )
        declare(
            "method",
            "FITC",
            values=("FITC"),
            desc="Methods available for Sparse Multi-fidelity",
            types=(str),
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
            "inducing_method",
            "kmeans",
            types=str,
            values=["random", "kmeans"],
            desc="The chosen method to induce points",
        )
        self.options.declare(
            "seed",
            default=0,
            types=(type(None), int),
            desc="seed number which controls random draws",
        )

        self.options["hyper_opt"] = (
            "Cobyla-nlopt"  # MFCK doesn't support gradient-based optimizers
        )
        self.woodbury_data = {"vec": None, "inv": None}

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
        zt = []
        i = 0
        while self.training_points.get(i, None) is not None:
            xt.append(self.training_points[i][0][0])
            yt.append(self.training_points[i][0][1])

            if self.options["inducing_method"] == "random":
                idx = np.random.permutation(self.nt)[: self.options["n_inducing"][i]]
                zt.append(xt[idx])
            elif self.options["inducing_method"] == "kmeans":
                zt.append(
                    kmeans(
                        self.training_points[i][0][0],
                        self.options["n_inducing"][i],
                        rng=self.options["seed"],
                    )[0]
                )
            i = i + 1
        xt.append(self.training_points[None][0][0])
        yt.append(self.training_points[None][0][1])

        if self.options["inducing_method"] == "random":
            idx = np.random.permutation(self.nt)[: self.options["n_inducing"][i]]
            zt.append(xt[idx])
        elif self.options["inducing_method"] == "kmeans":
            seed = None
            if self.options["seed"] is not None:
                seed = self.options["seed"] + 1
            zt.append(
                kmeans(
                    self.training_points[None][0][0],
                    self.options["n_inducing"][i],
                    rng=seed,
                )[0]
            )
        # zt.append(kmeans(self.training_points[None][0][0],self.options["n_inducing"][i])[0])
        self.lvl = i + 1
        self.X = xt
        self.Z = zt

        if np.shape(self.options["n_inducing"])[0] == self.lvl:
            self.y = np.vstack(yt)
            self._check_param()

            (
                _,
                _,
                self.X_offset,
                self.y_mean,
                self.X_scale,
                self.y_std,
            ) = standardization(np.concatenate(xt, axis=0), np.concatenate(yt, axis=0))

            self.X_norma_all = [(x - self.X_offset) / self.X_scale for x in xt]
            self.Z_norma_all = [(x - self.X_offset) / self.X_scale for x in zt]
            self.y_norma_all = np.vstack([(f - self.y_mean) / self.y_std for f in yt])

        else:
            raise ValueError(
                f"n_inducing {self.options['n_inducing']} don't correspond to the fidelities"
            )

        if self.lvl == 1:
            # For a single level, initialize theta_ini, lower_bounds, and
            # upper_bounds with consistent shapes
            theta_ini = np.hstack(
                (self.options["sigma0"], self.options["theta0"])
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
            lower_bounds[: nb_params + 1] = np.log10(lower_bounds[: nb_params + 1])
            upper_bounds[: nb_params + 1] = np.log10(upper_bounds[: nb_params + 1])
        else:
            for lvl in range(self.lvl):
                if lvl == 0:
                    # Initialize theta_ini for level 0
                    theta_ini = np.hstack(
                        (self.options["sigma0"], self.options["theta0"])
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
                    thetat = np.hstack((self.options["sigma0"], self.options["theta0"]))
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

        if self.options["eval_noise"]:
            if not self.options["use_het_noise"]:
                theta_ini = np.hstack(
                    [theta_ini, np.full(self.lvl, self.options["noise0"][0])]
                )

                lower_bounds = np.hstack(
                    [lower_bounds, np.full(self.lvl, self.options["noise_bounds"][0])]
                )

                upper_bounds = np.hstack(
                    [upper_bounds, np.full(self.lvl, self.options["noise_bounds"][1])]
                )

        theta_ini[-self.lvl : :] = np.log10(theta_ini[-self.lvl : :])
        upper_bounds[-self.lvl : :] = np.log10(upper_bounds[-self.lvl : :])
        lower_bounds[-self.lvl : :] = np.log10(lower_bounds[-self.lvl : :])

        theta_ini = theta_ini[:].T
        x_opt = theta_ini

        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds

        if self.options["hyper_opt"] == "Cobyla":
            if self.options["n_start"] > 1:
                sampling = LHS(
                    xlimits=np.stack((lower_bounds, upper_bounds), axis=1),
                    criterion="ese",
                    seed=self.options["seed"],
                )
                theta_lhs_loops = sampling(self.options["n_start"])
                theta0 = np.vstack((theta_ini, theta_lhs_loops))
            else:
                theta0 = np.vstack((theta_ini, theta_ini))
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
                        "rhobeg": 0.5,
                        "tol": 1e-6,
                        "maxiter": 50,
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
            opt.set_maxeval(80)
            opt.set_xtol_rel(1e-6)
            x0 = np.copy(theta_ini)
            x_opt = opt.optimize(x0)
        else:
            raise ValueError(
                f"The optimizer {self.options['hyper_opt']} is not available"
            )

        if not self.options["eval_noise"] or self.options["use_het_noise"]:
            x_opt1 = np.array(x_opt, copy=True)
        else:
            x_opt1 = np.array(x_opt[: -self.lvl], copy=True)

        x_opt1[0] = 10 ** (x_opt1[0])  # Apply 10** to Sigma 0
        x_opt1[1 : self.nx + 1] = (
            10 ** (x_opt1[1 : self.nx + 1])
        )  # Apply 10** to length scales 0

        x_opt1[self.nx + 1 :: self.nx + 2] = (
            10 ** (x_opt1[self.nx + 1 :: self.nx + 2])
        )  # Apply 10** to sigmas gamma

        for i in np.arange(self.nx + 2, x_opt1.shape[0] - 1, self.nx + 2):
            x_opt1[i : i + self.nx] = 10 ** x_opt1[i : i + self.nx]

        x_opt = np.array(x_opt, copy=True)

        if self.options["eval_noise"]:
            if self.options["use_het_noise"]:
                x_opt = x_opt1
            else:
                x_opt[-self.lvl :] = 10 ** x_opt[-self.lvl :]
                x_opt[: -self.lvl] = x_opt1
        else:
            x_opt = x_opt1
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

        sigma_0 = param[0]
        l_0 = param[1 : self.nx + 1]
        # param0 = param[0 : self.nx+1]
        sigmas_gamma = param[self.nx + 1 :: self.nx + 2]
        l_s = [
            param[i : i + self.nx].tolist()
            for i in np.arange(self.nx + 2, param.shape[0] - 1, self.nx + 2)
        ]
        # ls_gamma = param[3::3]
        rho_values = param[2 + 2 * self.nx :: self.nx + 2]

        # Sum of j=0 until l_^prime
        for j in range(Lp + 1):
            eta_j_l = self.eta(j, L, rho_values)
            eta_j_lp = self.eta(j, Lp, rho_values)

            if j == 0:
                # Cov(γ_j(x), γ_j(x')) using the kernel for K_00
                cov_gamma_j = self._compute_K(x, xp, [sigma_0, l_0])
            else:
                # Cov(γ_j(x), γ_j(x')) using the kernel
                cov_gamma_j = self._compute_K(x, xp, [sigmas_gamma[j - 1], l_s[j - 1]])
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
        means = []
        covariances = []
        x = (x - self.X_offset) / self.X_scale

        k_xZ = []

        for ind in range(self.lvl):
            if self.options["use_het_noise"]:
                k_xx = self.compute_diag_K(x, x, ind, ind, self.optimal_theta)
                varis = np.concatenate(self.options["noise0"])

                for j in range(self.lvl):
                    if ind >= j:
                        k_xZ.append(
                            self.compute_cross_K(
                                self.Z_norma_all[j],
                                x,
                                ind,
                                j,
                                self.optimal_theta,
                            )
                        )
                    else:
                        k_xZ.append(
                            self.compute_cross_K(
                                self.Z_norma_all[j],
                                x,
                                j,
                                ind,
                                self.optimal_theta,
                            )
                        )

            else:
                noises = self.optimal_theta[-self.lvl : :]
                k_xx = self.compute_diag_K(
                    x, x, ind, ind, self.optimal_theta[: -self.lvl]
                )
                varis = []
                for i, v in enumerate(noises):
                    varis = np.hstack(
                        [varis, np.full(self.X_norma_all[i].shape[0], noises[i])]
                    )

                for j in range(self.lvl):
                    if ind >= j:
                        k_xZ.append(
                            self.compute_cross_K(
                                self.Z_norma_all[j],
                                x,
                                ind,
                                j,
                                self.optimal_theta[: -self.lvl],
                            )
                        )
                    else:
                        k_xZ.append(
                            self.compute_cross_K(
                                self.Z_norma_all[j],
                                x,
                                j,
                                ind,
                                self.optimal_theta[: -self.lvl],
                            )
                        )
            means.append(
                self.y_std * (np.vstack(k_xZ).T @ self.woodbury_data["vec"])
                + self.y_mean
            )
            val = np.sum(
                np.dot(self.woodbury_data["inv"].T, np.vstack(k_xZ)) * np.vstack(k_xZ),
                0,
            )

            if self.options["use_het_noise"]:
                if self.options["predict_with_noise"]:
                    try:
                        import nlopt
                    except ImportError:
                        print("predict_with_noise olny available with nlopt")

                    opt = nlopt.opt(nlopt.LN_COBYLA, len(self.optimal_theta))
                    opt.set_lower_bounds(
                        10**self.lower_bounds
                    )  # Lower bounds for each dimension
                    opt.set_upper_bounds(
                        10**self.upper_bounds
                    )  # Upper bounds for each dimension
                    opt.set_min_objective(self.neg_log_likelihood_noise)
                    opt.set_maxeval(1000)
                    opt.set_xtol_rel(1e-6)
                    x_opt = opt.optimize(10**self.lower_bounds + 1)

                    opt_params_noise = x_opt

                    k_xZ.clear()
                    for j in range(self.lvl):
                        if ind >= j:
                            k_xZ.append(
                                self.compute_cross_K(
                                    self.Z[j],
                                    x * self.X_scale + self.X_offset,
                                    ind,
                                    j,
                                    opt_params_noise,
                                )
                            )
                        else:
                            k_xZ.append(
                                self.compute_cross_K(
                                    self.Z[j],
                                    x * self.X_scale + self.X_offset,
                                    j,
                                    ind,
                                    opt_params_noise,
                                )
                            )

                    pred_noise = (
                        np.vstack(k_xZ).T @ self.woodbury_vec_prednoise
                    ).flatten()

                    pred_noise = np.clip(pred_noise, 1e-15, np.inf)

                    var = (k_xx - val)[:, None]
                else:
                    var = (k_xx - val)[:, None]
            else:
                var = (k_xx + noises[ind] - val)[:, None]

            var = np.clip(var, 1e-15, np.inf)
            covariances.append(var * self.y_std**2)
            k_xZ.clear()
        # print("Optimal noise",noises * self.y_std**2)
        return means, covariances

    def predict_values(self, x, is_acting=None):
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

        return means[self.lvl - 1]

    def predict_variances(
        self, X: np.ndarray, is_acting=None, is_ri=False
    ) -> np.ndarray:
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
        means, covariances = self.predict_all_levels(X)

        return covariances[self.lvl - 1]

    def neg_log_likelihood(self, param, grad=None):
        if self.options["method"] == "FITC":
            likelihood, w_vec, w_inv = self._FITC(
                self.X_norma_all, self.y_norma_all, self.Z_norma_all, param
            )
            likelihood = likelihood[0][0]

        elif self.options["method"] == "VFE":
            likelihood, w_vec, w_inv = self._VFE(
                self.X_norma_all, self.y_norma_all, self.Z_norma_all, param
            )
            likelihood = likelihood[0][0]

        self.woodbury_data["vec"] = w_vec
        self.woodbury_data["inv"] = w_inv

        return likelihood

    def _VFE(self, X, Y, Z, param):
        """
        Compute VFE likelihood and associated Woodbury terms in a numerically stable way.
        """
        noises = param[-self.lvl : :]
        varis = []
        for i, v in enumerate(noises):
            varis = np.hstack([varis, np.full(X[i].shape[0], noises[i])])

        diag = []
        for i in range(self.lvl):
            diag.append(self.compute_diag_K(X[i], X[i], i, i, param[: -self.lvl]))
        K = np.concatenate(diag)

        Kmm = self.compute_blockwise_K(Z, Z, param[: -self.lvl])
        Knm = self.compute_blockwise_K(X, Z, param[: -self.lvl])
        U = np.linalg.cholesky(Kmm + np.eye(Kmm.shape[0]) * self.options["nugget"])

        Ui = np.linalg.inv(U)
        V = Ui @ Knm.T

        nu = varis
        beta = 1.0 / nu

        trace_term = beta * K - beta * np.sum(np.square(V), 0)

        A = np.eye(Kmm.shape[0]) + V * beta @ V.T
        L = np.linalg.cholesky(A + np.eye(A.shape[0]) * self.options["nugget"])
        Li = np.linalg.inv(L)
        a = np.einsum("ij,i->ij", self.y_norma_all, beta)
        b = Li @ V @ a

        likelihood = 0.5 * (
            +np.sum(np.log(nu))
            + 2.0 * np.sum(np.log(np.diag(L)))
            + a.T @ self.y_norma_all
            - np.einsum("ij,ij->", b, b)
            + np.sum(trace_term)
        )

        LiUi = Li @ Ui
        LiUiT = LiUi.T
        woodbury_vec = LiUiT @ b
        woodbury_inv = Ui.T @ Ui - LiUiT @ LiUi

        return likelihood, woodbury_vec, woodbury_inv

    def _FITC(self, X, Y, Z, param):
        varis = []
        if self.options["use_het_noise"]:
            varis = np.concatenate(self.options["noise0"])
        else:
            noises = param[-self.lvl : :]
            for i, v in enumerate(noises):
                varis = np.hstack([varis, np.full(X[i].shape[0], noises[i])])

        diag = []

        if self.options["use_het_noise"]:
            for i in range(self.lvl):
                diag.append(self.compute_diag_K(X[i], X[i], i, i, param))
            K = np.concatenate(diag)

            Kmm = self.compute_blockwise_K(Z, Z, param)
            Knm = self.compute_blockwise_K(X, Z, param)

        else:
            for i in range(self.lvl):
                diag.append(self.compute_diag_K(X[i], X[i], i, i, param[: -self.lvl]))
            K = np.concatenate(diag)

            Kmm = self.compute_blockwise_K(Z, Z, param[: -self.lvl])
            Knm = self.compute_blockwise_K(X, Z, param[: -self.lvl])
        U = np.linalg.cholesky(Kmm + np.eye(Kmm.shape[0]) * self.options["nugget"])

        Ui = np.linalg.inv(U)
        V = Ui @ Knm.T

        nu = K - np.sum(np.square(V), 0) + varis  # [:,0]

        beta = 1.0 / nu

        A = np.eye(Kmm.shape[0]) + V * beta @ V.T
        L = np.linalg.cholesky(A + np.eye(A.shape[0]) * self.options["nugget"])
        Li = np.linalg.inv(L)
        a = np.einsum("ij,i->ij", self.y_norma_all, beta)
        b = Li @ V @ a

        likelihood = 0.5 * (
            +np.sum(np.log(nu))
            + 2.0 * np.sum(np.log(np.diag(L)))
            + a.T @ self.y_norma_all
            - np.einsum("ij,ij->", b, b)
        )

        LiUi = Li @ Ui
        LiUiT = LiUi.T
        woodbury_vec = LiUiT @ b
        woodbury_inv = Ui.T @ Ui - LiUiT @ LiUi

        return likelihood, woodbury_vec, woodbury_inv

    def compute_diag_K(self, x, xp, L, Lp, param):
        """
        Calculation of the diagonal of K using the autoregressive formulation.
        Parmeters:
        - x: First input for the covariannce (np.ndarray)
        - xp: Second input for the covariannce (np.ndarray)
        - L: Level index of the first output (scalar)
        - Lp: Level index of the second output (scalar)
        - param: Set of Hyper-parameters (vector)
        Returns:
        - Diagonal variance for cov(y_l(x), y_{l'}(x')) (np.ndarray)
        """
        v_value = 0.0

        sigma_0 = param[0]

        sigmas_gamma = param[self.nx + 1 :: self.nx + 2]

        # ls_gamma = param[3::3]
        rho_values = param[2 + 2 * self.nx :: self.nx + 2]

        # Sum of j=0 until l_^prime
        for j in range(Lp + 1):
            eta_j_l = self.eta(j, L, rho_values)
            eta_j_lp = self.eta(j, Lp, rho_values)

            if j == 0:
                # Cov(γ_j(x), γ_j(x')) using the kernel for K_00
                variance = np.full(x.shape[0], sigma_0)

            else:
                # Cov(γ_j(x), γ_j(x')) using the kernel
                variance = np.full(x.shape[0], sigmas_gamma[j - 1])
            # Add to the value of the covariance
            v_value += eta_j_l * eta_j_lp * variance

        return v_value

    def neg_log_likelihood_scipy(self, param):
        """
        Likelihood for Cobyla-scipy (SMT) optimizer
        """
        if self.options["eval_noise"]:
            param1 = np.array(param[: -self.lvl], copy=True)
        else:
            param1 = np.array(param, copy=True)

        param1[0] = 10 ** (param[0])  # Apply 10** to Sigma 0
        param1[1 : self.nx + 1] = (
            10 ** (param[1 : self.nx + 1])
        )  # Apply 10** to length scales 0
        param1[self.nx + 1 :: self.nx + 2] = (
            10 ** (param1[self.nx + 1 :: self.nx + 2])
        )  # Apply 10** to sigmas gamma

        for i in np.arange(self.nx + 2, param1.shape[0] - 1, self.nx + 2):
            param1[i : i + self.nx] = 10 ** param1[i : i + self.nx]

        if self.options["eval_noise"]:
            param = np.array(param, copy=True)

            param[-self.lvl : :] = 10 ** param[-self.lvl : :]

            param[: -self.lvl] = param1
        else:
            param = np.array(param, copy=True)
            param = param1
        return self.neg_log_likelihood(param)

    def neg_log_likelihood_noise(self, param, grad=None):
        # param = np.append(param,self.optimal_theta[-1])

        if self.options["method"] == "FITC":
            likelihood, w_vec, w_inv = self._FITC(
                self.X, np.concatenate(self.options["noise0"]), self.Z, param
            )
            likelihood = likelihood[0][0]

        elif self.options["method"] == "VFE":
            likelihood, w_vec, w_inv = self._VFE(
                self.X, np.concatenate(self.options["noise0"]), self.Z, param
            )
            likelihood = likelihood[0][0]

        self.woodbury_vec_prednoise = w_vec

        return likelihood

    def neg_log_likelihood_nlopt(self, param, grad=None):
        """
        Likelihood for nlopt optimizers
        """
        if self.options["eval_noise"]:
            param1 = np.array(param[: -self.lvl], copy=True)
        else:
            param1 = np.array(param, copy=True)

        param1[0] = 10 ** (param[0])  # Apply 10** to Sigma 0
        param1[1 : self.nx + 1] = (
            10 ** (param[1 : self.nx + 1])
        )  # Apply 10** to length scales 0
        param1[self.nx + 1 :: self.nx + 2] = (
            10 ** (param1[self.nx + 1 :: self.nx + 2])
        )  # Apply 10** to sigmas gamma

        for i in np.arange(self.nx + 2, param1.shape[0] - 1, self.nx + 2):
            param1[i : i + self.nx] = 10 ** param1[i : i + self.nx]

        if self.options["eval_noise"]:
            param = np.array(param, copy=True)
            param[-self.lvl : :] = 10 ** param[-self.lvl : :]
            param[: -self.lvl] = param1
        else:
            param = np.array(param, copy=True)
            param = param1
        return self.neg_log_likelihood(param, grad)

    def compute_blockwise_K(self, X, Xprime, param):
        K_block = {}
        n = 0
        nprime = 0
        for i in X:
            n = n + i.shape[0]

        for i in Xprime:
            nprime = nprime + i.shape[0]
        for jp in range(self.lvl):
            for j in range(self.lvl):
                if jp >= j:
                    K_block[str(jp) + str(j)] = self.compute_cross_K(
                        X[j], Xprime[jp], jp, j, param
                    )
                else:
                    K_block[str(jp) + str(j)] = self.compute_cross_K(
                        X[j], Xprime[jp], j, jp, param
                    )
        K = np.zeros((n, nprime))
        row_init, col_init = 0, 0
        for j in range(self.lvl):
            col_init = 0
            for jp in range(self.lvl):
                r, c = K_block[str(jp) + str(j)].shape
                K[row_init : row_init + r, col_init : col_init + c] = K_block[
                    str(jp) + str(j)
                ]
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
        self.corr.theta = np.asarray(param[1])
        r = self.corr(d)
        R = r.reshape(A.shape[0], B.shape[0])
        K = param[0] * R
        return K
