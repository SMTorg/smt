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
from smt.applications.mfck import MFCK
from smt.utils.misc import standardization
from scipy.cluster.vq import kmeans


class SMFCK(MFCK):
    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        self.name = "SMFCK"

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
            desc="if use_het_noise is true, prediction of the variance over the test set",
        )
        declare(
            "random_state",
            0,
            types=int,
            desc="Seed for reproducibility",
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
            values=["FITC", "VFE"],
            desc="Methods available for Sparse Multi-fidelity",
            types=str,
        )
        declare(
            "nugget",
            1000000.0
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
        self.options["hyper_opt"] = (
            "Cobyla-nlopt"  # MFCK doesn't support gradient-based optimizers
        )
        self.woodbury_data = {"vec": None, "inv": None}

    def train(self):
        """
        Overrides MFCK implementation
        Trains the Sparse Multi-Fidelity co-Kriging model
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
                        seed=self.options["random_state"],
                    )[0]
                )
            i = i + 1
        xt.append(self.training_points[None][0][0])
        yt.append(self.training_points[None][0][1])

        if self.options["inducing_method"] == "random":
            idx = np.random.permutation(self.nt)[: self.options["n_inducing"][i]]
            zt.append(xt[idx])
        elif self.options["inducing_method"] == "kmeans":
            zt.append(
                kmeans(
                    self.training_points[None][0][0],
                    self.options["n_inducing"][i],
                    seed=self.options["random_state"],
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

        super().train()

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
