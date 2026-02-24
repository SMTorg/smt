"""
Created on Fri Sep 20 14:55:43 2024

@author: mcastano
Multi-Fidelity co-Kriging: recursive formulation with autoregressive model of
order 1 (AR1)
Adapted on January 2021 by Andres Lopez-Lopera to the new SMT version
"""

import numpy as np
from scipy import linalg
from scipy.cluster.vq import kmeans
from scipy.linalg import solve_triangular

from smt.surrogate_models.sgp import SGP
from smt.surrogate_models.krg_based.distances import (
    cross_distances,
)
from smt.applications.mfk import MFK


class SMFK(MFK):
    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        declare(
            "rho_regr",
            "constant",
            values=("constant", "linear", "quadratic"),
            desc="Regression function type for rho",
        )
        declare(
            "optim_var",
            False,
            types=bool,
            values=(True, False),
            desc="If True, the variance at HF samples is forced to zero",
        )
        declare(
            "n_inducing",
            4,
            types=(int),
            desc="Number of inducing points for the lowest fidelity \
                level must be less or equal to the DoE in the lowest fideliy.",
        )
        declare(
            "propagate_uncertainty",
            True,
            types=bool,
            values=(True, False),
            desc="If True, the variance cotribution of lower fidelity levels are considered",
        )
        self.name = "SMFK"

        self.X2_norma = {}
        self.X2_offset = {}
        self.X2_scale = {}
        self.Z = None

    def _new_train_iteration(self, lvl):
        # n_samples = self.nt_all
        self.options["noise0"] = np.array([self.options["noise0"][lvl]]).flatten()
        self.options["theta0"] = self.options["theta0"][lvl, :]

        self.X_norma = self.X_norma_all[lvl]
        self.y_norma = self.y_norma_all[lvl]

        if self.options["eval_noise"]:
            if self.options["use_het_noise"]:
                # hetGP works with unique design variables
                (
                    self.X_norma,
                    self.index_unique,  # do we need to store it?
                    self.nt_reps,  # do we need to store it?
                ) = np.unique(
                    self.X_norma, return_inverse=True, return_counts=True, axis=0
                )
                self.nt_all[lvl] = self.X_norma.shape[0]

                # computing the mean of the output per unique design variable (see Binois et al., 2018)
                y_norma_unique = []
                for i in range(self.nt_all[lvl]):
                    y_norma_unique.append(np.mean(self.y_norma[self.index_unique == i]))
                y_norma_unique = np.array(y_norma_unique).reshape(-1, 1)

                # pointwise sensible estimates of the noise variances (see Ankenman et al., 2010)
                self.optimal_noise = self.options["noise0"] * np.ones(self.nt_all[lvl])
                for i in range(self.nt_all[lvl]):
                    diff = self.y_norma[self.index_unique == i] - y_norma_unique[i]
                    if np.sum(diff**2) != 0.0:
                        self.optimal_noise[i] = np.std(diff, ddof=1) ** 2
                self.optimal_noise = self.optimal_noise / self.nt_reps
                self.optimal_noise_all[lvl] = self.optimal_noise
                self.y_norma = y_norma_unique

                self.X_norma_all[lvl] = self.X_norma
                self.y_norma_all[lvl] = self.y_norma
        else:
            self.optimal_noise = self.options["noise0"] / self.y_std**2
            self.optimal_noise_all[lvl] = self.optimal_noise

        # Calculate matrix of distances D between samples
        if self.is_continuous:
            if lvl != 0:
                self.D_all[lvl] = cross_distances(self.X_norma)

        # Regression matrix and parameters
        if lvl != 0:
            self.F_all[lvl] = self._regression_types[self.options["poly"]](self.X_norma)
            self.p_all[lvl] = self.F_all[lvl].shape[1]

        if lvl == 0:
            # l = np.std(self.X_norma, axis=0)
            # theta = 1/l**2
            # bounds = [1e-8, 1e2]

            data = np.hstack((self.X_norma, self.y_norma))
            Z2 = kmeans(data, self.options["n_inducing"])[0][:, :-1]

            sgp = SGP(
                method="FITC",
                nugget=self.options["nugget"],
                print_prediction=False,
                print_global=False,
            )
            sgp.set_training_values(self.X_norma, self.y_norma)
            sgp.set_inducing_inputs(Z=Z2)
            sgp.train()

            self.sgp = sgp
            self.Z = Z2 * self.X_scale + self.X_offset

        # Concatenate the autoregressive part for levels > 0
        if lvl > 0:
            D, self.ij = self.D_all[lvl]
            self.F = self.F_all[lvl]
            self._lvl = lvl
            self.nt = self.nt_all[lvl]
            self.q = self.q_all[lvl]
            self.p = self.p_all[lvl]
            self.kplsk_second_loop = False

            F_rho = self._regression_types[self.options["rho_regr"]](self.X_norma)
            self.q_all[lvl] = F_rho.shape[1]

            if self.is_continuous:
                if lvl == 1:
                    pred_old = self.sgp.predict_values(self.X_norma)
                else:
                    pred_old = self._predict_intermediate_values(
                        self.X_norma, lvl, descale=False
                    )

                self.F_all[lvl] = np.hstack(
                    (
                        F_rho
                        * np.dot(
                            pred_old,
                            np.ones((1, self.q_all[lvl])),
                        ),
                        self.F_all[lvl],
                    )
                )
            else:
                self.F_all[lvl] = np.hstack(
                    (
                        F_rho
                        * np.dot(
                            self._predict_intermediate_values(
                                self.X[lvl], lvl, descale=False
                            ),
                            np.ones((1, self.q_all[lvl])),
                        ),
                        self.F_all[lvl],
                    )
                )
            # n_samples_F_i = self.F_all[lvl].shape[0]

            self.F = self.F_all[lvl]
            D, self.ij = self.D_all[lvl]
            self._lvl = lvl
            self.nt = self.nt_all[lvl]
            self.q = self.q_all[lvl]
            self.p = self.p_all[lvl]
            self.kplsk_second_loop = False
            (
                self.optimal_rlf_value[lvl],
                self.optimal_par[lvl],
                self.optimal_theta[lvl],
            ) = self._optimize_hyperparam(D)

            if self.options["eval_noise"] and not self.options["use_het_noise"]:
                tmp_list = self.optimal_theta[lvl]
                self.optimal_theta[lvl] = tmp_list[:-1]
                self.optimal_noise = tmp_list[-1]
                self.optimal_noise_all[lvl] = self.optimal_noise
            del self.y_norma, self.D, self.optimal_noise

    def _predict_intermediate_values(self, X, lvl, descale=True):
        """
        Evaluates the model at a set of points.
        Used for training the model at level lvl.
        Allows to relax the order problem.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values
        lvl : level at which the prediction is made

        Returns
        -------
        y : np.ndarray
            Evaluation point output variable values
        """
        n_eval, _ = X.shape
        #        if n_features_X != self.n_features:
        #            raise ValueError("Design must be an array of n_features columns.")

        # Calculate kriging mean and variance at level 0
        mu = np.zeros((n_eval, lvl))

        # if not (self.is_continuous):
        #     X_usc = X
        if descale and self.is_continuous:
            X = (X - self.X_offset) / self.X_scale

        f = self._regression_types[self.options["poly"]](X)
        f0 = self._regression_types[self.options["poly"]](X)

        mu[:, 0] = self.sgp.predict_values(X)[:, 0]

        # Calculate recursively kriging mean and variance at level i
        for i in range(1, lvl):
            g = self._regression_types[self.options["rho_regr"]](X)

            if self.is_continuous:
                dx = self._differences(X, Y=self.X_norma_all[i])
                d = self._componentwise_distance(dx)
                self.corr.theta = self.optimal_theta[i]
                r_ = self.corr(d).reshape(n_eval, self.nt_all[i])

            f = np.vstack((g.T * mu[:, i - 1], f0.T))
            beta = self.optimal_par[i]["beta"]
            gamma = self.optimal_par[i]["gamma"]
            # scaled predictor
            mu[:, i] = (np.dot(f.T, beta) + np.dot(r_, gamma)).ravel()

        # scaled predictor
        if descale:
            mu = mu * self.y_std + self.y_mean

        return mu[:, -1].reshape((n_eval, 1))

    def _predict_values(self, X, is_acting=None):
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

        return self._predict_intermediate_values(X, self.nlvl)

    def _predict_variances(
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
        return self.predict_variances_all_levels(X)[0][:, -1]

    def predict_variances_all_levels(self, X, is_acting=None):
        # def predict_variances_all_levels(self, X):
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
        # Initialization X = atleast_2d(X)
        nlevel = self.nlvl
        sigma2_rhos = []
        n_eval, n_features_X = X.shape
        #        if n_features_X != self.n_features:
        #            raise ValueError("Design must be an array of n_features columns.")

        X = (X - self.X_offset) / self.X_scale

        # Calculate kriging mean and variance at level 0
        mu = np.zeros((n_eval, nlevel))

        f = self._regression_types[self.options["poly"]](X)
        f0 = self._regression_types[self.options["poly"]](X)

        # Get regression function and correlation
        # F = self.F_all[0]
        # C = self.optimal_par[0]["C"]

        # beta = self.optimal_par[0]["beta"]
        # Ft = solve_triangular(C, F, lower=True)

        # if self.is_continuous:
        #     dx = self._differences(X, Y=self.X_norma_all[0])
        #     d = self._componentwise_distance(dx)
        #     self.corr.theta = self.optimal_theta[0]
        #     r_ = self.corr(d).reshape(n_eval, self.nt_all[0])

        # gamma = self.optimal_par[0]["gamma"]

        # Scaled predictor
        # mu[:, 0] = (np.dot(f, beta) + np.dot(r_, gamma)).ravel()

        mu[:, 0] = self.sgp.predict_values(X)[:, 0]

        self.sigma2_rho = nlevel * [None]
        MSE = np.zeros((n_eval, nlevel))
        # r_t = solve_triangular(C, r_.T, lower=True)
        # G = self.optimal_par[0]["G"]

        # u_ = solve_triangular(G.T, f.T - np.dot(Ft.T, r_t), lower=True)
        # sigma2 = self.optimal_par[0]["sigma2"] / self.y_std**2
        # MSE[:, 0] = sigma2 * (
        #     # 1 + self.optimal_noise_all[0] - (r_t ** 2).sum(axis=0) + (u_ ** 2).sum(axis=0)
        #     1 - (r_t**2).sum(axis=0) + (u_**2).sum(axis=0)
        # )

        MSE[:, 0] = self.sgp.predict_variances(X)[:, 0]

        # Calculate recursively kriging variance at level i
        for i in range(1, nlevel):
            F = self.F_all[i]
            C = self.optimal_par[i]["C"]

            g = self._regression_types[self.options["rho_regr"]](X)

            if self.is_continuous:
                dx = self._differences(X, Y=self.X_norma_all[i])
                d = self._componentwise_distance(dx)
                self.corr.theta = self.optimal_theta[i]
                r_ = self.corr(d).reshape(n_eval, self.nt_all[i])

            f = np.vstack((g.T * mu[:, i - 1], f0.T))

            Ft = solve_triangular(C, F, lower=True)
            yt = solve_triangular(C, self.y_norma_all[i], lower=True)
            r_t = solve_triangular(C, r_.T, lower=True)
            G = self.optimal_par[i]["G"]
            beta = self.optimal_par[i]["beta"]

            # scaled predictor
            sigma2 = self.optimal_par[i]["sigma2"] / self.y_std**2
            q = self.q_all[i]
            u_ = solve_triangular(G.T, f - np.dot(Ft.T, r_t), lower=True)
            sigma2_rho = np.dot(
                g,
                sigma2 * linalg.inv(np.dot(G.T, G))[:q, :q]
                + np.dot(beta[:q], beta[:q].T),
            )
            sigma2_rho = (sigma2_rho * g).sum(axis=1)
            sigma2_rhos.append(sigma2_rho)

            if self.name in ["MFKPLS", "MFKPLSK"]:
                p = self.p_all[i]
                Q_ = (np.dot((yt - np.dot(Ft, beta)).T, yt - np.dot(Ft, beta)))[0, 0]
                MSE[:, i] = (
                    # sigma2_rho * MSE[:, i - 1]
                    +Q_
                    / (2 * (self.nt_all[i] - p - q))
                    # * (1 + self.optimal_noise_all[i] - (r_t ** 2).sum(axis=0))
                    * (1 - (r_t**2).sum(axis=0))
                    + sigma2 * (u_**2).sum(axis=0)
                )
            else:
                MSE[:, i] = sigma2 * (
                    1
                    + self.optimal_noise_all[i]
                    - (r_t**2).sum(axis=0)
                    + (u_**2).sum(axis=0)
                    # 1 - (r_t**2).sum(axis=0) + (u_**2).sum(axis=0)
                )  # + sigma2_rho * MSE[:, i - 1]
            if self.options["propagate_uncertainty"]:
                MSE[:, i] = MSE[:, i] + sigma2_rho * MSE[:, i - 1]

        # scaled predictor
        MSE *= self.y_std**2

        return MSE, sigma2_rhos
