# -*- coding: utf-8 -*-
"""
Created on Fri May 04 10:26:49 2018

@author: Mostafa Meliani <melimostafa@gmail.com>
Multi-Fidelity co-Kriging: recursive formulation with autoregressive model of order 1 (AR1)
Partial Least Square decomposition added on highest fidelity level
Adapted March 2020 by Nathalie Bartoli to the new SMT version
"""

from copy import deepcopy
from sys import exit
import numpy as np
from scipy.linalg import solve_triangular
from scipy import linalg
from packaging import version
from sklearn import __version__ as sklversion

if version.parse(sklversion) < version.parse("0.22"):
    from sklearn.cross_decomposition.pls_ import PLSRegression as pls
else:
    from sklearn.cross_decomposition import PLSRegression as pls
from sklearn.metrics.pairwise import manhattan_distances

from smt.utils.kriging_utils import (
    cross_distances,
    componentwise_distance,
    standardization,
)
from smt.applications import MFK
from smt.utils.kriging_utils import componentwise_distance_PLS


class MFKPLS(MFK):
    def _initialize(self):
        super(MFKPLS, self)._initialize()
        declare = self.options.declare
        declare("n_comp", 1, types=int, desc="Number of principal components")
        self.name = "MFKPLS"

    def _componentwise_distance(self, dx, opt=0):
        d = componentwise_distance_PLS(
            dx, self.options["corr"], self.options["n_comp"], self.coeff_pls
        )
        return d

    def _compute_pls(self, X, y):
        _pls = pls(self.options["n_comp"])
        self.coeff_pls = _pls.fit(X.copy(), y.copy()).x_rotations_

        return X, y

    def predict_variances_all_levels(self, X):
        """
        Overrides MFK implementation
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
        #        if self.normalize:
        f = self._regression_types[self.options["poly"]](X)
        f0 = self._regression_types[self.options["poly"]](X)
        dx = self.differences(X, Y=self.X_norma_all[0])
        d = self._componentwise_distance(dx)

        # Get regression function and correlation
        F = self.F_all[0]
        C = self.optimal_par[0]["C"]

        beta = self.optimal_par[0]["beta"]
        Ft = solve_triangular(C, F, lower=True)
        yt = solve_triangular(C, self.y_norma_all[0], lower=True)
        r_ = self._correlation_types[self.options["corr"]](
            self.optimal_theta[0], d
        ).reshape(n_eval, self.nt_all[0])
        gamma = self.optimal_par[0]["gamma"]

        # Scaled predictor
        mu[:, 0] = (np.dot(f, beta) + np.dot(r_, gamma)).ravel()

        self.sigma2_rho = nlevel * [None]
        MSE = np.zeros((n_eval, nlevel))
        r_t = solve_triangular(C, r_.T, lower=True)
        G = self.optimal_par[0]["G"]

        u_ = solve_triangular(G.T, f.T - np.dot(Ft.T, r_t), lower=True)
        MSE[:, 0] = self.optimal_par[0]["sigma2"] * (
            # 1 + self.noise[0] - (r_t ** 2).sum(axis=0) + (u_ ** 2).sum(axis=0)
            1
            - (r_t ** 2).sum(axis=0)
            + (u_ ** 2).sum(axis=0)
        )

        # Calculate recursively kriging variance at level i
        for i in range(1, nlevel):
            F = self.F_all[i]
            C = self.optimal_par[i]["C"]
            g = self._regression_types[self.options["rho_regr"]](X)
            dx = self.differences(X, Y=self.X_norma_all[i])
            d = self._componentwise_distance(dx)
            r_ = self._correlation_types[self.options["corr"]](
                self.optimal_theta[i], d
            ).reshape(n_eval, self.nt_all[i])
            f = np.vstack((g.T * mu[:, i - 1], f0.T))

            Ft = solve_triangular(C, F, lower=True)
            yt = solve_triangular(C, self.y_norma_all[i], lower=True)
            r_t = solve_triangular(C, r_.T, lower=True)
            G = self.optimal_par[i]["G"]
            beta = self.optimal_par[i]["beta"]

            # scaled predictor
            sigma2 = self.optimal_par[i]["sigma2"]
            q = self.q_all[i]
            p = self.p_all[i]
            Q_ = (np.dot((yt - np.dot(Ft, beta)).T, yt - np.dot(Ft, beta)))[0, 0]
            u_ = solve_triangular(G.T, f - np.dot(Ft.T, r_t), lower=True)
            sigma2_rho = np.dot(
                g,
                sigma2 * linalg.inv(np.dot(G.T, G))[:q, :q]
                + np.dot(beta[:q], beta[:q].T),
            )
            sigma2_rho = (sigma2_rho * g).sum(axis=1)
            sigma2_rhos.append(sigma2_rho)

            MSE[:, i] = (
                sigma2_rho * MSE[:, i - 1]
                + Q_ / (2 * (self.nt_all[i] - p - q))
                # * (1 + self.noise[i] - (r_t ** 2).sum(axis=0))
                * (1 - (r_t ** 2).sum(axis=0))
                + sigma2 * (u_ ** 2).sum(axis=0)
            )

        # scaled predictor
        for i in range(nlevel):  # Predictor
            MSE[:, i] = self.y_std ** 2 * MSE[:, i]

        return MSE, sigma2_rhos
