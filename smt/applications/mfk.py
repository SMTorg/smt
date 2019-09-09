# -*- coding: utf-8 -*-
"""
Created on Fri May 04 10:26:49 2018

@author: Mostafa Meliani <melimostafa@gmail.com>
Multi-Fidelity co-Kriging: recursive formulation with autoregressive model of 
order 1 (AR1)
"""

from __future__ import division
import numpy as np
from smt.surrogate_models.krg_based import KrgBased
from types import FunctionType
from smt.utils.kriging_utils import (
    l1_cross_distances,
    componentwise_distance,
    standardization,
)
from scipy.linalg import solve_triangular
from scipy import linalg
from sklearn.metrics.pairwise import manhattan_distances
import copy
from copy import deepcopy
from sys import exit

"""
The MFK class.
"""


class MFK(KrgBased):

    """
    - MFK
    """

    def _initialize(self):
        super(MFK, self)._initialize()
        declare = self.options.declare

        declare(
            "rho_regr",
            "constant",
            values=("constant", "linear", "quadratic"),
            desc="Regression function type for rho",
        )
        declare("theta0", types=(list, np.ndarray), desc="Initial hyperparameters")
        declare(
            "optim_var",
            False,
            types=bool,
            values=(True, False),
            desc="Turning this option to True, forces variance to zero at HF samples ",
        )
        declare(
            "eval_noise",
            False,
            types=bool,
            values=(True, False),
            desc="noise evaluation flag",
        )
        declare("noise0", 1e-6, types=float, desc="Initial noise hyperparameter")
        self.name = "MFK"

    def _check_list_structure(self, X, y):
        """
        checks if the data structure is compatible with MFK.
        sets class attributes such as (number of levels of Fidelity, training points in each level, ...)
        
        Arguments :
        X : list of arrays, each array corresponds to a fidelity level. starts from lowest to highest
        y : same as X 
        """

        if type(X) is not list:
            nlevel = 1
            X = [X]
        else:
            nlevel = len(X)

        if type(y) is not list:
            y = [y]

        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        n_samples = np.zeros(nlevel, dtype=int)
        n_features = np.zeros(nlevel, dtype=int)
        n_samples_y = np.zeros(nlevel, dtype=int)
        for i in range(nlevel):
            n_samples[i], n_features[i] = X[i].shape
            if i > 1 and n_features[i] != n_features[i - 1]:
                raise ValueError("All X must have the same number of columns.")
            y[i] = np.asarray(y[i]).ravel()[:, np.newaxis]
            n_samples_y[i] = y[i].shape[0]
            if n_samples[i] != n_samples_y[i]:
                raise ValueError("X and y must have the same number of rows.")

        self.nx = n_features[0]
        self.nt_all = n_samples
        self.nlvl = nlevel
        self.ny = y[0].shape[1]
        self.X = X[:]
        self.y = y[:]

    def _new_train(self):
        """
        Overrides KrgBased implementation
        Trains the Multi-Fidelity model
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

        self._check_list_structure(xt, yt)
        self._check_param()
        X = self.X
        y = self.y

        _, _, self.X_mean, self.y_mean, self.X_std, self.y_std = standardization(
            np.concatenate(xt, axis=0), np.concatenate(yt, axis=0)
        )

        nlevel = self.nlvl
        n_samples = self.nt_all

        # initialize lists
        self.noise = nlevel * [0]
        self.D_all = nlevel * [0]
        self.F_all = nlevel * [0]
        self.p_all = nlevel * [0]
        self.q_all = nlevel * [0]
        self.optimal_rlf_value = nlevel * [0]
        self.optimal_par = nlevel * [{}]
        self.optimal_theta = nlevel * [0]
        self.X_norma_all = [(x - self.X_mean) / self.X_std for x in X]
        self.y_norma_all = [(f - self.y_mean) / self.y_std for f in y]

        for lvl in range(nlevel):
            self.X_norma = self.X_norma_all[lvl]
            self.y_norma = self.y_norma_all[lvl]
            # Calculate matrix of distances D between samples
            self.D_all[lvl] = l1_cross_distances(self.X_norma)

            # Regression matrix and parameters
            self.F_all[lvl] = self._regression_types[self.options["poly"]](self.X_norma)
            self.p_all[lvl] = self.F_all[lvl].shape[1]

            # Concatenate the autoregressive part for levels > 0
            if lvl > 0:
                F_rho = self._regression_types[self.options["rho_regr"]](self.X_norma)
                self.q_all[lvl] = F_rho.shape[1]
                self.F_all[lvl] = np.hstack(
                    (
                        F_rho
                        * np.dot(
                            self._predict_intermediate_values(
                                self.X_norma, lvl, descale=False
                            ),
                            np.ones((1, self.q_all[lvl])),
                        ),
                        self.F_all[lvl],
                    )
                )
            else:
                self.q_all[lvl] = 0

            n_samples_F_i = self.F_all[lvl].shape[0]

            if n_samples_F_i != n_samples[lvl]:
                raise Exception(
                    "Number of rows in F and X do not match. Most "
                    "likely something is going wrong with the "
                    "regression model."
                )

            if int(self.p_all[lvl] + self.q_all[lvl]) >= n_samples_F_i:
                raise Exception(
                    (
                        "Ordinary least squares problem is undetermined "
                        "n_samples=%d must be greater than the regression"
                        " model size p+q=%d."
                    )
                    % (n_samples[i], self.p_all[lvl] + self.q_all[lvl])
                )

            # Determine Gaussian Process model parameters
            self.F = self.F_all[lvl]
            D, self.ij = self.D_all[lvl]
            self._lvl = lvl
            self.nt = self.nt_all[lvl]
            self.q = self.q_all[lvl]
            self.p = self.p_all[lvl]
            self.optimal_rlf_value[lvl], self.optimal_par[lvl], self.optimal_theta[
                lvl
            ] = self._optimize_hyperparam(D)
            if self.options["eval_noise"]:
                tmp_list = self.optimal_theta[lvl]
                self.optimal_theta[lvl] = tmp_list[:-1]
                self.noise[lvl] = tmp_list[-1]
            del self.y_norma, self.D

        if self.options["eval_noise"] and self.options["optim_var"]:
            for lvl in range(self.nlvl - 1):
                self.set_training_values(
                    X[lvl], self._predict_intermediate_values(X[lvl], lvl + 1), name=lvl
                )
            self.set_training_values(
                X[-1], self._predict_intermediate_values(X[-1], self.nlvl)
            )
            self.options["eval_noise"] = False
            self._new_train()

    def _componentwise_distance(self, dx, opt=0):
        d = componentwise_distance(dx, self.options["corr"], self.nx)
        return d

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
        #        if self.normalize:
        if descale:
            X = (X - self.X_mean) / self.X_std
        ##                X = (X - self.X_mean[0]) / self.X_std[0]
        f = self._regression_types[self.options["poly"]](X)
        f0 = self._regression_types[self.options["poly"]](X)
        dx = manhattan_distances(X, Y=self.X_norma_all[0], sum_over_features=False)
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

        # Calculate recursively kriging mean and variance at level i
        for i in range(1, lvl):
            F = self.F_all[i]
            C = self.optimal_par[i]["C"]
            g = self._regression_types[self.options["rho_regr"]](X)
            dx = manhattan_distances(X, Y=self.X_norma_all[i], sum_over_features=False)
            d = self._componentwise_distance(dx)
            r_ = self._correlation_types[self.options["corr"]](
                self.optimal_theta[i], d
            ).reshape(n_eval, self.nt_all[i])
            f = np.vstack((g.T * mu[:, i - 1], f0.T))
            Ft = solve_triangular(C, F, lower=True)
            yt = solve_triangular(C, self.y_norma_all[i], lower=True)
            beta = self.optimal_par[i]["beta"]
            gamma = self.optimal_par[i]["gamma"]
            # scaled predictor
            mu[:, i] = (np.dot(f.T, beta) + np.dot(r_, gamma)).ravel()

        # scaled predictor
        if descale:
            for i in range(lvl):  # Predictor
                mu[:, i] = self.y_mean + self.y_std * mu[:, i]

        return mu[:, -1].reshape((n_eval, 1))

    def _predict_values(self, X):
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

    def _predict_variances(self, X):
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

    def predict_variances_all_levels(self, X):
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
        X = (X - self.X_mean) / self.X_std
        # Calculate kriging mean and variance at level 0
        mu = np.zeros((n_eval, nlevel))
        #        if self.normalize:
        f = self._regression_types[self.options["poly"]](X)
        f0 = self._regression_types[self.options["poly"]](X)
        dx = manhattan_distances(X, Y=self.X_norma_all[0], sum_over_features=False)
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
            1 + self.noise[0] - (r_t ** 2).sum(axis=0) + (u_ ** 2).sum(axis=0)
        )

        # Calculate recursively kriging variance at level i
        for i in range(1, nlevel):
            F = self.F_all[i]
            C = self.optimal_par[i]["C"]
            g = self._regression_types[self.options["rho_regr"]](X)
            dx = manhattan_distances(X, Y=self.X_norma_all[i], sum_over_features=False)
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
                + Q_
                / (2 * (self.nt_all[i] - p - q))
                * (1 + self.noise[i] - (r_t ** 2).sum(axis=0))
                + sigma2 * (u_ ** 2).sum(axis=0)
            )

        # scaled predictor
        for i in range(nlevel):  # Predictor
            MSE[:, i] = self.y_std ** 2 * MSE[:, i]

        return MSE, sigma2_rhos

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
        y : np.ndarray*self.y_std/self.X_std[kx])
            Derivative values.
        """

        lvl = self.nlvl
        # Initialization

        n_eval, n_features_x = x.shape
        x = (x - self.X_mean) / self.X_std

        dy_dx = np.zeros((n_eval, lvl))

        if self.options["corr"] != "squar_exp":
            raise ValueError(
                "The derivative is only available for square exponential kernel"
            )
        if self.options["poly"] == "constant":
            df = np.zeros([n_eval, 1])
        elif self.options["poly"] == "linear":
            df = np.zeros((n_eval, self.nx + 1))
            df[:, 1:] = 1
        else:
            raise ValueError(
                "The derivative is only available for ordinary kriging or "
                + "universal kriging using a linear trend"
            )
        df0 = copy.deepcopy(df)
        if self.options["rho_regr"] != "constant":
            raise ValueError(
                "The derivative is only available for regression rho constant"
            )
        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(x, Y=self.X_norma_all[0], sum_over_features=False)
        d = self._componentwise_distance(dx)
        # Compute the correlation function
        r_ = self._correlation_types[self.options["corr"]](
            self.optimal_theta[0], d
        ).reshape(n_eval, self.nt_all[0])

        # Beta and gamma = R^-1(y-FBeta)
        beta = self.optimal_par[0]["beta"]
        gamma = self.optimal_par[0]["gamma"]

        df_dx = np.dot(df, beta)
        d_dx = x[:, kx].reshape((n_eval, 1)) - self.X_norma_all[0][:, kx].reshape(
            (1, self.nt_all[0])
        )
        theta = self.optimal_theta[0]

        dy_dx[:, 0] = np.ravel((df_dx - 2 * theta[kx] * np.dot(d_dx * r_, gamma)))

        # Calculate recursively derivative at level i
        for i in range(1, lvl):
            F = self.F_all[i]
            C = self.optimal_par[i]["C"]
            g = self._regression_types[self.options["rho_regr"]](x)
            dx = manhattan_distances(x, Y=self.X_norma_all[i], sum_over_features=False)
            d = self._componentwise_distance(dx)
            r_ = self._correlation_types[self.options["corr"]](
                self.optimal_theta[i], d
            ).reshape(n_eval, self.nt_all[i])
            df = np.vstack((g.T * dy_dx[:, i - 1], df0.T))

            Ft = solve_triangular(C, F, lower=True)
            yt = solve_triangular(C, self.y_norma_all[i], lower=True)
            beta = self.optimal_par[i]["beta"]
            gamma = self.optimal_par[i]["gamma"]

            df_dx = np.dot(df.T, beta)
            d_dx = x[:, kx].reshape((n_eval, 1)) - self.X_norma_all[i][:, kx].reshape(
                (1, self.nt_all[i])
            )
            theta = self.optimal_theta[i]
            # scaled predictor
            dy_dx[:, i] = np.ravel(df_dx - 2 * theta[kx] * np.dot(d_dx * r_, gamma))

        return dy_dx[:, -1] * self.y_std / self.X_std[kx]
