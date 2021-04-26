# -*- coding: utf-8 -*-
"""
Created on Fri May 04 10:26:49 2018

@author: Mostafa Meliani <melimostafa@gmail.com>
Multi-Fidelity co-Kriging: recursive formulation with autoregressive model of
order 1 (AR1)
Adapted on January 2021 by Andres Lopez-Lopera to the new SMT version
"""

from copy import deepcopy

import numpy as np
from scipy.linalg import solve_triangular
from scipy import linalg
from scipy.spatial.distance import cdist

from packaging import version
from sklearn import __version__ as sklversion

if version.parse(sklversion) < version.parse("0.22"):
    from sklearn.cross_decomposition.pls_ import PLSRegression as pls
else:
    from sklearn.cross_decomposition import PLSRegression as pls

from smt.surrogate_models.krg_based import KrgBased
from smt.sampling_methods import LHS
from smt.utils.kriging_utils import (
    cross_distances,
    componentwise_distance,
    standardization,
    differences,
)


class NestedLHS(object):
    def __init__(self, nlevel, xlimits, random_state=None):
        """
        Constructor where values of options can be passed in.

        Parameters
        ----------
        nlevel : integer.
            The number of design of experiments to be built

        xlimits : ndarray
            The interval of the domain in each dimension with shape (nx, 2)

        random_state : Numpy RandomState object or seed number which controls random draws

        """
        self.nlevel = nlevel
        self.xlimits = xlimits
        self.random_state = random_state

    def __call__(self, nb_samples_hifi):
        """
        Builds nlevel nested design of experiments of dimension dim and size n_samples.
        Each doe sis built with the optmized lhs procedure.
        Builds the highest level first; nested properties are ensured by deleting
        the nearest neighbours in lower levels of fidelity.

        Parameters
        ----------

        nb_samples_hifi: The number of samples of the highest fidelity model.
            nb_samples_fi(n-1) = 2 * nb_samples_fi(n)


        Returns
        ------

        list of length nlevel of design of experiemnts from low to high fidelity level.
        """
        nt = []
        for i in range(self.nlevel, 0, -1):
            nt.append(pow(2, i - 1) * nb_samples_hifi)

        if len(nt) != self.nlevel:
            raise ValueError("nt must be a list of nlevel elements")
        if np.allclose(np.sort(nt)[::-1], nt) == False:
            raise ValueError("nt must be a list of decreasing integers")

        doe = []
        p0 = LHS(xlimits=self.xlimits, criterion="ese", random_state=self.random_state)
        doe.append(p0(nt[0]))

        for i in range(1, self.nlevel):
            p = LHS(
                xlimits=self.xlimits, criterion="ese", random_state=self.random_state
            )
            doe.append(p(nt[i]))

        for i in range(1, self.nlevel)[::-1]:
            ind = []
            d = cdist(doe[i], doe[i - 1], "euclidean")
            for j in range(doe[i].shape[0]):
                dj = np.sort(d[j, :])
                k = dj[0]
                l = (np.where(d[j, :] == k))[0][0]
                m = 0
                while l in ind:
                    m = m + 1
                    k = dj[m]
                    l = (np.where(d[j, :] == k))[0][0]
                ind.append(l)

            doe[i - 1] = np.delete(doe[i - 1], ind, axis=0)
            doe[i - 1] = np.vstack((doe[i - 1], doe[i]))
        return doe


class MFK(KrgBased):
    def _initialize(self):
        super(MFK, self)._initialize()
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
        self.name = "MFK"

    def _differences(self, X, Y):
        """
        Compute the distances
        """
        return differences(X, Y)

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
        self._new_train_init()
        theta0 = self.options["theta0"].copy()
        noise0 = self.options["noise0"].copy()

        for lvl in range(self.nlvl):
            self._new_train_iteration(lvl)
            self.options["theta0"] = theta0
            self.options["noise0"] = noise0

        self._new_train_finalize(lvl)

    def _new_train_init(self):
        if self.name in ["MFKPLS", "MFKPLSK"]:
            _pls = pls(self.options["n_comp"])

            # As of sklearn 0.24.1 PLS with zeroed outputs raises an exception while sklearn 0.23 returns zeroed x_rotations
            # For now the try/except below is a workaround to restore the 0.23 behaviour
            try:
                # PLS is done on the highest fidelity identified by the key None
                self.m_pls = _pls.fit(
                    self.training_points[None][0][0].copy(),
                    self.training_points[None][0][1].copy(),
                )
                self.coeff_pls = self.m_pls.x_rotations_
            except StopIteration:
                self.coeff_pls = np.zeros(
                    self.training_points[None][0][0].shape[1], self.options["n_comp"]
                )

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

        _, _, self.X_offset, self.y_mean, self.X_scale, self.y_std = standardization(
            np.concatenate(xt, axis=0), np.concatenate(yt, axis=0)
        )

        nlevel = self.nlvl

        # initialize lists
        self.optimal_noise_all = nlevel * [0]
        self.D_all = nlevel * [0]
        self.F_all = nlevel * [0]
        self.p_all = nlevel * [0]
        self.q_all = nlevel * [0]
        self.optimal_rlf_value = nlevel * [0]
        self.optimal_par = nlevel * [{}]
        self.optimal_theta = nlevel * [0]
        self.X_norma_all = [(x - self.X_offset) / self.X_scale for x in X]
        self.y_norma_all = [(f - self.y_mean) / self.y_std for f in y]

    def _new_train_iteration(self, lvl):
        n_samples = self.nt_all
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
                    if np.sum(diff ** 2) != 0.0:
                        self.optimal_noise[i] = np.std(diff, ddof=1) ** 2
                self.optimal_noise = self.optimal_noise / self.nt_reps
                self.optimal_noise_all[lvl] = self.optimal_noise
                self.y_norma = y_norma_unique

                self.X_norma_all[lvl] = self.X_norma
                self.y_norma_all[lvl] = self.y_norma
        else:
            self.optimal_noise = self.options["noise0"]
            self.optimal_noise_all[lvl] = self.optimal_noise

        # Calculate matrix of distances D between samples
        self.D_all[lvl] = cross_distances(self.X_norma)

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
                % (n_samples_F_i, self.p_all[lvl] + self.q_all[lvl])
            )

        # Determine Gaussian Process model parameters
        self.F = self.F_all[lvl]
        D, self.ij = self.D_all[lvl]
        self._lvl = lvl
        self.nt = self.nt_all[lvl]
        self.q = self.q_all[lvl]
        self.p = self.p_all[lvl]
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

    def _new_train_finalize(self, lvl):
        if self.options["eval_noise"] and self.options["optim_var"]:
            X = self.X
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
        if descale:
            X = (X - self.X_offset) / self.X_scale
        f = self._regression_types[self.options["poly"]](X)
        f0 = self._regression_types[self.options["poly"]](X)

        dx = self._differences(X, Y=self.X_norma_all[0])
        d = self._componentwise_distance(dx)

        beta = self.optimal_par[0]["beta"]
        r_ = self._correlation_types[self.options["corr"]](
            self.optimal_theta[0], d
        ).reshape(n_eval, self.nt_all[0])
        gamma = self.optimal_par[0]["gamma"]

        # Scaled predictor
        mu[:, 0] = (np.dot(f, beta) + np.dot(r_, gamma)).ravel()

        # Calculate recursively kriging mean and variance at level i
        for i in range(1, lvl):
            g = self._regression_types[self.options["rho_regr"]](X)
            dx = self._differences(X, Y=self.X_norma_all[i])
            d = self._componentwise_distance(dx)
            r_ = self._correlation_types[self.options["corr"]](
                self.optimal_theta[i], d
            ).reshape(n_eval, self.nt_all[i])
            f = np.vstack((g.T * mu[:, i - 1], f0.T))
            beta = self.optimal_par[i]["beta"]
            gamma = self.optimal_par[i]["gamma"]
            # scaled predictor
            mu[:, i] = (np.dot(f.T, beta) + np.dot(r_, gamma)).ravel()

        # scaled predictor
        if descale:
            mu = mu * self.y_std + self.y_mean

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
        X = (X - self.X_offset) / self.X_scale

        # Calculate kriging mean and variance at level 0
        mu = np.zeros((n_eval, nlevel))
        f = self._regression_types[self.options["poly"]](X)
        f0 = self._regression_types[self.options["poly"]](X)
        dx = self._differences(X, Y=self.X_norma_all[0])
        d = self._componentwise_distance(dx)

        # Get regression function and correlation
        F = self.F_all[0]
        C = self.optimal_par[0]["C"]

        beta = self.optimal_par[0]["beta"]
        Ft = solve_triangular(C, F, lower=True)
        # yt = solve_triangular(C, self.y_norma_all[0], lower=True)
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
        sigma2 = self.optimal_par[0]["sigma2"] / self.y_std ** 2
        MSE[:, 0] = sigma2 * (
            # 1 + self.optimal_noise_all[0] - (r_t ** 2).sum(axis=0) + (u_ ** 2).sum(axis=0)
            1
            - (r_t ** 2).sum(axis=0)
            + (u_ ** 2).sum(axis=0)
        )

        # Calculate recursively kriging variance at level i
        for i in range(1, nlevel):
            F = self.F_all[i]
            C = self.optimal_par[i]["C"]
            g = self._regression_types[self.options["rho_regr"]](X)
            dx = self._differences(X, Y=self.X_norma_all[i])
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
            sigma2 = self.optimal_par[i]["sigma2"] / self.y_std ** 2
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
                    sigma2_rho * MSE[:, i - 1]
                    + Q_ / (2 * (self.nt_all[i] - p - q))
                    # * (1 + self.optimal_noise_all[i] - (r_t ** 2).sum(axis=0))
                    * (1 - (r_t ** 2).sum(axis=0))
                    + sigma2 * (u_ ** 2).sum(axis=0)
                )
            else:
                MSE[:, i] = sigma2_rho * MSE[:, i - 1] + sigma2 * (
                    # 1 + self.optimal_noise_all[i] - (r_t ** 2).sum(axis=0) + (u_ ** 2).sum(axis=0)
                    1
                    - (r_t ** 2).sum(axis=0)
                    + (u_ ** 2).sum(axis=0)
                )

        # scaled predictor
        MSE *= self.y_std ** 2

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
        y : np.ndarray*self.y_std/self.X_scale[kx])
            Derivative values.
        """

        lvl = self.nlvl
        # Initialization

        n_eval, n_features_x = x.shape
        x = (x - self.X_offset) / self.X_scale

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

        df0 = deepcopy(df)

        if self.options["rho_regr"] != "constant":
            raise ValueError(
                "The derivative is only available for regression rho constant"
            )
        # Get pairwise componentwise L1-distances to the input training set
        dx = self._differences(x, Y=self.X_norma_all[0])
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

        theta = self._get_theta(0)
        dy_dx[:, 0] = np.ravel((df_dx - 2 * theta[kx] * np.dot(d_dx * r_, gamma)))

        # Calculate recursively derivative at level i
        for i in range(1, lvl):
            g = self._regression_types[self.options["rho_regr"]](x)
            dx = self._differences(x, Y=self.X_norma_all[i])
            d = self._componentwise_distance(dx)
            r_ = self._correlation_types[self.options["corr"]](
                self.optimal_theta[i], d
            ).reshape(n_eval, self.nt_all[i])
            df = np.vstack((g.T * dy_dx[:, i - 1], df0.T))

            beta = self.optimal_par[i]["beta"]
            gamma = self.optimal_par[i]["gamma"]

            df_dx = np.dot(df.T, beta)
            d_dx = x[:, kx].reshape((n_eval, 1)) - self.X_norma_all[i][:, kx].reshape(
                (1, self.nt_all[i])
            )

            theta = self._get_theta(i)

            # scaled predictor
            dy_dx[:, i] = np.ravel(df_dx - 2 * theta[kx] * np.dot(d_dx * r_, gamma))

        return dy_dx[:, -1] * self.y_std / self.X_scale[kx]

    def _get_theta(self, i):
        return self.optimal_theta[i]

    def _check_param(self):
        """
        Overrides KrgBased implementation
        This function checks some parameters of the model.
        """

        if self.name in ["MFKPLS", "MFKPLSK"]:
            d = self.options["n_comp"]
        else:
            d = self.nx

        if self.options["corr"] == "act_exp":
            raise ValueError("act_exp correlation function must be used with MGP")

        if self.name in ["MFKPLS"]:
            if self.options["corr"] not in ["squar_exp", "abs_exp"]:
                raise ValueError(
                    "MFKPLS only works with a squared exponential or an absolute exponential kernel"
                )
        elif self.name in ["MFKPLSK"]:
            if self.options["corr"] not in ["squar_exp"]:
                raise ValueError(
                    "MFKPLSK only works with a squared exponential kernel (until we prove the contrary)"
                )

        if isinstance(self.options["theta0"], np.ndarray):
            if self.options["theta0"].shape != (self.nlvl, d):
                raise ValueError(
                    "the dimensions of theta0 %s should coincide to the number of dim %s"
                    % (self.options["theta0"].shape, (self.nlvl, d))
                )
        else:
            if len(self.options["theta0"]) != d:
                if len(self.options["theta0"]) == 1:
                    self.options["theta0"] *= np.ones((self.nlvl, d))
                elif len(self.options["theta0"]) == self.nlvl:
                    self.options["theta0"] = np.array(self.options["theta0"]).reshape(
                        -1, 1
                    )
                    self.options["theta0"] *= np.ones((1, d))
                else:
                    raise ValueError(
                        "the length of theta0 (%s) should be equal to the number of dim (%s) or levels of fidelity (%s)."
                        % (len(self.options["theta0"]), d, self.nlvl)
                    )
            else:
                self.options["theta0"] *= np.ones((self.nlvl, 1))

        if len(self.options["noise0"]) != self.nlvl:
            if len(self.options["noise0"]) == 1:
                self.options["noise0"] = self.nlvl * [self.options["noise0"]]
            else:
                raise ValueError(
                    "the length of noise0 (%s) should be equal to the number of levels of fidelity (%s)."
                    % (len(self.options["noise0"]), self.nlvl)
                )

        for i in range(self.nlvl):
            if self.options["use_het_noise"]:
                if len(self.X[i]) == len(np.unique(self.X[i])):
                    if len(self.options["noise0"][i]) != self.nt_all[i]:
                        if len(self.options["noise0"][i]) == 1:
                            self.options["noise0"][i] *= np.ones(self.nt_all[i])
                        else:
                            raise ValueError(
                                "for the level of fidelity %s, the length of noise0 (%s) should be equal to the number of observations (%s)."
                                % (i, len(self.options["noise0"][i]), self.nt_all[i])
                            )
            else:
                if len(self.options["noise0"][i]) != 1:
                    raise ValueError(
                        "for the level of fidelity %s, the length of noise0 (%s) should be equal to one."
                        % (i, len(self.options["noise0"][i]))
                    )
