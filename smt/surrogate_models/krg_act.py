"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

from __future__ import division
import warnings
import numpy as np
from scipy import linalg, optimize
from sklearn.metrics.pairwise import manhattan_distances

from smt.surrogate_models.krg_based import KrgBased
from smt.utils.kriging_utils import componentwise_distance

"""
The Active kriging class.
"""


class AKRG(KrgBased):
    def _initialize(self):
        super(AKRG, self)._initialize()
        declare = self.options.declare
        declare("n_comp", 1, types=int, desc="Number of active dimensions")
        self.options["hyper_opt"] = "L-BFGS-B"
        self.options["corr"] = "act_exp"
        self.options["theta0"] = [0.5]
        self.name = "Active Kriging"

    def _componentwise_distance(self, dx, opt=0):
        d = componentwise_distance(dx, self.options["corr"], self.nx)
        return d

    def predict_variance(self, x):
        dy = self._predict_value_derivatives_hyper(x)
        dMSE, MSE = self._predict_variance_derivatives_hyper(x)

        arg_1 = np.dot(dy.T, self.sigma_R.dot(dy))
        arg_1 = np.einsum("ii->i", arg_1)

        arg_2 = np.dot(dMSE.T, self.sigma_R.dot(dMSE))
        arg_2 = np.einsum("ii->i", arg_2)

        AMSE = np.zeros(x.shape[0])

        AMSE[MSE != 0] = (
            (4.0 / 3.0) * MSE[MSE != 0]
            + arg_1[MSE != 0]
            + (1.0 / (3.0 * MSE[MSE != 0])) * arg_2[MSE != 0]
        )

        AMSE[AMSE < 0.0] = 0.0

        return AMSE

    def _predict_value_derivatives_hyper(self, x):
        """
        Compute the value derivatives over the optimal hyperparameters for a set of points

        Parameters
        ----------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        dy : np.ndarray
            Derivative values over the optimal hyperparameters.
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
        # Compute the regression function
        f = self._regression_types[self.options["poly"]](x)

        dy = np.zeros((len(self.optimal_theta), n_eval))

        gamma = self.optimal_par["gamma"]
        Rinv_dR_gamma = self.optimal_par["Rinv_dR_gamma"]
        Rinv_dmu = self.optimal_par["Rinv_dmu"]

        for theta in range(len(self.optimal_theta)):
            drdtheta = self._correlation_types[self.options["corr"]](
                self.optimal_theta, d, grad_ind=theta
            ).reshape(n_eval, self.nt)

            dbetadtheta = self.optimal_par["dbeta_all"][theta]

            drdtheta_gamma = drdtheta.dot(gamma)

            dy_theta = (
                f.dot(dbetadtheta)
                + drdtheta.dot(gamma)
                - r.dot(Rinv_dR_gamma[theta] + Rinv_dmu[theta])
            )

            dy[theta, :] = dy_theta[:, 0]

        return dy

    def _predict_variance_derivatives_hyper(self, x):
        """
        Compute the variance derivatives over the optimal hyperparameters for a set of points

        Parameters
        ----------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        dMSE : np.ndarray
            Derivative variance over the optimal hyperparameters.

        """
        # Initialization
        n_eval, n_features_x = x.shape
        x = (x - self.X_mean) / self.X_std
        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(x, Y=self.X_norma.copy(), sum_over_features=False)
        d = self._componentwise_distance(dx)

        # Compute the correlation function
        r = (
            self._correlation_types[self.options["corr"]](self.optimal_theta, d)
            .reshape(n_eval, self.nt)
            .T
        )
        f = self._regression_types[self.options["poly"]](x).T

        C = self.optimal_par["C"]
        G = self.optimal_par["G"]
        Ft = self.optimal_par["Ft"]
        sigma2 = self.optimal_par["sigma2"]

        rt = linalg.solve_triangular(C, r, lower=True)

        F_Rinv_r = np.dot(Ft.T, rt)

        u = linalg.solve_triangular(G.T, f - F_Rinv_r)

        MSE = self.optimal_par["sigma2"] * (
            1.0 - (rt ** 2.0).sum(axis=0) + (u ** 2.0).sum(axis=0)
        )
        # Mean Squared Error might be slightly negative depending on
        # machine precision: force to zero!
        MSE[MSE < 0.0] = 0.0

        Ginv_u = np.linalg.solve(G, u)
        Rinv_F = np.linalg.solve(C.T, Ft)
        Rinv_r = np.linalg.solve(C.T, rt)
        Rinv_F_Ginv_u = Rinv_F.dot(Ginv_u)

        dMSE = np.zeros((len(self.optimal_theta), n_eval))

        dr_all = self.optimal_par["dr"]
        dsigma = self.optimal_par["dsigma"]

        for theta in range(len(self.optimal_theta)):
            drdtheta = (
                self._correlation_types[self.options["corr"]](
                    self.optimal_theta, d, grad_ind=theta
                )
                .reshape(n_eval, self.nt)
                .T
            )

            dRdtheta = np.zeros((self.nt, self.nt))
            dRdtheta[self.ij[:, 0], self.ij[:, 1]] = dr_all[theta][:, 0]
            dRdtheta[self.ij[:, 1], self.ij[:, 0]] = dr_all[theta][:, 0]

            # Compute du2dtheta

            dRdtheta_Rinv_F_Ginv_u = dRdtheta.dot(Rinv_F_Ginv_u)

            r_Rinv_dRdtheta_Rinv_F_Ginv_u = np.dot(Rinv_r.T, dRdtheta_Rinv_F_Ginv_u)

            drdtheta_Rinv_F_Ginv_u = np.dot(drdtheta.T, Rinv_F_Ginv_u)

            u_Ginv_F_Rinv_dRdtheta_Rinv_F_Ginv_u = np.dot(
                Rinv_F_Ginv_u.T, dRdtheta_Rinv_F_Ginv_u
            )

            du2dtheta = np.einsum(
                "ii->i",
                r_Rinv_dRdtheta_Rinv_F_Ginv_u
                + r_Rinv_dRdtheta_Rinv_F_Ginv_u.T
                - drdtheta_Rinv_F_Ginv_u
                - drdtheta_Rinv_F_Ginv_u.T
                + u_Ginv_F_Rinv_dRdtheta_Rinv_F_Ginv_u,
            )
            du2dtheta = np.atleast_2d(du2dtheta)

            # Compute drt2dtheta

            drdtheta_Rinv_r = np.dot(drdtheta.T, Rinv_r)

            r_Rinv_dRdtheta_Rinv_r = np.dot(Rinv_r.T, dRdtheta.dot(Rinv_r))

            drt2dtheta = np.einsum(
                "ii->i", drdtheta_Rinv_r + drdtheta_Rinv_r.T - r_Rinv_dRdtheta_Rinv_r
            )
            drt2dtheta = np.atleast_2d(drt2dtheta)

            dMSE[theta] = dsigma[theta] * MSE / sigma2 + sigma2 * (
                -drt2dtheta + du2dtheta
            )

        return dMSE, MSE
