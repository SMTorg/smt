# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:36:13 2020

@author: Vincent Drouet and Nathalie Bartoli
in order to validate the variance formula for multifidelity on the branin 2D function
Comparisons are based  on the paper Le Gratiet et Cannamela 2015  :
    
Le Gratiet, L., & Cannamela, C. (2015). 
Cokriging-based sequential design strategies using fast cross-validation
techniques for multi-fidelity computer codes. Technometrics, 57(3), 418-427.
https://doi.org/10.1080/00401706.2014.928233
"""


import numpy as np
from smt.applications.mfk import MFK, NestedLHS
from smt.sampling_methods import LHS
import unittest

from smt.utils.sm_test_case import SMTestCase

print_output = True

# %%
# Define the low and high fidelity models
# Example on a 2D problem: Branin  Function
A_corr = 0


# high fidelity model
def HF(point):
    # Expensive Forretal function
    res = (
        (
            point[:, 1]
            - (5.1 / (4 * np.pi**2)) * point[:, 0] ** 2
            + (5 / np.pi) * point[:, 0]
            - 6
        )
        ** 2
        + 10 * (1 - (1 / 8 / np.pi)) * np.cos(point[:, 0])
        + 10
        + 5 * point[:, 0]
    )
    return res


# low fidelity model
def LF(point):
    # Cheap Forretal function
    res = (
        HF(point)
        - (0.5 * A_corr**2 + A_corr + 0.2)
        * (
            point[:, 1]
            - 5.1 / 4 / np.pi**2 * point[:, 0] ** 2
            + 5.0 / np.pi * point[:, 0]
            - 6
        )
        ** 2
    )
    return res


class TestMFK_variance(SMTestCase):
    @staticmethod
    def corr(x1, x2, sm, lvl):
        # Normalization of data with   X_scale and X_offset from sm
        # lvl start to 1
        X_scale = sm.X_scale
        X_offset = sm.X_offset
        X1 = (x1 - X_offset) / X_scale
        X2 = (x2 - X_offset) / X_scale
        thetas = sm.optimal_theta[lvl - 1]
        if sm.options["corr"] == "abs_exp":
            p = 1
        else:
            p = 2
        return np.prod(np.exp(-thetas * np.abs(X1 - X2) ** p))

    @staticmethod
    def hyperparam_LG(sm):
        # computation of the hyperparameters   based on the formula from LeGratiet 2015

        y_t_lf = sm.training_points[0][0][1]
        n_LF = y_t_lf.shape[0]
        C = sm.optimal_par[0]["C"]
        R = C @ C.T
        H = np.ones((n_LF, 1))
        R_inv_H = np.linalg.solve(R, H)
        M = H.T @ R_inv_H
        M_inv = np.linalg.inv(M)
        R_inv_y = np.linalg.solve(R, y_t_lf)
        beta_LG_1 = M_inv @ H.T @ R_inv_y
        sigma2_LG_1 = (
            (y_t_lf - H @ beta_LG_1).T
            @ np.linalg.solve(R, y_t_lf - H @ beta_LG_1)
            / (n_LF - 1)
        )[0, 0]

        y_t_hf = sm.training_points[None][0][1]
        n_HF = y_t_hf.shape[0]
        y_D_l = y_t_lf[-n_HF:, :].reshape(
            -1, 1
        )  # to get y^l-1(D^l) we need n_HF last points of y_t_lf
        H = np.hstack((y_D_l, np.ones((n_HF, 1))))
        C = sm.optimal_par[1]["C"]
        R = C @ C.T
        R_inv_H = np.linalg.solve(R, H)
        M = H.T @ R_inv_H
        M_inv = np.linalg.inv(M)
        R_inv_y = np.linalg.solve(R, y_t_hf)
        rho_beta = M_inv @ H.T @ R_inv_y
        rho_LG = rho_beta[0, 0]
        beta_LG_2 = rho_beta[1, 0]
        sigma2_LG_2 = (
            (y_t_hf - H @ rho_beta).T
            @ np.linalg.solve(R, y_t_hf - H @ rho_beta)
            / (n_HF - 2)
        )[0, 0]
        sigma2_rho_LG = rho_LG**2 + sigma2_LG_2 * M_inv[0, 0]

        return beta_LG_1, sigma2_LG_1, beta_LG_2, sigma2_LG_2, rho_LG, sigma2_rho_LG

    @staticmethod
    def Mu_LG_LG(x, sm):
        # computation of the mean based on the formula from LeGratiet 2015
        # and using beta from LeGratiet 2015
        (
            beta_LG_1,
            sigma2_LG_1,
            beta_LG_2,
            sigma2_LG_2,
            rho_LG,
            sigma2_rho_LG,
        ) = TestMFK_variance.hyperparam_LG(sm)
        C = sm.optimal_par[0]["C"]
        R = C @ C.T
        x_t_lf = sm.training_points[0][0][0]
        y_t_lf = sm.training_points[0][0][1]
        n_LF = x_t_lf.shape[0]
        r_x = np.empty((n_LF, 1))
        for i in range(n_LF):
            r_x[i, 0] = TestMFK_variance.corr(x, x_t_lf[i], sm, 1)
        M = np.linalg.solve(R, y_t_lf - beta_LG_1 * np.ones((n_LF, 1)))
        mu_0 = beta_LG_1 + r_x.T @ M

        x_t_hf = sm.training_points[None][0][0]
        y_t_hf = sm.training_points[None][0][1]
        n_HF = x_t_hf.shape[0]
        C = sm.optimal_par[1]["C"]
        R = C @ C.T
        r_x = np.empty((n_HF, 1))
        for i in range(n_HF):
            r_x[i, 0] = TestMFK_variance.corr(x, x_t_hf[i], sm, 2)
        M = np.linalg.solve(
            R,
            y_t_hf
            - beta_LG_2 * np.ones((n_HF, 1))
            - rho_LG * y_t_lf[-n_HF:, 0, np.newaxis],
        )
        return rho_LG * mu_0 + beta_LG_2 + r_x.T @ M

    @staticmethod
    def Mu_LG_sm(x, sm):
        # computation of the mean based on the formula from LeGratiet 2015
        # and using beta from sm
        beta = sm.optimal_par[0]["beta"]
        C = sm.optimal_par[0]["C"]
        R = C @ C.T
        x_t_lf = sm.training_points[0][0][0]
        y_t_lf = sm.training_points[0][0][1]
        n_LF = x_t_lf.shape[0]
        r_x = np.empty((n_LF, 1))
        for i in range(n_LF):
            r_x[i, 0] = TestMFK_variance.corr(x, x_t_lf[i], sm, 1)
        M = np.linalg.solve(R, y_t_lf - beta * np.ones((n_LF, 1)))
        mu_0 = beta + r_x.T @ M

        x_t_hf = sm.training_points[None][0][0]
        y_t_hf = sm.training_points[None][0][1]
        n_HF = x_t_hf.shape[0]
        beta = sm.optimal_par[1]["beta"]
        rho = beta[0]
        C = sm.optimal_par[1]["C"]
        R = C @ C.T
        r_x = np.empty((n_HF, 1))
        for i in range(n_HF):
            r_x[i, 0] = TestMFK_variance.corr(x, x_t_hf[i], sm, 2)
        M = np.linalg.solve(
            R,
            y_t_hf - beta[1] * np.ones((n_HF, 1)) - rho * y_t_lf[-n_HF:, 0, np.newaxis],
        )
        return rho * mu_0 + beta[1] + r_x.T @ M

    @staticmethod
    def Cov_LG_sm(x1, x2, sm):
        # computation of the covariance  based on the formula from LeGratiet 2015
        # Using the Sigma2 output from sm
        x_t_lf = sm.training_points[0][0][0]
        y_t_lf = sm.training_points[0][0][1]
        n_LF = x_t_lf.shape[0]
        sigma2 = sm.optimal_par[0]["sigma2"]
        C = sm.optimal_par[0]["C"]
        R = C @ C.T
        r_x1 = np.empty((n_LF, 1))
        r_x2 = np.empty((n_LF, 1))
        for i in range(n_LF):
            r_x1[i, 0] = TestMFK_variance.corr(x1, x_t_lf[i], sm, 1)
            r_x2[i, 0] = TestMFK_variance.corr(x2, x_t_lf[i], sm, 1)
        h_x1 = np.ones((1, 1))
        h_x2 = np.ones((1, 1))
        H = np.ones((n_LF, 1))
        hr_1 = np.vstack((h_x1, r_x1))
        hr_2 = np.vstack((h_x2, r_x2))
        M1 = np.hstack((np.zeros((1, 1)), H.T))
        M2 = np.hstack((H, R))
        M3 = np.vstack((M1, M2))
        M4 = np.linalg.solve(M3, hr_2)  # M4 = M3^-1 @ hr_2
        k_0 = sigma2 * (
            TestMFK_variance.corr(x1, x2, sm, 1) - hr_1.T @ M4
        )  # covariance of level 0

        x_t_hf = sm.training_points[None][0][0]
        n_HF = x_t_hf.shape[0]
        sigma2 = sm.optimal_par[1]["sigma2"]
        (var_all_pred, sigma2_rho) = sm.predict_variances_all_levels(x1)
        sigma2_rho = sigma2_rho[0]
        C = sm.optimal_par[1]["C"]
        R = C @ C.T
        r_x1 = np.empty((n_HF, 1))
        r_x2 = np.empty((n_HF, 1))
        for i in range(n_HF):
            r_x1[i, 0] = TestMFK_variance.corr(x1, x_t_hf[i], sm, 2)
            r_x2[i, 0] = TestMFK_variance.corr(x2, x_t_hf[i], sm, 2)
        mu_x1 = sm._predict_intermediate_values(x1, 1).reshape(-1, 1)
        mu_x2 = sm._predict_intermediate_values(x2, 1).reshape(-1, 1)
        h_x1 = np.vstack((mu_x1, np.ones((1, 1))))
        h_x2 = np.vstack((mu_x2, np.ones((1, 1))))
        y_D_l = y_t_lf[-n_HF:, :].reshape(
            -1, 1
        )  # to get y^l-1(D^l) we need n_HF last points of y_t_lf
        H = np.hstack((y_D_l, np.ones((n_HF, 1))))
        hr_1 = np.vstack((h_x1, r_x1))
        hr_2 = np.vstack((h_x2, r_x2))
        M1 = np.hstack((np.zeros((2, 2)), H.T))
        M2 = np.hstack((H, R))
        M3 = np.vstack((M1, M2))
        M4 = np.linalg.solve(M3, hr_2)  # M4 = M3^-1 @ hr_2
        k_1 = sigma2_rho * k_0 + sigma2 * (
            TestMFK_variance.corr(x1, x2, sm, 2) - hr_1.T @ M4
        )
        return k_0, k_1

    @staticmethod
    def Cov_LG_LG(x1, x2, sm):
        # computation of the covariance  based on the formula from LeGratiet 2015
        # Using the Sigma2 output from Le Gratiet paper
        x_t_lf = sm.training_points[0][0][0]
        y_t_lf = sm.training_points[0][0][1]
        n_LF = x_t_lf.shape[0]
        (
            beta_LG_1,
            sigma2_LG_1,
            beta_LG_2,
            sigma2_LG_2,
            rho_LG,
            sigma2_rho_LG,
        ) = TestMFK_variance.hyperparam_LG(sm)
        sigma2 = sigma2_LG_1
        C = sm.optimal_par[0]["C"]
        R = C @ C.T
        r_x1 = np.empty((n_LF, 1))
        r_x2 = np.empty((n_LF, 1))
        for i in range(n_LF):
            r_x1[i, 0] = TestMFK_variance.corr(x1, x_t_lf[i], sm, 1)
            r_x2[i, 0] = TestMFK_variance.corr(x2, x_t_lf[i], sm, 1)
        h_x1 = np.ones((1, 1))
        h_x2 = np.ones((1, 1))
        H = np.ones((n_LF, 1))
        hr_1 = np.vstack((h_x1, r_x1))
        hr_2 = np.vstack((h_x2, r_x2))
        M1 = np.hstack((np.zeros((1, 1)), H.T))
        M2 = np.hstack((H, R))
        M3 = np.vstack((M1, M2))
        M4 = np.linalg.solve(M3, hr_2)  # M4 = M3^-1 @ hr_2
        k_0 = sigma2 * (
            TestMFK_variance.corr(x1, x2, sm, 1) - hr_1.T @ M4
        )  # covariance of level 0

        x_t_hf = sm.training_points[None][0][0]
        n_HF = x_t_hf.shape[0]
        sigma2 = sigma2_LG_2
        C = sm.optimal_par[1]["C"]
        R = C @ C.T
        r_x1 = np.empty((n_HF, 1))
        r_x2 = np.empty((n_HF, 1))
        for i in range(n_HF):
            r_x1[i, 0] = TestMFK_variance.corr(x1, x_t_hf[i], sm, 2)
            r_x2[i, 0] = TestMFK_variance.corr(x2, x_t_hf[i], sm, 2)
        mu_x1 = sm._predict_intermediate_values(x1, 1).reshape(-1, 1)
        mu_x2 = sm._predict_intermediate_values(x2, 1).reshape(-1, 1)
        h_x1 = np.vstack((mu_x1, np.ones((1, 1))))
        h_x2 = np.vstack((mu_x2, np.ones((1, 1))))
        y_D_l = y_t_lf[-n_HF:, :].reshape(
            -1, 1
        )  # to get y^l-1(D^l) we need n_HF last points of y_t_lf
        H = np.hstack((y_D_l, np.ones((n_HF, 1))))
        hr_1 = np.vstack((h_x1, r_x1))
        hr_2 = np.vstack((h_x2, r_x2))
        M1 = np.hstack((np.zeros((2, 2)), H.T))
        M2 = np.hstack((H, R))
        M3 = np.vstack((M1, M2))
        M4 = np.linalg.solve(M3, hr_2)  # M4 = M3^-1 @ hr_2

        k_1 = sigma2_rho_LG * k_0 + sigma2 * (
            TestMFK_variance.corr(x1, x2, sm, 2) - hr_1.T @ M4
        )
        return k_0, k_1

    @staticmethod
    def verif_hyperparam(sm, x_test_LHS):
        # get the hyperparameters from sm
        beta_sm_1 = sm.optimal_par[0]["beta"][0, 0]
        sigma2_sm_1 = sm.optimal_par[0]["sigma2"][0]
        rho_beta_sm = sm.optimal_par[1]["beta"]
        beta_sm_2 = rho_beta_sm[1, 0]
        sigma2_sm_2 = sm.optimal_par[1]["sigma2"][0]
        rho_sm = rho_beta_sm[0, 0]
        (var_all_pred, sigma2_rho) = sm.predict_variances_all_levels(x_test_LHS)
        sigma2_rho_sm = sigma2_rho[0]

        (
            beta_LG_1,
            sigma2_LG_1,
            beta_LG_2,
            sigma2_LG_2,
            rho_LG,
            sigma2_rho_LG,
        ) = TestMFK_variance.hyperparam_LG(sm)

        return (
            beta_sm_1,
            sigma2_sm_1,
            beta_sm_2,
            sigma2_sm_2,
            rho_sm,
            sigma2_rho_sm,
            beta_LG_1,
            sigma2_LG_1,
            beta_LG_2,
            sigma2_LG_2,
            rho_LG,
            sigma2_rho_LG,
        )

    def test_mfk_variance(self):
        # To create the doe
        # dim = 2
        nlevel = 2
        ub0 = 10.0
        ub1 = 15.0
        lb0 = -5.0
        lb1 = 0.0
        xlimits = np.array([[lb0, ub0], [lb1, ub1]])

        # Constants
        n_HF = 5  # number of high fidelity points (number of low fi is twice)
        xdoes = NestedLHS(nlevel=nlevel, xlimits=xlimits)
        x_t_lf, x_t_hf = xdoes(n_HF)

        # Evaluate the HF and LF functions
        y_t_lf = LF(x_t_lf)
        y_t_hf = HF(x_t_hf)

        sm = MFK(
            theta0=x_t_hf.shape[1] * [1e-2],
            print_global=False,
            rho_regr="constant",
        )

        # low-fidelity dataset names being integers from 0 to level-1
        sm.set_training_values(x_t_lf, y_t_lf, name=0)
        # high-fidelity dataset without name
        sm.set_training_values(x_t_hf, y_t_hf)
        # train the model
        sm.train()

        # Validation set
        # for validation with LHS
        ntest = 1
        sampling = LHS(xlimits=xlimits)
        x_test_LHS = sampling(ntest)
        # y_test_LHS = HF(x_test_LHS)

        # compare the mean value between different formula
        if print_output:
            print("Mu sm  : {}".format(sm.predict_values(x_test_LHS)[0, 0]))
            print(
                "Mu LG_sm : {}".format(TestMFK_variance.Mu_LG_sm(x_test_LHS, sm)[0, 0])
            )
            print(
                "Mu LG_LG : {}".format(TestMFK_variance.Mu_LG_LG(x_test_LHS, sm)[0, 0])
            )

        # self.assertAlmostEqual(
        #     TestMFK_variance.Mu_LG_sm(x_test_LHS, sm)[0, 0],
        #     TestMFK_variance.Mu_LG_LG(x_test_LHS, sm)[0, 0],
        #     delta=1,
        # )
        self.assertAlmostEqual(
            sm.predict_values(x_test_LHS)[0, 0],
            TestMFK_variance.Mu_LG_LG(x_test_LHS, sm)[0, 0],
            delta=1,
        )

        # compare the variance value between different formula
        (k_0_LG_sm, k_1_LG_sm) = TestMFK_variance.Cov_LG_sm(x_test_LHS, x_test_LHS, sm)
        (k_0_LG_LG, k_1_LG_LG) = TestMFK_variance.Cov_LG_LG(x_test_LHS, x_test_LHS, sm)
        k_0_sm = sm.predict_variances_all_levels(x_test_LHS)[0][0, 0]
        k_1_sm = sm.predict_variances_all_levels(x_test_LHS)[0][0, 1]

        if print_output:
            print("Level 0")
            print("Var sm  : {}".format(k_0_sm))
            print("Var LG_sm : {}".format(k_0_LG_sm[0, 0]))
            print("Var LG_LG : {}".format(k_0_LG_LG[0, 0]))

            print("Level 1")
            print("Var sm  : {}".format(k_1_sm))
            print("Var LG_sm : {}".format(k_1_LG_sm[0, 0]))
            print("Var LG_LG : {}".format(k_1_LG_LG[0, 0]))

        # for level 0
        self.assertAlmostEqual(k_0_sm, k_0_LG_sm[0, 0], delta=1)
        self.assertAlmostEqual(k_0_LG_sm[0, 0], k_0_LG_LG[0, 0], delta=1)
        # for level 1
        self.assertAlmostEqual(k_1_sm, k_1_LG_sm[0, 0], delta=1)
        self.assertAlmostEqual(k_1_LG_sm[0, 0], k_1_LG_LG[0, 0], delta=1)

        (
            beta_sm_1,
            sigma2_sm_1,
            beta_sm_2,
            sigma2_sm_2,
            rho_sm,
            sigma2_rho_sm,
            beta_LG_1,
            sigma2_LG_1,
            beta_LG_2,
            sigma2_LG_2,
            rho_LG,
            sigma2_rho_LG,
        ) = TestMFK_variance.verif_hyperparam(sm, x_test_LHS)
        if print_output:
            print("Hyperparameters")
            print("rho_sm : {}".format(rho_sm))
            print("rho_LG : {}".format(rho_LG))
            print("sigma2_rho_sm : {}".format(sigma2_rho_sm[0]))
            print("sigma2_rho_LG : {}".format(sigma2_rho_LG))
            print("beta_sm_1 : {}".format(beta_sm_1))
            print("beta_LG_1 : {}".format(beta_LG_1[0, 0]))
            print("beta_sm_2 : {}".format(beta_sm_2))
            print("beta_LG_2 : {}".format(beta_LG_2))
            print("sigma2_sm_1 : {}".format(sigma2_sm_1))
            print("sigma2_LG_1 : {}".format(sigma2_LG_1))
            print("sigma2_sm_2 : {}".format(sigma2_sm_2))
            print("sigma2_LG_2 : {}".format(sigma2_LG_2))


if __name__ == "__main__":
    unittest.main()
