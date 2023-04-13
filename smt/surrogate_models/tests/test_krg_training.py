#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:20:29 2020

@author: ninamoello
"""

from __future__ import print_function, division
import numpy as np
import unittest
from smt.utils.sm_test_case import SMTestCase
from smt.utils.kriging import (
    pow_exp,
    abs_exp,
    squar_exp,
    act_exp,
    cross_distances,
    componentwise_distance,
    matern52,
    matern32,
)
from smt.utils.misc import standardization
from smt.sampling_methods.lhs import LHS
from smt.surrogate_models import KRG, MGP

print_output = False


class Test(SMTestCase):
    def setUp(self):
        eps = 1e-8
        xlimits = np.asarray([[0, 1], [0, 1]])
        self.random = np.random.RandomState(42)
        lhs = LHS(xlimits=xlimits, random_state=self.random)
        X = lhs(8)
        y = LHS(xlimits=np.asarray([[0, 1]]), random_state=self.random)(8)
        X_norma, y_norma, X_offset, y_mean, X_scale, y_std = standardization(X, y)
        D, ij = cross_distances(X_norma)
        theta = self.random.rand(2)
        corr_str = [
            "pow_exp",
            "abs_exp",
            "squar_exp",
            "act_exp",
            "matern32",
            "matern52",
        ]
        corr_def = [pow_exp, abs_exp, squar_exp, act_exp, matern32, matern52]
        power_val = {
            "pow_exp": 1.9,
            "abs_exp": 1.0,
            "squar_exp": 2.0,
            "act_exp": 1.0,
            "matern32": 1.0,
            "matern52": 1.0,
        }

        self.eps = eps
        self.X = X
        self.y = y
        (
            self.X_norma,
            self.y_norma,
            self.X_offset,
            self.y_mean,
            self.X_scale,
            self.y_std,
        ) = (
            X_norma,
            y_norma,
            X_offset,
            y_mean,
            X_scale,
            y_std,
        )
        self.D, self.ij = D, ij
        self.theta = theta
        self.corr_str = corr_str
        self.corr_def = corr_def
        self.power_val = power_val

        def test_noise_estimation(self):
            xt = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
            yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])
            sm = KRG(hyper_opt="Cobyla", eval_noise=True, noise0=[1e-4])

            sm.set_training_values(xt, yt)
            sm.train()
            self.assert_error(np.array(sm.optimal_theta), np.array([1.6]), 1e-1, 1e-1)

    def test_corr_derivatives(self):
        for ind, corr in enumerate(self.corr_def):  # For every kernel
            # self.corr_str[ind] = self.corr_def[ind]
            D = componentwise_distance(
                self.D,
                self.corr_str[ind],
                self.X.shape[1],
                self.power_val[self.corr_str[ind]],
            )

            k = corr(self.theta, D)
            K = np.eye(self.X.shape[0])
            K[self.ij[:, 0], self.ij[:, 1]] = k[:, 0]
            K[self.ij[:, 1], self.ij[:, 0]] = k[:, 0]
            grad_norm_all = []
            diff_norm_all = []
            ind_theta = []
            for i, theta_i in enumerate(self.theta):
                eps_theta = np.zeros(self.theta.shape)
                eps_theta[i] = self.eps

                k_dk = corr(self.theta + eps_theta, D)

                K_dk = np.eye(self.X.shape[0])
                K_dk[self.ij[:, 0], self.ij[:, 1]] = k_dk[:, 0]
                K_dk[self.ij[:, 1], self.ij[:, 0]] = k_dk[:, 0]

                grad_eps = (K_dk - K) / self.eps

                dk = corr(self.theta, D, grad_ind=i)
                dK = np.zeros((self.X.shape[0], self.X.shape[0]))
                dK[self.ij[:, 0], self.ij[:, 1]] = dk[:, 0]
                dK[self.ij[:, 1], self.ij[:, 0]] = dk[:, 0]
                grad_norm_all.append(np.linalg.norm(dK))
                diff_norm_all.append(np.linalg.norm(grad_eps))
                ind_theta.append(r"$x_%d$" % i)
            self.assert_error(
                np.array(grad_norm_all), np.array(diff_norm_all), 1e-5, 1e-5
            )  # from utils/smt_test_case.py

    def test_corr_hessian(self):
        for ind, corr in enumerate(self.corr_def):  # For every kernel
            # self.corr_str[ind] = self.corr_def[ind]
            D = componentwise_distance(
                self.D,
                self.corr_str[ind],
                self.X.shape[1],
                self.power_val[self.corr_str[ind]],
            )

            grad_norm_all = []
            diff_norm_all = []
            for i, theta_i in enumerate(self.theta):
                k = corr(self.theta, D, grad_ind=i)

                K = np.eye(self.X.shape[0])
                K[self.ij[:, 0], self.ij[:, 1]] = k[:, 0]
                K[self.ij[:, 1], self.ij[:, 0]] = k[:, 0]
                for j, omega_j in enumerate(self.theta):
                    eps_omega = np.zeros(self.theta.shape)
                    eps_omega[j] = self.eps

                    k_dk = corr(self.theta + eps_omega, D, grad_ind=i)

                    K_dk = np.eye(self.X.shape[0])
                    K_dk[self.ij[:, 0], self.ij[:, 1]] = k_dk[:, 0]
                    K_dk[self.ij[:, 1], self.ij[:, 0]] = k_dk[:, 0]

                    grad_eps = (K_dk - K) / self.eps

                    dk = corr(self.theta, D, grad_ind=i, hess_ind=j)
                    dK = np.zeros((self.X.shape[0], self.X.shape[0]))
                    dK[self.ij[:, 0], self.ij[:, 1]] = dk[:, 0]
                    dK[self.ij[:, 1], self.ij[:, 0]] = dk[:, 0]

                    grad_norm_all.append(np.linalg.norm(dK))
                    diff_norm_all.append(np.linalg.norm(grad_eps))

            self.assert_error(
                np.array(grad_norm_all), np.array(diff_norm_all), 1e-5, 1e-5
            )  # from utils/smt_test_case.py

    def test_likelihood_derivatives(self):
        for corr_str in [
            "pow_exp",
            "abs_exp",
            "squar_exp",
            "act_exp",
            "matern32",
            "matern52",
        ]:  # For every kernel
            for poly_str in ["constant", "linear", "quadratic"]:  # For every method
                if corr_str == "act_exp":
                    kr = MGP(print_global=False)
                    theta = self.random.rand(4)
                else:
                    kr = KRG(print_global=False)
                    theta = self.theta
                kr.options["poly"] = poly_str
                kr.options["corr"] = corr_str
                kr.options["pow_exp_power"] = self.power_val[corr_str]
                kr.set_training_values(self.X, self.y)
                kr.train()

                grad_red, dpar = kr._reduced_likelihood_gradient(theta)
                red, par = kr._reduced_likelihood_function(theta)

                grad_norm_all = []
                diff_norm_all = []
                ind_theta = []
                for i, theta_i in enumerate(theta):
                    eps_theta = theta.copy()
                    eps_theta[i] = eps_theta[i] + self.eps

                    red_dk, par_dk = kr._reduced_likelihood_function(eps_theta)
                    dred_dk = (red_dk - red) / self.eps

                    grad_norm_all.append(grad_red[i])
                    diff_norm_all.append(float(dred_dk))
                    ind_theta.append(r"$x_%d$" % i)

                grad_norm_all = np.atleast_2d(grad_norm_all)
                diff_norm_all = np.atleast_2d(diff_norm_all).T
                self.assert_error(
                    grad_norm_all, diff_norm_all, atol=1e-5, rtol=1e-3
                )  # from utils/smt_test_case.py

    def test_likelihood_hessian(self):
        for corr_str in [
            "pow_exp",
            "abs_exp",
            "squar_exp",
            "act_exp",
            "matern32",
            "matern52",
        ]:  # For every kernel
            for poly_str in ["constant", "linear", "quadratic"]:  # For every method
                if corr_str == "act_exp":
                    kr = MGP(print_global=False)
                    theta = self.random.rand(4)
                else:
                    kr = KRG(print_global=False)
                    theta = self.theta
                kr.options["poly"] = poly_str
                kr.options["corr"] = corr_str
                kr.options["pow_exp_power"] = self.power_val[corr_str]
                kr.set_training_values(self.X, self.y)
                kr.train()
                grad_red, dpar = kr._reduced_likelihood_gradient(theta)

                hess, hess_ij, _ = kr._reduced_likelihood_hessian(theta)
                Hess = np.zeros((theta.shape[0], theta.shape[0]))
                Hess[hess_ij[:, 0], hess_ij[:, 1]] = hess[:, 0]
                Hess[hess_ij[:, 1], hess_ij[:, 0]] = hess[:, 0]

                grad_norm_all = []
                diff_norm_all = []
                ind_theta = []
                for j, omega_j in enumerate(theta):
                    eps_omega = theta.copy()
                    eps_omega[j] += self.eps

                    grad_red_eps, _ = kr._reduced_likelihood_gradient(eps_omega)
                    for i, theta_i in enumerate(theta):
                        hess_eps = (grad_red_eps[i] - grad_red[i]) / self.eps

                        grad_norm_all.append(
                            np.linalg.norm(Hess[i, j]) / np.linalg.norm(Hess)
                        )
                        diff_norm_all.append(
                            np.linalg.norm(hess_eps) / np.linalg.norm(Hess)
                        )
                        ind_theta.append(r"$x_%d,x_%d$" % (j, i))
                self.assert_error(
                    np.array(grad_norm_all),
                    np.array(diff_norm_all),
                    atol=1e-5,
                    rtol=1e-3,
                )  # from utils/smt_test_case.py

    def test_variance_derivatives(self):
        for corr_str in [
            "abs_exp",
            "squar_exp",
            "matern32",
            "matern52",
            "pow_exp",
        ]:
            kr = KRG(print_global=False)
            kr.options["poly"] = "constant"
            kr.options["corr"] = corr_str
            kr.options["pow_exp_power"] = self.power_val[corr_str]
            kr.set_training_values(self.X, self.y)
            kr.train()

            e = 1e-6
            xa = self.random.random()
            xb = self.random.random()
            x_valid = [[xa, xb], [xa + e, xb], [xa - e, xb], [xa, xb + e], [xa, xb - e]]

            y_predicted = kr.predict_variances(np.array(x_valid))
            y_jacob = np.zeros((2, 5))

            for i in range(np.shape(x_valid)[0]):
                l0 = kr.predict_variance_derivatives(np.atleast_2d(x_valid[i]), 0)[0]
                l1 = kr.predict_variance_derivatives(np.atleast_2d(x_valid[i]), 1)[0]
                y_jacob[0, i] = l0
                y_jacob[1, i] = l1

            diff_g = (y_predicted[1] - y_predicted[2]) / (2 * e)
            diff_d = (y_predicted[3] - y_predicted[4]) / (2 * e)

            jac_rel_error1 = abs((y_jacob[0][0] - diff_g) / y_jacob[0][0])
            self.assert_error(jac_rel_error1, 1e-3, atol=0.01, rtol=0.01)

            jac_rel_error2 = abs((y_jacob[1][0] - diff_d) / y_jacob[1][1])
            self.assert_error(jac_rel_error2, 1e-3, atol=0.01, rtol=0.01)


if __name__ == "__main__":
    print_output = True
    unittest.main()
