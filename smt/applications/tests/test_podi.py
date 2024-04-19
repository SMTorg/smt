"""
Author: Hugo Reimeringer <hugo.reimeringer@onera.fr>
"""

import unittest
from smt.utils.sm_test_case import SMTestCase

import numpy as np
from scipy import special
from smt.sampling_methods import LHS
from smt.applications import PODI
import warnings

warnings.simplefilter("ignore")

# -------------------- Functions for basis/coeff --------------------
# Functions for coefficients


# -------------------- Functions for orthogonal basis --------------------


def cos_coeff(i, x):
    a = 2 * i // 2 - 1
    return a * x[:, 0] * np.cos(i * x[:, 0])


def Legendre(i, t):
    return special.legendre(i)(t)


class Test(SMTestCase):
    def setUp(self):
        self.nt = 40
        xlimits = np.array([[0, 4]])
        sampling = LHS(xlimits=xlimits)
        self.xt = sampling(self.nt)

        self.ny = 100
        self.t = np.linspace(-1, 1, self.ny)

        self.n_modes_test = 10

        self.nn = 15
        self.xn = sampling(self.nn)

        self.nv = 10 * self.nt
        self.xv = sampling(self.nv)

        self.x = np.concatenate((self.xt, self.xv))

        def pb_1d(x):
            u0 = np.zeros((1, self.ny))

            alpha = np.zeros((x.shape[0], self.n_modes_test))
            for i in range(self.n_modes_test):
                alpha[:, i] = cos_coeff(i, x)

            V_init = np.zeros((self.ny, self.n_modes_test))
            for i in range(self.n_modes_test):
                V_init[:, i] = Legendre(i, self.t)

            V = Test.gram_schmidt(V_init.T).T  # orthonormal basis
            database = u0 + np.dot(alpha, V.T)
            self.basis_original = V.T

            return database

        self.full_database = pb_1d(self.x)
        self.database = self.full_database[: self.nt]

    @staticmethod
    def gram_schmidt(polynomials):
        basis = np.zeros_like(polynomials)
        for i in range(len(polynomials)):
            basis[i] = polynomials[i]
            for j in range(i):
                basis[i] -= (
                    np.dot(polynomials[i], basis[j])
                    / np.dot(basis[j], basis[j])
                    * basis[j]
                )
            basis[i] /= np.linalg.norm(basis[i])
        return basis

    @staticmethod
    def _check_projection(basis_original, basis_SVD):
        norm_proj = np.zeros(len(basis_SVD))
        norm_residue = np.zeros(len(basis_SVD))

        projection = np.dot(basis_SVD, basis_original.T).dot(basis_original)

        for i in range(len(projection)):
            proj = projection[i]
            norm_residue[i] = np.linalg.norm(basis_SVD[i] - proj)
            norm_proj[i] = np.linalg.norm(proj)
        return norm_proj, norm_residue

    def test_predict(self):
        sm = PODI()

        sm.compute_pod(self.database, tol=1)
        sm.set_interp_options("KRG")
        sm.set_training_values(self.xt)
        sm.train()

        mean_xt = sm.predict_values(self.xt)
        var_xt = sm.predict_variances(self.xt)
        diff = self.database - mean_xt
        rms_error = [np.sqrt(np.mean(diff[i] ** 2)) for i in range(diff.shape[0])]

        np.testing.assert_allclose(rms_error, np.zeros(self.nt), atol=1e-6)
        # np.testing.assert_allclose(self.database, mean_xt, atol = 1e-6)
        np.testing.assert_allclose(var_xt, np.zeros(var_xt.shape), atol=1e-6)

        mean_xn = sm.predict_values(self.xn)
        var_xn = sm.predict_variances(self.xn)
        deriv_xn = sm.predict_derivatives(self.xn, 0)

        assert mean_xn.shape == (self.nn, self.ny)
        assert var_xn.shape == (self.nn, self.ny)
        assert deriv_xn.shape == (self.nn, self.ny)

        mean_xv = sm.predict_values(self.xv)

        diff = self.full_database[self.nt :] - mean_xv
        rms_error = [np.sqrt(np.mean(diff[i] ** 2)) for i in range(diff.shape[0])]

        np.testing.assert_allclose(rms_error, np.zeros(self.nv), atol=1e-2)

    def test_set_options(self):
        sm = PODI()
        sm.compute_pod(self.database, n_modes=1)
        options = [
            {
                "poly": "quadratic",
                "corr": "matern32",
                "pow_exp_power": 0.38,
                "theta0": [1e-1],
            }
        ]
        sm.set_interp_options("KRG", options)

        sm_list = sm.get_interp_coef()
        for interp_coeff in sm_list:
            for key in options[0].keys():
                assert interp_coeff.options[key] == options[0][key]

    def test_pod(self):
        sm = PODI()

        sm.compute_pod(self.database, tol=1)
        assert sm.get_ev_ratio() == 1

        n_modes = sm.get_n_modes()
        assert n_modes <= self.n_modes_test and n_modes > 0

        basis_SVD = sm.get_left_basis()
        assert basis_SVD.shape == (self.nt, self.ny)

        singular_values = sm.get_singular_values()
        assert len(singular_values) == self.nt
        np.testing.assert_allclose(
            singular_values[n_modes:], np.zeros(self.nt - n_modes), atol=1e-6
        )

        norm_proj, residue = Test._check_projection(self.basis_original, basis_SVD)
        np.testing.assert_allclose(norm_proj[:n_modes], np.ones(n_modes), atol=1e-6)
        np.testing.assert_allclose(residue[:n_modes], np.zeros(n_modes), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
