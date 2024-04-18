"""
Author: Hugo Reimeringer <hugo.reimeringer@onera.fr>
"""

import unittest
from smt.utils.sm_test_case import SMTestCase

import numpy as np
from scipy import special
from smt.sampling_methods import LHS
from smt.surrogate_models import PODGP


def cos_coeff(i, x):
    a = 2 * i // 2 - 1
    return a * x[:, 0] * np.cos(i * x[:, 0])


def Legendre(i, t):
    return special.legendre(i)(t)


class Test(SMTestCase):
    def setUp(self):
        self.nt = 20
        xlimits = np.array([[0, 4]])
        sampling = LHS(xlimits=xlimits, random_state=123)
        self.xt = sampling(self.nt)

        self.ny = 100
        self.t = np.linspace(-1, 1, self.ny)

        self.n_mods_test = 10

        self.nn = 15
        self.xn = sampling(self.nn)

        def pb(x):
            u0 = np.zeros((1, self.ny))
            # True function for validation
            alpha = np.zeros((x.shape[0], self.n_mods_test))
            for i in range(self.n_mods_test):
                alpha[:, i] = cos_coeff(i, x)

            V_init = np.zeros((self.ny, self.n_mods_test))
            for i in range(self.n_mods_test):
                V_init[:, i] = Legendre(i, self.t)

            V = Test.gram_schmidt(V_init.T).T  # orthonormal basis
            database = u0 + np.dot(alpha, V.T)
            self.basis_original = V

            return database

        self.database = pb(self.xt)

    @staticmethod
    def gram_schmidt(polynomials):  # gram_schmidt
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
        sm = PODGP(print_global=False)

        sm.POD(self.database, n_mods=self.nt)
        sm.set_training_values(self.xt)
        sm.train()

        mean_xt = sm.predict_values(self.xt)
        var_xt = sm.predict_variances(self.xt)
        np.testing.assert_allclose(self.database, mean_xt, atol=1e-9)
        np.testing.assert_allclose(var_xt, np.zeros(var_xt.shape), atol=1e-9)

        mean_xn = sm.predict_variances(self.xn)
        var_xn = sm.predict_variances(self.xn)
        deriv_xn = sm.predict_derivatives(self.xn, 0)

        assert mean_xn.shape == (self.nn, self.ny)
        assert var_xn.shape == (self.nn, self.ny)
        assert deriv_xn.shape == (self.nn, self.ny)

    def test_set_options(self):
        sm = PODGP(print_global=False)
        sm.POD(self.database, n_mods=1)
        options = [
            {
                "poly": "quadratic",
                "corr": "matern32",
                "pow_exp_power": 0.38,
                "theta0": [1e-1],
            }
        ]
        sm.set_GP_options(options)

        gp_list = sm.get_gp_coef()
        for gp in gp_list:
            for key in options[0].keys():
                assert gp.options[key] == options[0][key]

    def test_pod(self):
        sm = PODGP(print_global=False)

        sm.POD(self.database, tol=1)
        assert sm.get_ev_ratio() == 1

        n_mods = sm.get_n_mods()
        assert n_mods <= self.n_mods_test and n_mods > 0

        basis_SVD = sm.get_left_basis()
        assert basis_SVD.shape == (self.ny, self.nt)

        singular_values = sm.get_singular_values()
        assert len(singular_values) == self.nt
        np.testing.assert_allclose(
            singular_values[n_mods:], np.zeros(self.nt - n_mods), atol=1e-9
        )

        norm_proj, residue = Test._check_projection(self.basis_original.T, basis_SVD.T)
        np.testing.assert_allclose(norm_proj[:n_mods], np.ones(n_mods), atol=1e-9)
        np.testing.assert_allclose(residue[:n_mods], np.zeros(n_mods), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
