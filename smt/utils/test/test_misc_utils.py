"""
Author: P. Saves

This package is distributed under New BSD license.
"""

import unittest

import numpy as np

from smt.utils.misc import (
    compute_q2,
    compute_pva,
    compute_rmse,
    standardization,
)
from smt.problems import Sphere
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG


class TestMisc(unittest.TestCase):
    def test_standardization(self):
        X = np.array([[0], [1], [2]])
        y = np.array([[1], [3], [5]])
        X2, y2, X_offset, y_mean, X_scale, y_std = standardization(
            np.copy(X), np.copy(y)
        )

        self.assertTrue(np.array_equal(X2.T, np.array([[-1, 0, 1]])))
        self.assertTrue(np.array_equal(y2.T, np.array([[-1, 0, 1]])))

        self.assertTrue(np.array_equal(X_offset, np.array([1])))
        self.assertTrue(np.array_equal(y_mean, np.array([3])))

        self.assertTrue(np.array_equal(X_scale, np.array([1])))
        self.assertTrue(np.array_equal(y_std, np.array([2])))

    def prepare_tests_errors(self):
        ndim = 2
        fun = Sphere(ndim=ndim)

        sampling = LHS(xlimits=fun.xlimits, criterion="ese", random_state=42)
        xt = sampling(40)
        yt = fun(xt)
        xe = sampling(120)
        ye = fun(xe)
        return xt, yt, xe, ye

    def test_pva_error(self):
        xt, yt, xe, ye = self.prepare_tests_errors()
        sm = KRG(print_global=False, random_state=42)
        sm.set_training_values(xt, yt)
        sm.train()

        pva = compute_pva(sm, xe, ye)
        self.assertAlmostEqual(pva, 2.314, delta=1e-3)

    def test_rmse_error(self):
        xt, yt, xe, ye = self.prepare_tests_errors()
        sm = KRG(print_global=False, random_state=42)
        sm.set_training_values(xt, yt)
        sm.train()

        rmse = compute_rmse(sm, xe, ye)
        self.assertAlmostEqual(rmse, 0.0, delta=1e-3)

    def test_q2_error(self):
        xt, yt, xe, ye = self.prepare_tests_errors()
        sm = KRG(print_global=False, random_state=42)
        sm.set_training_values(xt, yt)
        sm.train()

        q2 = compute_q2(sm, xe, ye)
        self.assertAlmostEqual(q2, 1.0, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
