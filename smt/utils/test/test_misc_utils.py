"""
Author: P. Saves

This package is distributed under New BSD license.
"""

import unittest

import numpy as np

from smt.utils.misc import (
    compute_q2_error,
    compute_pva_error,
    compute_rmse_error,
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

        sampling = LHS(xlimits=fun.xlimits, criterion="m")
        xe = sampling(20)
        ye = fun(xe)
        return xe, ye

    def test_pva_error(self):
        xe, ye = self.prepare_tests_errors()
        sm = KRG(print_global=False)
        sm.set_training_values(xe, ye)
        sm.train()

        pva = compute_pva_error(sm, xe, ye)
        self.assertLess(pva, 0.7)

    def test_rmse_error(self):
        xe, ye = self.prepare_tests_errors()
        sm = KRG(print_global=False)
        sm.set_training_values(xe, ye)
        sm.train()

        rmse = compute_rmse_error(sm, xe, ye)
        self.assertLess(rmse, 0.1)

    def test_q2_error(self):
        xe, ye = self.prepare_tests_errors()
        sm = KRG(print_global=False)
        sm.set_training_values(xe, ye)
        sm.train()

        q2 = compute_q2_error(sm, xe, ye)
        self.assertAlmostEqual(q2, 1.0, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
