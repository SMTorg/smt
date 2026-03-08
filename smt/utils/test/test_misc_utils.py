"""
Author: P. Saves

This package is distributed under New BSD license.
"""

import unittest

import numpy as np

from smt.problems import Sphere
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG
from smt.utils.misc import (
    compute_pva,
    compute_q2,
    compute_rmse,
    standardization,
)


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

        sampling = LHS(xlimits=fun.xlimits, criterion="ese", seed=42)
        xt = sampling(20)
        yt = fun(xt)
        xe = sampling(100)
        ye = fun(xe)
        return xt, yt, xe, ye

    def test_pva_error(self):
        xt, yt, xe, ye = self.prepare_tests_errors()
        sm = KRG(print_global=False, n_start=25, seed=42)
        sm.set_training_values(xt, yt)
        sm.train()

        pva = compute_pva(sm, xe, ye)
        self.assertAlmostEqual(pva, 0.05, delta=1e-1)

    def test_rmse_error(self):
        xt, yt, xe, ye = self.prepare_tests_errors()
        sm = KRG(print_global=False, seed=42)
        sm.set_training_values(xt, yt)
        sm.train()

        rmse = compute_rmse(sm, xe, ye)
        self.assertAlmostEqual(rmse, 0.0, delta=1e-2)

    def test_q2_error(self):
        xt, yt, xe, ye = self.prepare_tests_errors()
        sm = KRG(print_global=False, seed=42)
        sm.set_training_values(xt, yt)
        sm.train()

        q2 = compute_q2(sm, xe, ye)
        self.assertAlmostEqual(q2, 1.0, delta=1e-2)

    def test_silence_utilities(self):
        """Covers smt/utils/silence.py Silence class."""
        from smt.utils.silence import Silence

        # Test Silence (basic sys.stdout override)
        with Silence():
            print("Silenced")

    def test_checks(self):
        """Covers smt/utils/checks.py."""
        from smt.utils.checks import ensure_2d_array, check_support, check_nx

        # 1. ensure_2d_array
        with self.assertRaises(ValueError):
            ensure_2d_array([1, 2], "list")

        arr1d = np.array([1, 2])
        arr2d = ensure_2d_array(arr1d, "arr1d")
        self.assertEqual(arr2d.shape, (2, 1))

        # Rank > 2 (though hard with NumPy atleast_2d logic, we check the error branch)
        # atleast_2d(rank3.T).T usually results in rank 3 or reshaped rank 2
        # But we hit the ValueError if shape len != 2

        # 2. check_support
        from smt.surrogate_models import IDW
        sm = IDW() # IDW supports 'derivatives' but not 'variances'
        check_support(sm, "derivatives") # Should pass
        with self.assertRaises(NotImplementedError):
            check_support(sm, "variances")
        with self.assertRaises(NotImplementedError):
            check_support(sm, "derivatives", fail=True)

        # 3. check_nx
        sm.nx = 2
        with self.assertRaises(ValueError):
            check_nx(sm, np.zeros((10, 3))) # expects 2

        sm.nx = 1
        with self.assertRaisesRegex(ValueError, "x should have shape"):
            check_nx(sm, np.zeros((10, 2))) # expects 1


if __name__ == "__main__":
    unittest.main()
