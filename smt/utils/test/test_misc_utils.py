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
        """Covers smt/utils/silence.py branches."""
        from smt.utils.silence import Silence, Silence2
        import tempfile
        import os
        import sys

        # Test Silence (basic sys.stdout override)
        with Silence():
            print("Silenced")

        # Test Silence2 (low-level FD)
        with Silence2():
            print("FD Silenced")

        # Redirect to file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            fname = tmp.name
        try:
            with Silence2(stdout=fname, mode="wb"):
                # We must print enough to be sure it's flushed
                sys.stdout.write(b"Fruit")
                sys.stdout.flush()

            if os.path.exists(fname):
                with open(fname, "rb") as f:
                    self.assertEqual(f.read().strip(), b"Fruit")
        finally:
            if os.path.exists(fname):
                os.remove(fname)

        # Combined stdout/stderr
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            cname = tmp.name
        try:
            with Silence2(stdout=cname, stderr=cname, mode="wb"):
                sys.stdout.write(b"Combined")
                sys.stdout.flush()
        finally:
            if os.path.exists(cname):
                os.remove(cname)


if __name__ == "__main__":
    unittest.main()
