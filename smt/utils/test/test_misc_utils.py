"""
Author: P. Saves

This package is distributed under New BSD license.
"""
from smt.utils import misc
import unittest
import numpy as np


class TestMisc(unittest.TestCase):
    def test_standardization(self):
        X = np.array([[0], [1], [2]])
        y = np.array([[1], [3], [5]])
        X2, y2, X_offset, y_mean, X_scale, y_std = misc.standardization(
            np.copy(X), np.copy(y)
        )

        self.assertTrue(np.array_equal(X2.T, np.array([[-1, 0, 1]])))
        self.assertTrue(np.array_equal(y2.T, np.array([[-1, 0, 1]])))

        self.assertTrue(np.array_equal(X_offset, np.array([1])))
        self.assertTrue(np.array_equal(y_mean, np.array([3])))

        self.assertTrue(np.array_equal(X_scale, np.array([1])))
        self.assertTrue(np.array_equal(y_std, np.array([2])))


if __name__ == "__main__":
    unittest.main()
