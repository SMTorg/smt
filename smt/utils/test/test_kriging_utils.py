import unittest
import numpy as np

from smt.utils.sm_test_case import SMTestCase
from smt.utils.kriging_utils import standardization


class Test(SMTestCase):
    def test_standardization(self):
        d, n = (10, 100)
        X = np.random.normal(size=(n, d))
        y = np.random.normal(size=(n, 1))
        X_norm, _, _, _, _, _ = standardization(X, y, scale_X_to_unit=True)

        interval = (np.min(X_norm), np.max(X_norm))
        self.assertEqual((0, 1), interval)


if __name__ == "__main__":
    unittest.main()
