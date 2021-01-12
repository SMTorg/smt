import unittest
import numpy as np

from smt.utils.sm_test_case import SMTestCase
from smt.utils.kriging_utils import standardization
from smt.sampling_methods import LHS


class Test(SMTestCase):
    def test_standardization(self):
        d, n = (10, 100)
        sx = LHS(
            xlimits=np.repeat(np.atleast_2d([0.0, 1.0]), d, axis=0),
            criterion="m",
            random_state=42,
        )
        X = sx(n)
        sy = LHS(
            xlimits=np.repeat(np.atleast_2d([0.0, 1.0]), 1, axis=0),
            criterion="m",
            random_state=42,
        )
        y = sy(n)
        X_norm, _, _, _, _, _ = standardization(X, y, scale_X_to_unit=True)
        interval = (np.min(X_norm), np.max(X_norm))
        self.assertEqual((0, 1), interval)


if __name__ == "__main__":
    unittest.main()
