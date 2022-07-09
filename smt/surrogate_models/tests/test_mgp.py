"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest
import numpy as np
from smt.surrogate_models import MGP
from smt.problems import Sphere
from smt.sampling_methods import FullFactorial, LHS


class TestMGP(unittest.TestCase):
    def test_predict_output_shapes(self):
        d, n = (3, 10)
        sx = LHS(
            xlimits=np.repeat(np.atleast_2d([0.0, 1.0]), d, axis=0),
            criterion="m",
            random_state=42,
        )
        x = sx(n)
        sy = LHS(
            xlimits=np.repeat(np.atleast_2d([0.0, 1.0]), 1, axis=0),
            criterion="m",
            random_state=42,
        )
        y = sy(n)
        y = y.flatten()

        mgp = MGP(n_comp=2)
        mgp.set_training_values(x, y)
        mgp.train()

        x_fail_1 = np.asarray([0, 0, 0, 0])
        x_fail_2 = np.asarray([0])

        self.assertRaises(ValueError, lambda: mgp.predict_values(x_fail_1))
        self.assertRaises(ValueError, lambda: mgp.predict_values(x_fail_2))

        self.assertRaises(ValueError, lambda: mgp.predict_variances(x_fail_1))
        self.assertRaises(ValueError, lambda: mgp.predict_variances(x_fail_2))

        xtest = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        n_samples = xtest.shape[0]
        ypred = mgp.predict_values(xtest)
        self.assertEqual(ypred.shape, (n_samples, 1))

        var = mgp.predict_variances(xtest)
        self.assertEqual(var.shape, (n_samples, 1))
        var_no_uq = mgp.predict_variances_no_uq(xtest)
        self.assertEqual(var_no_uq.shape, (n_samples, 1))


if __name__ == "__main__":
    unittest.main()
