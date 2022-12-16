"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest
import numpy as np
from smt.surrogate_models import KRG
from smt.problems import Sphere
from smt.sampling_methods import FullFactorial, LHS


class TestKRG(unittest.TestCase):
    def test_predict_output_shape(self):
        d, n = (3, 10)
        sx = LHS(
            xlimits=np.repeat(np.atleast_2d([0.0, 1.0]), d, axis=0),
            criterion="m",
            random_state=42,
        )
        x = sx(n)
        # 2-dimensional output
        n_s = 2
        sy = LHS(
            xlimits=np.repeat(np.atleast_2d([0.0, 1.0]), n_s, axis=0),
            criterion="m",
            random_state=42,
        )
        y = sy(n)

        kriging = KRG(poly="linear")
        kriging.set_training_values(x, y)
        kriging.train()

        val = kriging.predict_values(x)
        self.assertEqual(y.shape, val.shape)

        var = kriging.predict_variances(x)
        self.assertEqual(y.shape, var.shape)

        for kx in range(d):
            val_deriv = kriging.predict_derivatives(x, kx)
            self.assertEqual(y.shape, val_deriv.shape)
            var_deriv = kriging.predict_variance_derivatives(x, 0)
            self.assertEqual((n, n_s), var_deriv.shape)


if __name__ == "__main__":
    unittest.main()
