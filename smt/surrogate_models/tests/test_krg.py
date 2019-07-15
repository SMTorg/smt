"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest
import numpy as np
from smt.surrogate_models import KRG


class TestKRG(unittest.TestCase):
    def test_predict_output_shape(self):
        x = np.random.random((10, 3))
        y = np.random.random((10, 2))

        kriging = KRG()
        kriging.set_training_values(x, y)
        kriging.train()

        val = kriging.predict_values(x)
        self.assertEqual(y.shape, val.shape)

        var = kriging.predict_variances(x)
        self.assertEqual(y.shape, var.shape)


if __name__ == "__main__":
    unittest.main()
