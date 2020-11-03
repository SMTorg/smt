"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest
import numpy as np
from smt.surrogate_models import MGP
from smt.problems import Sphere
from smt.sampling_methods import FullFactorial


class TestMGP(unittest.TestCase):
    def test_predict_output(self):
        x = np.random.random((10, 3))
        y = np.random.random((10))

        kriging = MGP(n_comp=2)
        kriging.set_training_values(x, y)
        kriging.train()

        x_fail_1 = np.asarray([0, 0, 0, 0])
        x_fail_2 = np.asarray([0])

        self.assertRaises(ValueError, lambda: kriging.predict_values(x_fail_1))
        self.assertRaises(ValueError, lambda: kriging.predict_values(x_fail_2))

        self.assertRaises(ValueError, lambda: kriging.predict_variances(x_fail_1))
        self.assertRaises(ValueError, lambda: kriging.predict_variances(x_fail_2))

        x_1 = np.atleast_2d([0, 0, 0])

        var = kriging.predict_variances(x_1)
        var_1 = kriging.predict_variances(x_1, True)
        self.assertEqual(var, var_1[0])


if __name__ == "__main__":
    unittest.main()
