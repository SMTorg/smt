"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""


import unittest
import numpy as np
from smt.surrogate_models.krg_based import KrgBased


class TestKrgBased(unittest.TestCase):
    def test_theta0_default_init(self):
        krg = KrgBased()
        krg.set_training_values(np.array([[1, 2, 3]]), np.array([[1]]))
        krg._check_param()
        self.assertTrue(np.array_equal(krg.options["theta0"], [1e-2, 1e-2, 1e-2]))

    def test_theta0_one_dim_init(self):
        krg = KrgBased(theta0=[2e-2])
        krg.set_training_values(np.array([[1, 2, 3]]), np.array([[1]]))
        krg._check_param()
        self.assertTrue(np.array_equal(krg.options["theta0"], [2e-2, 2e-2, 2e-2]))

    def test_theta0_erroneous_init(self):
        krg = KrgBased(theta0=[2e-2, 1e-2])
        krg.set_training_values(np.array([[1, 2]]), np.array([[1]]))  # correct
        krg._check_param()
        krg.set_training_values(np.array([[1, 2, 3]]), np.array([[1]]))  # erroneous
        self.assertRaises(ValueError, krg._check_param)


if __name__ == "__main__":
    unittest.main()
