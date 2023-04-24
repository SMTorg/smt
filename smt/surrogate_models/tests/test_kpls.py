"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest
import numpy as np
from smt.surrogate_models import KPLS
from smt.problems import Sphere
from smt.sampling_methods import FullFactorial, LHS


class TestKPLS(unittest.TestCase):
    def test_predict_output(self):
        for corr_str in [
            "pow_exp",
            "abs_exp",
            "squar_exp",
        ]:
            d, n = (3, 3)
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

            kriging = KPLS(n_comp=2, corr=corr_str)
            kriging.set_training_values(x, y)
            kriging.train()

            x_fail_1 = np.asarray([[0, 0, 0, 0]])
            x_fail_2 = np.asarray([[0]])

            self.assertRaises(ValueError, lambda: kriging.predict_values(x_fail_1))
            self.assertRaises(ValueError, lambda: kriging.predict_values(x_fail_2))
            var = kriging.predict_variances(x)
            self.assertEqual(y.shape[0], var.shape[0])
            kriging = KPLS(n_comp=3)
            kriging.set_training_values(x, y)
            self.assertRaises(ValueError, lambda: kriging.train())

    def test_kpls_training_with_zeroed_outputs(self):
        # Test scikit-learn 0.24 regression cf. https://github.com/SMTorg/smt/issues/274
        for corr_str in [
            "pow_exp",
            "abs_exp",
            "squar_exp",
        ]:
            x = np.random.rand(50, 3)
            y = np.zeros(50)

            kpls = KPLS()
            kpls.options["corr"] = corr_str
            kpls.set_training_values(x, y)
            kpls.train()
            x_test = np.asarray([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
            y_test = kpls.predict_values(x_test)
            # KPLS training fails anyway but not due to PLS exception StopIteration
            self.assertEqual(np.linalg.norm(y_test - np.asarray([[0, 0, 0]])), 0)


if __name__ == "__main__":
    unittest.main()
