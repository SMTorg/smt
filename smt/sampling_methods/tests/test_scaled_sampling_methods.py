import unittest
import numpy as np

from smt.sampling_methods import LHS, FullFactorial, Random

SCALED_SAMPLINGS = [LHS, FullFactorial, Random]


class TestScaledSamplingMethods(unittest.TestCase):
    def test_xlimits_missing_error(self):
        for method in SCALED_SAMPLINGS:
            with self.assertRaises(ValueError) as context:
                method()
                self.assertEqual(
                    "xlimits keyword argument is required", str(context.exception)
                )

    def test_default_nt(self):
        for method in SCALED_SAMPLINGS:
            xlimits = np.array([[-5.5, 3.0], [2.0, 3]])
            nx = xlimits.shape[0]
            sampling = method(xlimits=xlimits)
            doe = sampling()
            self.assertEqual(doe.shape, (2 * nx, nx))
