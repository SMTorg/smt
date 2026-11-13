import unittest

import numpy as np

from smt.sampling_methods import Halton, Sobol


class TestQMCSampling(unittest.TestCase):
    def test_xlimits_required(self):
        for cls in (Sobol, Halton):
            with self.assertRaises(ValueError):
                cls()

    def test_shapes_bounds_and_reproducibility(self):
        xlimits = np.array([[-2.0, 5.0], [10.0, 20.0], [0.0, 1.0]])
        nx = xlimits.shape[0]

        for cls in (Sobol, Halton):
            s1 = cls(xlimits=xlimits, seed=123)
            s2 = cls(xlimits=xlimits, seed=123)

            # Default nt = 2 * nx
            x_def = s1()
            self.assertEqual(x_def.shape, (2 * nx, nx))
            np.testing.assert_allclose(x_def, s2())

            # Non-power-of-two sample size (UQ[py]Lab style)
            x50 = cls(xlimits=xlimits, seed=7)(50)
            self.assertEqual(x50.shape, (50, nx))
            self.assertTrue(np.all(x50 >= xlimits[:, 0]))
            self.assertTrue(np.all(x50 <= xlimits[:, 1]))

            # Zero samples
            x0 = cls(xlimits=xlimits, seed=7)(0)
            self.assertEqual(x0.shape, (0, nx))

    def test_unscrambled_deterministic(self):
        xlimits = np.array([[0.0, 1.0], [0.0, 1.0]])
        x_a = Sobol(xlimits=xlimits, scramble=False)(8)
        x_b = Sobol(xlimits=xlimits, scramble=False)(8)
        np.testing.assert_allclose(x_a, x_b)
        x_h_a = Halton(xlimits=xlimits, scramble=False)(10)
        x_h_b = Halton(xlimits=xlimits, scramble=False)(10)
        np.testing.assert_allclose(x_h_a, x_h_b)


if __name__ == "__main__":
    unittest.main()
