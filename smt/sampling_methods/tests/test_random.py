import unittest
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from smt.sampling_methods import Random


class TestRandomSamplingMethod(unittest.TestCase):
    def setUp(self):
        self.xlimits = np.array([[0.0, 1.0], [0.0, 1.0]])  # 2D unit hypercube

    def test_random_state_initialization_new(self):
        with patch("smt.sampling_methods.random.numpy_version", new=(2, 0)):
            sampler = Random(xlimits=self.xlimits, seed=12)
            self.assertIsInstance(sampler.random_state, np.random.Generator)

    def test_compute_new(self):
        with patch("smt.sampling_methods.random.numpy_version", new=(2, 2)):
            sampler = Random(xlimits=self.xlimits, seed=12)
            points = sampler(4)
            self.assertEqual(points.shape, (4, 2))
            self.assertTrue(np.all(points >= 0) and np.all(points <= 1))
            # Check almost equality with known seed-generated data (example)
            expected_points = np.array(
                [
                    [0.250824, 0.946753],
                    [0.18932, 0.179291],
                    [0.349889, 0.230541],
                    [0.670446, 0.115079],
                ]
            )
            npt.assert_allclose(points, expected_points, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
