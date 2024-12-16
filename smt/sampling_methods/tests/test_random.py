import unittest
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from smt.sampling_methods import Random


class TestRandomSamplingMethod(unittest.TestCase):
    def setUp(self):
        self.xlimits = np.array([[0.0, 1.0], [0.0, 1.0]])  # 2D unit hypercube

    def test_random_state_initialization_legacy(self):
        # Test random state initialization for numpy < 2.0.0
        with patch("smt.sampling_methods.random.numpy_version", new=(1, 21)):
            sampler = Random(xlimits=self.xlimits, random_state=12)
            self.assertIsInstance(sampler.random_state, np.random.RandomState)

    def test_random_state_initialization_new(self):
        # Test random state initialization for numpy >= 2.0.0
        with patch("smt.sampling_methods.random.numpy_version", new=(2, 0)):
            sampler = Random(xlimits=self.xlimits, random_state=12)
            self.assertIsInstance(sampler.random_state, np.random.Generator)

    def test_random_state_warning_for_generator_legacy(self):
        # Test that a warning is issued when using Generator with numpy < 2.0.0
        with (
            patch("smt.sampling_methods.random.numpy_version", new=(1, 21)),
            self.assertWarns(FutureWarning),
        ):
            sampler = Random(xlimits=self.xlimits, random_state=np.random.default_rng())
            self.assertIsInstance(sampler.random_state, np.random.RandomState)

    def test_compute_legacy(self):
        # Test _compute method for numpy < 2.0.0
        with patch("smt.sampling_methods.random.numpy_version", new=(1, 26)):
            sampler = Random(xlimits=self.xlimits, random_state=12)
            points = sampler(4)
            self.assertEqual(points.shape, (4, 2))
            self.assertTrue(np.all(points >= 0) and np.all(points <= 1))
            # Check almost equality with known seed-generated data (example)
            expected_points = np.array(
                [
                    [0.154163, 0.74005],
                    [0.263315, 0.533739],
                    [0.014575, 0.918747],
                    [0.900715, 0.033421],
                ]
            )
            npt.assert_allclose(points, expected_points, rtol=1e-4)

    def test_compute_new(self):
        # Test _compute method for numpy >= 2.0.0
        with patch("smt.sampling_methods.random.numpy_version", new=(2, 2)):
            sampler = Random(xlimits=self.xlimits, random_state=12)
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
