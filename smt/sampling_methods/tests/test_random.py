import unittest

import numpy as np
import numpy.testing as npt

from smt.sampling_methods import Random


class TestRandomSamplingMethod(unittest.TestCase):
    def setUp(self):
        self.xlimits = np.array([[0.0, 1.0], [0.0, 1.0]])  # 2D unit hypercube

    def test_seed_initialization(self):
        sampler = Random(xlimits=self.xlimits, seed=12)
        self.assertIsInstance(sampler.rng, np.random.Generator)

    def test_compute_new(self):
        sampler = Random(xlimits=self.xlimits, seed=12)
        points = sampler(4)
        self.assertEqual(points.shape, (4, 2))
        self.assertTrue(np.all(points >= 0) and np.all(points <= 1))
        expected_points = np.array(
            [
                [0.250824, 0.946753],
                [0.18932, 0.179291],
                [0.349889, 0.230541],
                [0.670446, 0.115079],
            ]
        )
        npt.assert_allclose(points, expected_points, rtol=1e-4)

    def test_random_generator(self):
        xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
        num = 10
        sampling = Random(xlimits=xlimits, seed=42)
        doe1 = sampling(num)
        sampling = Random(xlimits=xlimits, seed=np.random.default_rng(42))
        doe2 = sampling(num)
        self.assertTrue(np.allclose(doe1, doe2))

    def test_removed_random_state(self):
        xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
        with self.assertRaises(AssertionError):
            Random(xlimits=xlimits, random_state=42)
        with self.assertRaises(AssertionError):
            Random(xlimits=xlimits, random_state=np.random.RandomState(42))


if __name__ == "__main__":
    unittest.main()
