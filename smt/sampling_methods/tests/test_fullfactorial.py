import unittest
import numpy as np

from smt.sampling_methods import FullFactorial


class Test(unittest.TestCase):
    def test_ff_weights(self):
        xlimits = np.array([[0.0, 1.0], [0.0, 1.0]])
        sampling = FullFactorial(xlimits=xlimits, weights=[0.25, 0.75])
        num = 10
        x = sampling(num)
        self.assertEqual((10, 2), x.shape)

    def test_ff_rectify(self):
        xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
        sampling = FullFactorial(xlimits=xlimits, clip=True)

        num = 50
        x = sampling(num)
        self.assertEqual((56, 2), x.shape)


if __name__ == "__main__":
    unittest.main()
