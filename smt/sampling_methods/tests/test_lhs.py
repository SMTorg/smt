import unittest
import numpy as np

from smt.sampling_methods import LHS


class Test(unittest.TestCase):
    def test_lhs_ese(self):
        xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
        sampling = LHS(xlimits=xlimits, criterion="ese")
        num = 50
        x = sampling(num)

        self.assertEqual((50, 2), x.shape)

    def test_random_state(self):
        xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
        num = 10
        sampling = LHS(xlimits=xlimits, criterion="ese")
        doe1 = sampling(num)
        sampling = LHS(xlimits=xlimits, criterion="ese")
        doe2 = sampling(num)
        self.assertFalse(np.allclose(doe1, doe2))

        sampling = LHS(xlimits=xlimits, criterion="ese", random_state=42)
        doe1 = sampling(num)
        sampling = LHS(
            xlimits=xlimits, criterion="ese", random_state=np.random.RandomState(42)
        )
        doe2 = sampling(num)
        self.assertTrue(np.allclose(doe1, doe2))


if __name__ == "__main__":
    unittest.main()
