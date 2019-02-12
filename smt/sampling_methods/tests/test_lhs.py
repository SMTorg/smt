import unittest
import numpy as np

from smt.sampling_methods import LHS

class Test(unittest.TestCase):
    def test_lhs_ese(self):
        xlimits = np.array([
            [0., 4.],
            [0., 3.],
        ])
        sampling = LHS(xlimits=xlimits, criterion='ese')
        num = 50
        x = sampling(num)

        self.assertEqual((50, 2), x.shape)

if __name__=='__main__':
    unittest.main()
