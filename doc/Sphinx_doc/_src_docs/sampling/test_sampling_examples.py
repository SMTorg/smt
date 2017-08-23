'''
Author: Dr. John T. Hwang <hwangjt@umich.edu>
        Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>
        
This package is distributed under BSD license
'''

import unittest

import matplotlib
matplotlib.use('Agg')


class Test(unittest.TestCase):

    def test_random(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.sampling import Random

        xlimits = np.array([
            [0., 4.],
            [0., 3.],
        ])
        sampling = Random(xlimits=xlimits)

        num = 50
        x = sampling(num)

        print(x.shape)

        plt.plot(x[:, 0], x[:, 1], 'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def test_lhs(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.sampling import LHS

        xlimits = np.array([
            [0., 4.],
            [0., 3.],
        ])
        sampling = LHS(xlimits=xlimits)

        num = 50
        x = sampling(num)

        print(x.shape)

        plt.plot(x[:, 0], x[:, 1], 'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def test_full_factorial(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.sampling import FullFactorial

        xlimits = np.array([
            [0., 4.],
            [0., 3.],
        ])
        sampling = FullFactorial(xlimits=xlimits)

        num = 50
        x = sampling(num)

        print(x.shape)

        plt.plot(x[:, 0], x[:, 1], 'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def test_clustered(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.sampling import Clustered, Random

        xlimits = np.array([
            [0., 4.],
            [0., 3.],
        ])
        sampling = Clustered(kernel=Random(xlimits=xlimits))

        num = 50
        x = sampling(num)

        print(x.shape)

        plt.plot(x[:, 0], x[:, 1], 'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


if __name__ == '__main__':
    unittest.main()
