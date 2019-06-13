import unittest

import matplotlib

matplotlib.use("Agg")


class Test(unittest.TestCase):
    def test_random(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.sampling_methods import Random

        xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
        sampling = Random(xlimits=xlimits)

        num = 50
        x = sampling(num)

        print(x.shape)

        plt.plot(x[:, 0], x[:, 1], "o")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def test_lhs(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.sampling_methods import LHS

        xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
        sampling = LHS(xlimits=xlimits)

        num = 50
        x = sampling(num)

        print(x.shape)

        plt.plot(x[:, 0], x[:, 1], "o")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def test_full_factorial(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.sampling_methods import FullFactorial

        xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
        sampling = FullFactorial(xlimits=xlimits)

        num = 50
        x = sampling(num)

        print(x.shape)

        plt.plot(x[:, 0], x[:, 1], "o")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


if __name__ == "__main__":
    unittest.main()
