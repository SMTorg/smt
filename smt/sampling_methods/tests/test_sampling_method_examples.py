import unittest

try:
    import matplotlib

    matplotlib.use("Agg")
    NO_MATPLOTLIB = False
except ImportError:
    NO_MATPLOTLIB = True


class Test(unittest.TestCase):
    @staticmethod
    def run_random():
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

    @staticmethod
    def run_lhs():
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

    @staticmethod
    def run_full_factorial():
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

    @staticmethod
    def run_box_behnken():
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.sampling_methods import BoxBehnken

        xlimits = np.array([[0.0, 4.0], [0.0, 3.0], [-6.0, 1.0]])
        sampling = BoxBehnken(xlimits=xlimits)

        x = sampling()

        print(x.shape)

        ax = plt.axes(projection="3d")
        ax.plot3D(x[:, 0], x[:, 1], x[:, 2], "o")

        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_zlabel("x2")
        plt.show()

    @staticmethod
    def run_plackett_burman():
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.sampling_methods import PlackettBurman

        xlimits = np.array([[0.0, 4.0], [0.0, 3.0], [-6.0, 1.0]])
        sampling = PlackettBurman(xlimits=xlimits)

        x = sampling()

        print(x.shape)

        ax = plt.axes(projection="3d")
        ax.plot3D(x[:, 0], x[:, 1], x[:, 2], "o")

        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_zlabel("x2")
        plt.show()

    @staticmethod
    def run_factorial():
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.sampling_methods import Factorial

        xlimits = np.array([[0.0, 4.0], [0.0, 3.0], [-6.0, 1.0]])
        sampling = Factorial(xlimits=xlimits, levels=[3, 6, 4])

        x = sampling()

        print(x.shape)

        ax = plt.axes(projection="3d")
        ax.plot3D(x[:, 0], x[:, 1], x[:, 2], "o")

        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_zlabel("x2")
        plt.show()

    @staticmethod
    def run_gsd():
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.sampling_methods import Gsd

        xlimits = np.array([[0.0, 4.0], [0.0, 3.0], [-6.0, 1.0]])
        sampling = Gsd(xlimits=xlimits, levels=[3, 6, 4])

        x = sampling()

        print(x.shape)

        ax = plt.axes(projection="3d")
        ax.plot3D(x[:, 0], x[:, 1], x[:, 2], "o")

        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_zlabel("x2")
        plt.show()

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_sampling_methods_examples(self):
        self.run_lhs()
        self.run_full_factorial()
        self.run_random()
        self.run_plackett_burman
        self.run_box_behnken()
        self.run_factorial()
        self.run_gsd()


if __name__ == "__main__":
    unittest.main()
