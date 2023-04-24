import unittest

import matplotlib
import matplotlib.pyplot

matplotlib.use("Agg")
matplotlib.pyplot.switch_backend("Agg")


class Test(unittest.TestCase):
    def test_cantilever_beam(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.problems import CantileverBeam

        ndim = 3
        problem = CantileverBeam(ndim=ndim)

        num = 100
        x = np.ones((num, ndim))
        x[:, 0] = np.linspace(0.01, 0.05, num)
        x[:, 1] = 0.5
        x[:, 2] = 0.5
        y = problem(x)

        yd = np.empty((num, ndim))
        for i in range(ndim):
            yd[:, i] = problem(x, kx=i).flatten()

        print(y.shape)
        print(yd.shape)

        ds = problem.design_space
        self.assertEqual(len(ds.design_variables), ndim)

        plt.plot(x[:, 0], y[:, 0])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def test_mixed_cantilever_beam(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from smt.problems import MixedCantileverBeam
        from smt.utils.kriging import XSpecs
        from smt.applications.mixed_integer import MixedIntegerSamplingMethod
        from smt.sampling_methods import LHS

        problem = MixedCantileverBeam()
        ds = problem.design_space
        self.assertEqual(len(ds.design_variables), 3)
        self.assertEqual(problem.options['ndim'], 3)

        n_doe = 100
        xtypes = ds.get_x_types()
        xlimits = ds.get_x_limits()
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        sampling = MixedIntegerSamplingMethod(
            LHS,
            xspecs,
            criterion="ese",
        )
        xdoe = sampling(n_doe)
        y = problem(xdoe)

        xdoe2, is_acting2 = ds.sample_valid_x(n_doe)
        y2 = problem(xdoe2)
        self.assertTrue(np.all(is_acting2))

        plt.scatter(xdoe[:, 0], y)
        plt.scatter(xdoe2[:, 0], y2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def test_hier_neural_network(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from smt.problems import HierarchicalNeuralNetwork

        problem = HierarchicalNeuralNetwork()
        ds = problem.design_space
        assert len(ds.design_variables) == 8

        x_corr, is_active = ds.correct_get_acting(np.array([
            [0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 1],
            [2, 0, 0, 0, 0, 1, 1, 1],
        ]))
        self.assertTrue(np.all(x_corr == np.array([
            [0, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 1, 0],
            [2, 0, 0, 0, 0, 1, 1, 1],
        ])))
        self.assertTrue(np.all(is_active == np.array([
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=bool)))

        n_doe = 100
        xdoe, is_active = ds.sample_valid_x(n_doe)
        self.assertFalse(np.all(is_active))
        y = problem(xdoe)

        plt.scatter(xdoe[:, 0], y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def test_robot_arm(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.problems import RobotArm

        ndim = 2
        problem = RobotArm(ndim=ndim)

        num = 100
        x = np.ones((num, ndim))
        x[:, 0] = np.linspace(0.0, 1.0, num)
        x[:, 1] = np.pi
        y = problem(x)

        yd = np.empty((num, ndim))
        for i in range(ndim):
            yd[:, i] = problem(x, kx=i).flatten()

        print(y.shape)
        print(yd.shape)

        plt.plot(x[:, 0], y[:, 0])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def test_rosenbrock(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.problems import Rosenbrock

        ndim = 2
        problem = Rosenbrock(ndim=ndim)

        num = 100
        x = np.ones((num, ndim))
        x[:, 0] = np.linspace(-2, 2.0, num)
        x[:, 1] = 0.0
        y = problem(x)

        yd = np.empty((num, ndim))
        for i in range(ndim):
            yd[:, i] = problem(x, kx=i).flatten()

        print(y.shape)
        print(yd.shape)

        plt.plot(x[:, 0], y[:, 0])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def test_sphere(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.problems import Sphere

        ndim = 2
        problem = Sphere(ndim=ndim)

        num = 100
        x = np.ones((num, ndim))
        x[:, 0] = np.linspace(-10, 10.0, num)
        x[:, 1] = 0.0
        y = problem(x)

        yd = np.empty((num, ndim))
        for i in range(ndim):
            yd[:, i] = problem(x, kx=i).flatten()

        print(y.shape)
        print(yd.shape)

        plt.plot(x[:, 0], y[:, 0])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def test_branin(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.problems import Branin

        ndim = 2
        problem = Branin(ndim=ndim)

        num = 100
        x = np.ones((num, ndim))
        x[:, 0] = np.linspace(-5.0, 10.0, num)
        x[:, 1] = np.linspace(0.0, 15.0, num)
        y = problem(x)

        yd = np.empty((num, ndim))
        for i in range(ndim):
            yd[:, i] = problem(x, kx=i).flatten()

        print(y.shape)
        print(yd.shape)

        plt.plot(x[:, 0], y[:, 0])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def test_lp_norm(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.problems import LpNorm

        ndim = 2
        problem = LpNorm(ndim=ndim, order=2)

        num = 100
        x = np.ones((num, ndim))
        x[:, 0] = np.linspace(-1.0, 1.0, num)
        x[:, 1] = np.linspace(-1.0, 1.0, num)
        y = problem(x)

        yd = np.empty((num, ndim))
        for i in range(ndim):
            yd[:, i] = problem(x, kx=i).flatten()

        print(y.shape)
        print(yd.shape)

        plt.plot(x[:, 0], y[:, 0])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def test_tensor_product(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.problems import TensorProduct

        ndim = 2
        problem = TensorProduct(ndim=ndim, func="cos")

        num = 100
        x = np.ones((num, ndim))
        x[:, 0] = np.linspace(-1, 1.0, num)
        x[:, 1] = 0.0
        y = problem(x)

        yd = np.empty((num, ndim))
        for i in range(ndim):
            yd[:, i] = problem(x, kx=i).flatten()

        print(y.shape)
        print(yd.shape)

        plt.plot(x[:, 0], y[:, 0])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def test_torsion_vibration(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.problems import TorsionVibration

        ndim = 15
        problem = TorsionVibration(ndim=ndim)

        num = 100
        x = np.ones((num, ndim))
        for i in range(ndim):
            x[:, i] = 0.5 * (problem.xlimits[i, 0] + problem.xlimits[i, 1])
        x[:, 0] = np.linspace(1.8, 2.2, num)
        y = problem(x)

        yd = np.empty((num, ndim))
        for i in range(ndim):
            yd[:, i] = problem(x, kx=i).flatten()

        print(y.shape)
        print(yd.shape)

        plt.plot(x[:, 0], y[:, 0])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def test_water_flow(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.problems import WaterFlow

        ndim = 8
        problem = WaterFlow(ndim=ndim)

        num = 100
        x = np.ones((num, ndim))
        for i in range(ndim):
            x[:, i] = 0.5 * (problem.xlimits[i, 0] + problem.xlimits[i, 1])
        x[:, 0] = np.linspace(0.05, 0.15, num)
        y = problem(x)

        yd = np.empty((num, ndim))
        for i in range(ndim):
            yd[:, i] = problem(x, kx=i).flatten()

        print(y.shape)
        print(yd.shape)

        plt.plot(x[:, 0], y[:, 0])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def test_welded_beam(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.problems import WeldedBeam

        ndim = 3
        problem = WeldedBeam(ndim=ndim)

        num = 100
        x = np.ones((num, ndim))
        for i in range(ndim):
            x[:, i] = 0.5 * (problem.xlimits[i, 0] + problem.xlimits[i, 1])
        x[:, 0] = np.linspace(5.0, 10.0, num)
        y = problem(x)

        yd = np.empty((num, ndim))
        for i in range(ndim):
            yd[:, i] = problem(x, kx=i).flatten()

        print(y.shape)
        print(yd.shape)

        plt.plot(x[:, 0], y[:, 0])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def test_wing_weight(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.problems import WingWeight

        ndim = 10
        problem = WingWeight(ndim=ndim)

        num = 100
        x = np.ones((num, ndim))
        for i in range(ndim):
            x[:, i] = 0.5 * (problem.xlimits[i, 0] + problem.xlimits[i, 1])
        x[:, 0] = np.linspace(150.0, 200.0, num)
        y = problem(x)

        yd = np.empty((num, ndim))
        for i in range(ndim):
            yd[:, i] = problem(x, kx=i).flatten()

        print(y.shape)
        print(yd.shape)

        plt.plot(x[:, 0], y[:, 0])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


if __name__ == "__main__":
    unittest.main()
