"""
Author: Remi Lafage <remi.lafage@onera.fr> and Nathalie Bartoli
This package is distributed under New BSD license.
"""
import unittest
import numpy as np
from sys import argv

from smt.extensions import EGO
from smt.utils.sm_test_case import SMTestCase
from smt.problems import Branin

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TestEGO(SMTestCase):
    """
    Test class
    """

    plot = None

    @staticmethod
    def function_test_1d(x):
        # function xsinx
        x = np.reshape(x, (-1,))
        y = np.zeros(x.shape)
        y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
        return y.reshape((-1, 1))

    def test_function_test_1d(self):
        ndim = 1
        niter = 15
        xlimits = np.array([[0.0, 25.0]])

        criterion = "EI"

        ego = EGO()

        x_opt, y_opt, _, _, _, _, _ = ego.optimize(
            fun=TestEGO.function_test_1d,
            n_iter=niter,
            criterion=criterion,
            ndim=ndim,
            ndoe=3,
            xlimits=xlimits,
        )

        self.assertAlmostEqual(18.9, float(x_opt), 1)
        self.assertAlmostEqual(-15.1, float(y_opt), 1)

    @staticmethod
    def run_ego_example():
        import numpy as np
        import six
        from smt.extensions import EGO
        from smt.sampling_methods import FullFactorial

        import sklearn
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from mpl_toolkits.mplot3d import Axes3D

        ndim = 1
        niter = 15
        xlimits = np.array([[0.0, 25.0]])

        criterion = "UCB"  #'EI' or 'SBO' or 'UCB'

        ego = EGO()

        x_opt, y_opt, _, _, _, _, _ = ego.optimize(
            fun=TestEGO.function_test_1d,
            n_iter=niter,
            criterion=criterion,
            ndim=ndim,
            ndoe=3,
            xlimits=xlimits,
        )
        print("Xopt", x_opt, y_opt)
        # Check if the optimal point is Xopt= (array([18.9]), array([-15.1]))


if __name__ == "__main__":
    if "--plot" in argv:
        TestEGO.plot = True
        argv.remove("--plot")
    if "--example" in argv:
        TestEGO.run_ego_example()
        exit()
    unittest.main()

