"""
Author: Emile Roux
Based on the test_ego.py by Remi Lafage <remi.lafage@onera.fr> and Nathalie Bartoli
This package is distributed under New BSD license.
"""

import warnings

warnings.filterwarnings("ignore")

import unittest
import numpy as np
from sys import argv
import matplotlib

matplotlib.use("Agg")

from smt.applications import EGO_para
from smt.utils.sm_test_case import SMTestCase
from smt.problems import Branin, Rosenbrock
from smt.sampling_methods import FullFactorial


class TestEGOp(SMTestCase):
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
        n_iter = 15
        xlimits = np.array([[0.0, 25.0]])

        criterion = "EI"
        n_par = 2

        ego = EGO_para(n_iter=n_iter, criterion=criterion, n_doe=3, xlimits=xlimits, n_par=n_par)

        x_opt, y_opt, _, _, _, _, _ = ego.optimize(fun=TestEGOp.function_test_1d)

        self.assertAlmostEqual(18.9, float(x_opt), delta=1)
        self.assertAlmostEqual(-15.1, float(y_opt), delta=1)

    def test_rosenbrock_2D(self):
        n_iter = 30
        fun = Rosenbrock(ndim=2)
        xlimits = fun.xlimits
        criterion = "UCB"  #'EI' or 'SBO' or 'UCB'

        xdoe = FullFactorial(xlimits=xlimits)(10)
        ego = EGO_para(xdoe=xdoe, n_iter=n_iter, criterion=criterion, xlimits=xlimits)

        x_opt, y_opt, _, _, _, _, _ = ego.optimize(fun=fun)

        self.assertTrue(np.allclose([[1, 1]], x_opt, rtol=0.5))
        self.assertAlmostEqual(0.0, float(y_opt), delta=1)

    def test_branin_2D(self):
        n_iter = 15
        fun = Branin(ndim=2)
        xlimits = fun.xlimits
        criterion = "UCB"  #'EI' or 'SBO' or 'UCB'

        xdoe = FullFactorial(xlimits=xlimits)(10)
        ego = EGO_para(xdoe=xdoe, n_iter=n_iter, criterion=criterion, xlimits=xlimits)

        x_opt, y_opt, _, _, _, _, _ = ego.optimize(fun=fun)

        # 3 optimal points possible: [-pi, 12.275], [pi, 2.275], [9.42478, 2.475]
        self.assertTrue(
            np.allclose([[-3.14, 12.275]], x_opt, rtol=0.1)
            or np.allclose([[3.14, 2.275]], x_opt, rtol=0.1)
            or np.allclose([[9.42, 2.475]], x_opt, rtol=0.1)
        )
        self.assertAlmostEqual(0.39, float(y_opt), delta=1)

    def test_ydoe_option(self):
        n_iter = 10
        fun = Branin(ndim=2)
        xlimits = fun.xlimits
        criterion = "UCB"  #'EI' or 'SBO' or 'UCB'

        xdoe = FullFactorial(xlimits=xlimits)(10)
        ydoe = fun(xdoe)

        ego = EGO_para(
            xdoe=xdoe, ydoe=ydoe, n_iter=n_iter, criterion=criterion, xlimits=xlimits
        )
        _, y_opt, _, _, _, _, y_doe = ego.optimize(fun=fun)

        self.assertAlmostEqual(0.39, float(y_opt), delta=1)
    
    def test_find_points(self):
        fun = TestEGOp.function_test_1d
        xlimits =  np.array([[0.0, 25.0]])
        xdoe = FullFactorial(xlimits=xlimits)(3)
        ydoe = fun(xdoe)
        ego = EGO_para(
            xdoe=xdoe, ydoe=ydoe, n_iter=1, criterion="UCB", xlimits=xlimits,
            n_start=30,
        )
        _, _, _, _, _, _, _ = ego.optimize(fun=fun)
        x, _ = ego._find_points(xdoe,ydoe)
        print(x)
        self.assertAlmostEqual(6.5, float(x), delta=1)
        
    @staticmethod
    def run_egop_example():
        #TODO
        pass


if __name__ == "__main__":
    if "--plot" in argv:
        TestEGO.plot = True
        argv.remove("--plot")
    if "--example" in argv:
        TestEGO.run_ego_example()
        exit()
    unittest.main()

