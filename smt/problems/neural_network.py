"""
Author: P.Saves 
This package is distributed under New BSD license.

Multi-Layer Perceptron problem from:
C. Audet, E. Hall e-Hannan, and S. Le Digabel. A general mathematical framework for constrained mixed-variable blackbox optimization problems with meta and categorical variables. Operations Research Forum,499
4:137, 2023.
    """
import numpy as np

from smt.problems.problem import Problem


class HierarchicalNeuralNetwork(Problem):
    def _initialize(self):
        self.options.declare("name", "HierarchicalNeuralNetwork", types=str)

    def _setup(self):
        self.options["ndim"] = 8

    def _evaluate(self, x, kx=0):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.

        Returns
        -------
        ndarray[ne, 1]
            Functions values.
        """

        def f_neu(x1, x2, x3, x4):
            if x4 == 0:
                return 2 * x1 + x2 - 0.5 * x3
            if x4 == 1:
                return -x1 + 2 * x2 - 0.5 * x3
            if x4 == 2:
                return -x1 + x2 + 0.5 * x3

        def f1(x1, x2, x3, x4, x5):
            return f_neu(x1, x2, x3, x4) + x5**2

        def f2(x1, x2, x3, x4, x5, x6):
            return f_neu(x1, x2, x3, x4) + (x5**2) + 0.3 * x6

        def f3(x1, x2, x3, x4, x5, x6, x7):
            return f_neu(x1, x2, x3, x4) + (x5**2) + 0.3 * x6 - 0.1 * x7**3

        def f(X):
            y = []
            for x in X:
                if x[0] == 1:
                    y.append(f1(x[1], x[2], x[3], x[4], x[5]))
                elif x[0] == 2:
                    y.append(f2(x[1], x[2], x[3], x[4], x[5], x[6]))
                elif x[0] == 3:
                    y.append(f3(x[1], x[2], x[3], x[4], x[5], x[6], x[7]))
            return np.array(y)

        return f(x)
