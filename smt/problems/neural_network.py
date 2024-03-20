"""
Author: P.Saves
This package is distributed under New BSD license.

Multi-Layer Perceptron problem from:
C. Audet, E. Hall e-Hannan, and S. Le Digabel. A general mathematical framework for
constrained mixed-variable blackbox optimization problems with meta and categorical variables.
Operations Research Forum,4994:137, 2023.
"""

import numpy as np

from smt.problems.problem import Problem
from smt.utils.design_space import (
    CategoricalVariable,
    DesignSpace,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
)


class HierarchicalNeuralNetwork(Problem):
    def _initialize(self):
        self.options.declare("name", "HierarchicalNeuralNetwork", types=str)

    def _setup(self):
        design_space = DesignSpace(
            [
                OrdinalVariable(values=[1, 2, 3]),  # x0
                FloatVariable(-5, 2),
                FloatVariable(-5, 2),
                OrdinalVariable(values=[8, 16, 32, 64, 128, 256]),  # x3
                CategoricalVariable(values=["ReLU", "SELU", "ISRLU"]),  # x4
                IntegerVariable(0, 5),  # x5
                IntegerVariable(0, 5),  # x6
                IntegerVariable(0, 5),  # x7
            ]
        )

        # x6 is active when x0 >= 2
        design_space.declare_decreed_var(decreed_var=6, meta_var=0, meta_value=[2, 3])
        # x7 is active when x0 >= 3
        design_space.declare_decreed_var(decreed_var=7, meta_var=0, meta_value=3)

        self._set_design_space(design_space)

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
        ds = self.design_space

        def f_neu(x1, x2, x3, x4):
            if x4 == "ReLU":
                return 2 * x1 + x2 - 0.5 * x3
            elif x4 == "SELU":
                return -x1 + 2 * x2 - 0.5 * x3
            elif x4 == "ISRLU":
                return -x1 + x2 + 0.5 * x3
            else:
                raise ValueError(f"Unexpected x4: {x4}")

        def f1(x1, x2, x3, x4, x5):
            return f_neu(x1, x2, x3, x4) + x5**2

        def f2(x1, x2, x3, x4, x5, x6):
            return f_neu(x1, x2, x3, x4) + (x5**2) + 0.3 * x6

        def f3(x1, x2, x3, x4, x5, x6, x7):
            return f_neu(x1, x2, x3, x4) + (x5**2) + 0.3 * x6 - 0.1 * x7**3

        def f(X):
            y = []
            x0_decoded = ds.decode_values(X, i_dv=0)
            x3_decoded = ds.decode_values(X, i_dv=3)
            x4_decoded = ds.decode_values(X, i_dv=4)
            for i, x in enumerate(X):
                x0 = x0_decoded[i]
                x3 = x3_decoded[i]
                x4 = x4_decoded[i]
                if x0 == 1:
                    y.append(f1(x[1], x[2], x3, x4, x[5]))
                elif x0 == 2:
                    y.append(f2(x[1], x[2], x3, x4, x[5], x[6]))
                elif x0 == 3:
                    y.append(f3(x[1], x[2], x3, x4, x[5], x[6], x[7]))
                else:
                    raise ValueError(f"Unexpected x0 value: {x0}")
            return np.array(y)

        return f(x)
