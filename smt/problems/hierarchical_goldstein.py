"""
Author: P.Saves and J.H. Bussemaker
This package is distributed under New BSD license.

Cantilever beam problem from:
P. Saves, Y. Diouane, N. Bartoli, T. Lefebvre, and J. Morlier. A mixed-categorical correlation kernel
for gaussian process, 2022
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


class HierarchicalGoldstein(Problem):
    def _setup(self):
        ds = DesignSpace(
            [
                CategoricalVariable(values=[0, 1, 2, 3]),  # meta
                OrdinalVariable(values=[0, 1]),  # x1
                FloatVariable(0, 100),  # x2
                FloatVariable(0, 100),
                FloatVariable(0, 100),
                FloatVariable(0, 100),
                FloatVariable(0, 100),
                IntegerVariable(0, 2),  # x7
                IntegerVariable(0, 2),
                IntegerVariable(0, 2),
                IntegerVariable(0, 2),
            ]
        )

        # x4 is acting if meta == 1, 3
        ds.declare_decreed_var(decreed_var=4, meta_var=0, meta_value=[1, 3])
        # x5 is acting if meta == 2, 3
        ds.declare_decreed_var(decreed_var=5, meta_var=0, meta_value=[2, 3])
        # x7 is acting if meta == 0, 2
        ds.declare_decreed_var(decreed_var=7, meta_var=0, meta_value=[0, 2])
        # x8 is acting if meta == 0, 1
        ds.declare_decreed_var(decreed_var=8, meta_var=0, meta_value=[0, 1])

        self._set_design_space(ds)

    def _evaluate(self, x: np.ndarray, kx=0) -> np.ndarray:
        def H(x1, x2, x3, x4, z3, z4, x5, cos_term):
            h = (
                53.3108
                + 0.184901 * x1
                - 5.02914 * x1**3 * 10 ** (-6)
                + 7.72522 * x1**z3 * 10 ** (-8)
                - 0.0870775 * x2
                - 0.106959 * x3
                + 7.98772 * x3**z4 * 10 ** (-6)
                + 0.00242482 * x4
                + 1.32851 * x4**3 * 10 ** (-6)
                - 0.00146393 * x1 * x2
                - 0.00301588 * x1 * x3
                - 0.00272291 * x1 * x4
                + 0.0017004 * x2 * x3
                + 0.0038428 * x2 * x4
                - 0.000198969 * x3 * x4
                + 1.86025 * x1 * x2 * x3 * 10 ** (-5)
                - 1.88719 * x1 * x2 * x4 * 10 ** (-6)
                + 2.50923 * x1 * x3 * x4 * 10 ** (-5)
                - 5.62199 * x2 * x3 * x4 * 10 ** (-5)
            )
            if cos_term:
                h += 5.0 * np.cos(2.0 * np.pi * (x5 / 100.0)) - 2.0
            return h

        def f1(x1, x2, z1, z2, z3, z4, x5, cos_term):
            c1 = z1 == 0
            c2 = z1 == 1
            c3 = z1 == 2

            c4 = z2 == 0
            c5 = z2 == 1
            c6 = z2 == 2

            y = (
                c4
                * (
                    c1 * H(x1, x2, 20, 20, z3, z4, x5, cos_term)
                    + c2 * H(x1, x2, 50, 20, z3, z4, x5, cos_term)
                    + c3 * H(x1, x2, 80, 20, z3, z4, x5, cos_term)
                )
                + c5
                * (
                    c1 * H(x1, x2, 20, 50, z3, z4, x5, cos_term)
                    + c2 * H(x1, x2, 50, 50, z3, z4, x5, cos_term)
                    + c3 * H(x1, x2, 80, 50, z3, z4, x5, cos_term)
                )
                + c6
                * (
                    c1 * H(x1, x2, 20, 80, z3, z4, x5, cos_term)
                    + c2 * H(x1, x2, 50, 80, z3, z4, x5, cos_term)
                    + c3 * H(x1, x2, 80, 80, z3, z4, x5, cos_term)
                )
            )
            return y

        def f2(x1, x2, x3, z2, z3, z4, x5, cos_term):
            c4 = z2 == 0
            c5 = z2 == 1
            c6 = z2 == 2

            y = (
                c4 * H(x1, x2, x3, 20, z3, z4, x5, cos_term)
                + c5 * H(x1, x2, x3, 50, z3, z4, x5, cos_term)
                + c6 * H(x1, x2, x3, 80, z3, z4, x5, cos_term)
            )
            return y

        def f3(x1, x2, x4, z1, z3, z4, x5, cos_term):
            c1 = z1 == 0
            c2 = z1 == 1
            c3 = z1 == 2

            y = (
                c1 * H(x1, x2, 20, x4, z3, z4, x5, cos_term)
                + c2 * H(x1, x2, 50, x4, z3, z4, x5, cos_term)
                + c3 * H(x1, x2, 80, x4, z3, z4, x5, cos_term)
            )
            return y

        y = []
        for xi in x:
            if xi[0] == 0:
                y.append(
                    f1(xi[2], xi[3], xi[7], xi[8], xi[9], xi[10], xi[6], cos_term=xi[1])
                )
            elif xi[0] == 1:
                y.append(
                    f2(xi[2], xi[3], xi[4], xi[8], xi[9], xi[10], xi[6], cos_term=xi[1])
                )
            elif xi[0] == 2:
                y.append(
                    f3(xi[2], xi[3], xi[5], xi[7], xi[9], xi[10], xi[6], cos_term=xi[1])
                )
            elif xi[0] == 3:
                y.append(
                    H(xi[2], xi[3], xi[4], xi[5], xi[9], xi[10], xi[6], cos_term=xi[1])
                )
            else:
                raise ValueError(f"Unexpected x0: {xi[0]}")
        return np.array(y)
