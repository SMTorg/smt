"""
Author: P.Saves
This package is distributed under New BSD license.

Cantilever beam problem from:
P. Saves, Y. Diouane, N. Bartoli, T. Lefebvre, and J. Morlier.
A mixed-categorical correlation kernel for gaussian process, 2022
"""

import numpy as np

from smt.problems.problem import Problem
from smt.utils.design_space import CategoricalVariable, DesignSpace, FloatVariable


class MixedCantileverBeam(Problem):
    def _initialize(self):
        self.options.declare("name", "MixedCantileverBeam", types=str)
        self.options.declare("P", 50e3, types=(int, float), desc="Tip load (50 kN)")
        self.options.declare(
            "E", 200e9, types=(int, float), desc="Modulus of elast. (200 GPa)"
        )

    def _setup(self):
        self.options["ndim"] = 3
        self.listI = [
            0.0833,
            0.139,
            0.380,
            0.0796,
            0.133,
            0.363,
            0.0859,
            0.136,
            0.360,
            0.0922,
            0.138,
            0.369,
        ]

        self._set_design_space(
            DesignSpace(
                [
                    CategoricalVariable(values=[str(i + 1) for i in range(12)]),
                    FloatVariable(10.0, 20.0),
                    FloatVariable(1.0, 2.0),
                ]
            )
        )

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
        P = self.options["P"]
        E = self.options["E"]
        I = np.int64(x[:, 0]) - 1  # noqa: E741
        L = x[:, 1]
        S = x[:, 2]
        Ival = np.array([self.listI[i] for i in I])
        y = (P * L**3) / (3 * E * S**2 * Ival)
        return y
