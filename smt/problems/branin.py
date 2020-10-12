"""
Author: Remi Lafage <remi.lafage@onera.fr>

This package is distributed under New BSD license.

Branin function.
"""
import numpy as np

from smt.problems.problem import Problem


class Branin(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[2], types=int)
        self.options.declare("name", "Branin", types=str)

    def _setup(self):
        assert self.options["ndim"] == 2, "ndim must be 2"

        self.xlimits[0, :] = [-5.0, 10]
        self.xlimits[1, :] = [0.0, 15]

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, 2]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape
        assert nx == 2, "x.shape[1] must be 2"

        y = np.zeros((ne, 1), complex)
        b = 5.1 / (4.0 * (np.pi) ** 2)
        c = 5.0 / np.pi
        t = 1.0 / (8.0 * np.pi)
        u = x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - 6
        if kx is None:
            r = 10.0 * (1.0 - t) * np.cos(x[:, 0]) + 10
            y[:, 0] = u ** 2 + r
        else:
            assert kx in [0, 1], "kx must be None, 0 or 1"
            if kx == 0:
                du_dx0 = -2 * b * x[:, 0] + c
                dr_dx0 = -10.0 * (1.0 - t) * np.sin(x[:, 0])
                y[:, 0] = 2 * du_dx0 * u + dr_dx0
            else:  # kx == 1
                y[:, 0] = 2 * u

        return y
