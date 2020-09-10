"""
Author: Remi Lafage <remi.lafage@onera.fr>

This package is distributed under New BSD license.

Norm function.
"""
import numpy as np

from smt.problems.problem import Problem


class LpNorm(Problem):
    def _initialize(self, ndim=1):
        self.options.declare("order", default=2, types=int)
        self.options.declare("name", "LpNorm", types=str)

    def _setup(self):
        self.xlimits[:, 0] = -1.0
        self.xlimits[:, 1] = 1.0

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, ndim]
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
        p = self.options["order"]
        assert p > 0

        y = np.zeros((ne, 1), complex)
        lp_norm = np.sum(np.abs(x) ** p, axis=-1) ** (1.0 / p)
        if kx is None:
            y[:, 0] = lp_norm
        else:
            norm_p = np.linalg.norm(x, ord=p)
            y[:, 0] = np.sign(x[:, kx]) * (np.absolute(x[:, kx]) / lp_norm) ** (p - 1)

        return y
