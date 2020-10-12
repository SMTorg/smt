"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang         <hwangjt@umich.edu>

This package is distributed under New BSD license.

Sphere function.
"""
import numpy as np

from smt.problems.problem import Problem


class Sphere(Problem):
    def _initialize(self):
        self.options.declare("name", "Sphere", types=str)

    def _setup(self):
        self.xlimits[:, 0] = -10.0
        self.xlimits[:, 1] = 10.0

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
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

        y = np.zeros((ne, 1), complex)
        if kx is None:
            y[:, 0] = np.sum(x ** 2, 1).T
        else:
            y[:, 0] = 2 * x[:, kx]

        return y
