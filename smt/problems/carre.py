"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang         <hwangjt@umich.edu>

Carre function.
"""
from __future__ import division
import numpy as np

from smt.problems.problem import Problem


class Carre(Problem):

    def _declare_options(self):
        self.options.declare('name', 'Carre', types=str)

    def _initialize(self):
        self.xlimits[:, 0] = -10.
        self.xlimits[:, 1] =  10.

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

        y = np.zeros((ne, 1))
        if kx is None:
            y[:, 0] = np.sum(x**2, 1).T
        else:
            y[:, 0] = 2 * x[:, kx]

        return y
