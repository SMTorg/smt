"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

Tensor-product of cos, exp, or tanh.
"""
from __future__ import division
import numpy as np

from smt.problems.problem import Problem


class TensorProduct(Problem):

    def _declare_options(self):
        self.options.declare('name', 'TP', types=str)
        self.options.declare('func', values=['cos', 'exp', 'tanh'])
        self.options.declare('width', 1., types=(float, int))

    def _initialize(self):
        self.xlimits[:, 0] = -1.
        self.xlimits[:, 1] =  1.

        if self.options['func'] == 'cos':
            a = self.options['width']
            self.func = lambda v: np.cos(a * np.pi * v)
            self.dfunc = lambda v: -a * np.pi * np.sin(a * np.pi * v)
        elif self.options['func'] == 'exp':
            a = self.options['width']
            self.func = lambda v: np.exp(a * v)
            self.dfunc = lambda v: a * np.exp(a * v)
        elif self.options['func'] == 'tanh':
            a = self.options['width']
            self.func = lambda v: np.tanh(a * v)
            self.dfunc = lambda v: a / np.cosh(a * v) ** 2

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

        y = np.ones((ne, 1))
        y[:, 0] = np.prod(self.func(x), 1).T

        if kx is not None:
            y[:, 0] /= self.func(x[:, kx])
            y[:, 0] *= self.dfunc(x[:, kx])

        return y
