"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

Tensor-product of cos, exp, or tanh.
"""
import numpy as np

from smt.problems.problem import Problem


class TensorProduct(Problem):
    def _initialize(self):
        self.options.declare("name", "TP", types=str)
        self.options.declare("func", values=["cos", "exp", "tanh", "gaussian"])
        self.options.declare("width", 1.0, types=(float, int))

    def _setup(self):
        self.xlimits[:, 0] = -1.0
        self.xlimits[:, 1] = 1.0

        a = self.options["width"]
        if self.options["func"] == "cos":
            self.func = lambda v: np.cos(a * np.pi * v)
            self.dfunc = lambda v: -a * np.pi * np.sin(a * np.pi * v)
        elif self.options["func"] == "exp":
            self.func = lambda v: np.exp(a * v)
            self.dfunc = lambda v: a * np.exp(a * v)
        elif self.options["func"] == "tanh":
            self.func = lambda v: np.tanh(a * v)
            self.dfunc = lambda v: a / np.cosh(a * v) ** 2
        elif self.options["func"] == "gaussian":
            self.func = lambda v: np.exp(-2.0 * a * v ** 2)
            self.dfunc = lambda v: -4.0 * a * v * np.exp(-2.0 * a * v ** 2)

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

        y = np.ones((ne, 1), complex)
        if kx is None:
            y[:, 0] = np.prod(self.func(x), 1).T
        else:
            for ix in range(nx):
                if kx == ix:
                    y[:, 0] *= self.dfunc(x[:, ix])
                else:
                    y[:, 0] *= self.func(x[:, ix])

        return y
