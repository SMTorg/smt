"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang         <hwangjt@umich.edu>

Water flow problem from:
Xiong, S., Qian, P. Z., & Wu, C. J. (2013). Sequential design and analysis of high-accuracy and low-accuracy computer codes. Technometrics, 55(1), 37-46.
"""
import numpy as np
from scipy.misc import derivative

from smt.problems.problem import Problem


class WaterFlowLFidelity(Problem):
    def _initialize(self):
        self.options.declare("name", "WaterFlowLFidelity", types=str)
        self.options.declare("use_FD", False, types=bool)
        self.options["ndim"] = 8

    def _setup(self):
        assert self.options["ndim"] == 8, "ndim must be 8"

        self.xlimits[:, 0] = [0.05, 100, 63070, 990, 63.1, 700, 1120, 9855]
        self.xlimits[:, 1] = [0.15, 50000, 115600, 1110, 116, 820, 1680, 12045]

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

        def partial_derivative(function, var=0, point=[]):
            args = point[:]

            def wraps(x):
                args[var] = x
                return func(*args)

            return derivative(wraps, point[var], dx=1e-6)

        def func(x0, x1, x2, x3, x4, x5, x6, x7):
            return (
                5
                * x2
                * (x3 - x5)
                / (
                    np.log(x1 / x0)
                    * (1.5 + 2 * x6 * x2 / (np.log(x1 / x0) * x0 ** 2 * x7) + x2 / x4)
                )
            )

        for i in range(ne):
            x0 = x[i, 0]
            x1 = x[i, 1]
            x2 = x[i, 2]
            x3 = x[i, 3]
            x4 = x[i, 4]
            x5 = x[i, 5]
            x6 = x[i, 6]
            x7 = x[i, 7]
            if kx is None:
                y[i, 0] = func(x0, x1, x2, x3, x4, x5, x6, x7)
            else:
                point = [x0, x1, x2, x3, x4, x5, x6, x7]
                if self.options["use_FD"]:
                    point = np.real(np.array(point))
                    y[i, 0] = partial_derivative(func, var=kx, point=point)
                else:
                    ch = 1e-20
                    point[kx] += complex(0, ch)
                    y[i, 0] = np.imag(func(*point)) / ch
                    point[kx] -= complex(0, ch)

        return y
