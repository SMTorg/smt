"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang         <hwangjt@umich.edu>

This package is distributed under New BSD license.

Water flow problem from:
Liu, H., Xu, S., & Wang, X. Sampling strategies and metamodeling techniques for engineering design: comparison and application. In ASME Turbo Expo 2016: Turbomachinery Technical Conference and Exposition. American Society of Mechanical Engineers. June, 2016.
Morris, M. D., Mitchell, T. J., and Ylvisaker, D. Bayesian Design and Analysis of Computer Experiments: Use of Derivatives in Surface Prediction. Technometrics, 35(3), pp. 243-255. 1993.
"""
import numpy as np
from scipy.misc import derivative

from smt.problems.problem import Problem


class WaterFlow(Problem):
    def _initialize(self):
        self.options.declare("name", "WaterFlow", types=str)
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
                2
                * np.pi
                * x2
                * (x3 - x5)
                / (
                    np.log(x1 / x0)
                    * (1 + 2 * x6 * x2 / (np.log(x1 / x0) * x0 ** 2 * x7) + x2 / x4)
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
