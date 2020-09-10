"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
       Dr. John T. Hwang         <hwangjt@umich.edu>

This package is distributed under New BSD license.

Welded beam problem from:
Liu, H., Xu, S., & Wang, X. Sampling strategies and metamodeling techniques for engineering design: comparison and application. In ASME Turbo Expo 2016: Turbomachinery Technical Conference and Exposition. American Society of Mechanical Engineers. June, 2016.
Deb, K. An Efficient Constraint Handling Method for Genetic Algorithms. Computer methods in applied mechanics and engineering, 186(2), pp. 311-338. 2000.
"""
import numpy as np
from scipy.misc import derivative
from smt.problems.problem import Problem


class WeldedBeam(Problem):
    def _initialize(self):
        self.options.declare("name", "WeldedBeam", types=str)
        self.options.declare("use_FD", False, types=bool)
        self.options["ndim"] = 3

    def _setup(self):
        assert self.options["ndim"] == 3, "ndim must be 3"  # t, h, l
        self.xlimits[:, 0] = [5, 0.125, 5]
        self.xlimits[:, 1] = [10, 1, 10]

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

        def func(x0, x1, x2):
            tau1 = 6000 / (np.sqrt(2) * x1 * x2)
            tau2 = (
                6000
                * (14 + 0.5 * x2)
                * np.sqrt(0.25 * (x2 ** 2 + (x1 + x0) ** 2))
                / (2 * (0.707 * x1 * x2 * (x2 / 12.0 + 0.25 * (x1 + x0) ** 2)))
            )
            return np.sqrt(
                tau1 ** 2
                + tau2 ** 2
                + x2 * tau1 * tau2 / np.sqrt(0.25 * (x2 ** 2 + (x1 + x0) ** 2))
            )

        for i in range(ne):
            x0 = x[i, 0]
            x1 = x[i, 1]
            x2 = x[i, 2]
            if kx is None:
                y[i, 0] = func(x0, x1, x2)
            else:
                point = [x0, x1, x2]
                if self.options["use_FD"]:
                    point = np.real(np.array(point))
                    y[i, 0] = partial_derivative(func, var=kx, point=point)
                else:
                    ch = 1e-20
                    point[kx] += complex(0, ch)
                    y[i, 0] = np.imag(func(*point)) / ch
                    point[kx] -= complex(0, ch)

        return y
