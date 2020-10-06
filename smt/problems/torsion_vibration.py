"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang         <hwangjt@umich.edu>

This package is distributed under New BSD license.

Torsion vibration problem from:
Liu, H., Xu, S., & Wang, X. Sampling strategies and metamodeling techniques for engineering design: comparison and application. In ASME Turbo Expo 2016: Turbomachinery Technical Conference and Exposition. American Society of Mechanical Engineers. June, 2016.
Wang, L., Beeson, D., Wiggs, G., and Rayasam, M. A Comparison of Metamodeling Methods Using Practical Industry Requirements. In Proceedings of the 47th AIAA/ASME/ASCE/AHS/ASC structures, structural dynamics, and materials conference, Newport, RI, pp. AIAA 2006-1811.
"""
import numpy as np
from scipy.misc import derivative

from smt.problems.problem import Problem


class TorsionVibration(Problem):
    def _initialize(self):
        self.options.declare("name", "TorsionVibration", types=str)
        self.options.declare("use_FD", False, types=bool)
        self.options["ndim"] = 15

    def _setup(self):
        assert self.options["ndim"] == 15, "ndim must be 15"

        self.xlimits[:, 0] = [
            1.8,
            9,
            10530000,
            7.2,
            3510000,
            10.8,
            1.6425,
            10.8,
            5580000,
            2.025,
            2.7,
            0.252,
            12.6,
            3.6,
            0.09,
        ]
        self.xlimits[:, 1] = [
            2.2,
            11,
            12870000,
            8.8,
            4290000,
            13.2,
            2.0075,
            13.2,
            6820000,
            2.475,
            3.3,
            0.308,
            15.4,
            4.4,
            0.11,
        ]

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

        def func(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14):
            K1 = np.pi * x2 * x0 / (32 * x1)
            K2 = np.pi * x8 * x6 / (32 * x7)
            K3 = np.pi * x4 * x9 / (32 * x3)
            M1 = x11 * np.pi * x10 * x5 / (4 * 9.80665)
            M2 = x14 * np.pi * x13 * x12 / (4 * 9.80665)
            J1 = 0.5 * M1 * (x5 / 2) ** 2
            J2 = 0.5 * M2 * (x12 / 2) ** 2
            a = 1
            b = -((K1 + K2) / J1 + (K2 + K3) / J2)
            c = (K1 * K2 + K2 * K3 + K3 * K1) / (J1 * J2)
            return np.sqrt((-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)) / (2 * np.pi)

        for i in range(ne):
            x0 = x[i, 0]
            x1 = x[i, 1]
            x2 = x[i, 2]
            x3 = x[i, 3]
            x4 = x[i, 4]
            x5 = x[i, 5]
            x6 = x[i, 6]
            x7 = x[i, 7]
            x8 = x[i, 8]
            x9 = x[i, 9]
            x10 = x[i, 10]
            x11 = x[i, 11]
            x12 = x[i, 12]
            x13 = x[i, 13]
            x14 = x[i, 14]
            if kx is None:
                y[i, 0] = func(
                    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14
                )
            else:
                point = [
                    x0,
                    x1,
                    x2,
                    x3,
                    x4,
                    x5,
                    x6,
                    x7,
                    x8,
                    x9,
                    x10,
                    x11,
                    x12,
                    x13,
                    x14,
                ]
                if self.options["use_FD"]:
                    point = np.real(np.array(point))
                    y[i, 0] = partial_derivative(func, var=kx, point=point)
                else:
                    ch = 1e-20
                    point[kx] += complex(0, ch)
                    y[i, 0] = np.imag(func(*point)) / ch
                    point[kx] -= complex(0, ch)

        return y
