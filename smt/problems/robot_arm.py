"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

Robot arm problem from:
Liu, H., Xu, S., & Wang, X. Sampling strategies and metamodeling techniques for engineering design: comparison and application. In ASME Turbo Expo 2016: Turbomachinery Technical Conference and Exposition. American Society of Mechanical Engineers. June, 2016.
An, J., and Owen, A. Quasi-Regression. Journal of complexity, 17(4), pp. 588-607, 2001.
"""
import numpy as np

from smt.problems.problem import Problem


class RobotArm(Problem):
    def _initialize(self):
        self.options.declare("name", "RobotArm", types=str)
        self.options.declare("ndim", 2, types=int)

    def _setup(self):
        assert self.options["ndim"] % 2 == 0, "ndim must be divisible by 2"

        # Length L
        self.xlimits[0::2, 0] = 0.0
        self.xlimits[0::2, 1] = 1.0

        # Angle theta
        self.xlimits[1::2, 0] = 0.0
        self.xlimits[1::2, 1] = 2 * np.pi

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

        nseg = int(self.options["ndim"] / 2)

        pos_x = np.zeros(ne, complex)
        pos_y = np.zeros(ne, complex)
        for iseg in range(nseg):
            L = x[:, 2 * iseg + 0]
            pos_x += L * np.cos(np.sum(x[:, 1 : 2 * iseg + 2 : 2], axis=1))
            pos_y += L * np.sin(np.sum(x[:, 1 : 2 * iseg + 2 : 2], axis=1))

        y = np.zeros((ne, 1), complex)
        d_pos_x = np.zeros(ne, complex)
        d_pos_y = np.zeros(ne, complex)
        if kx is None:
            y[:, 0] = (pos_x ** 2 + pos_y ** 2) ** 0.5
        else:
            kseg = int(np.floor(kx / 2))
            if kx % 2 == 0:
                d_pos_x[:] += np.cos(np.sum(x[:, 1 : 2 * kseg + 2 : 2], axis=1))
                d_pos_y[:] += np.sin(np.sum(x[:, 1 : 2 * kseg + 2 : 2], axis=1))
                y[:, 0] += pos_x / (pos_x ** 2 + pos_y ** 2) ** 0.5 * d_pos_x
                y[:, 0] += pos_y / (pos_x ** 2 + pos_y ** 2) ** 0.5 * d_pos_y
            elif kx % 2 == 1:
                for iseg in range(nseg):
                    L = x[:, 2 * iseg + 0]
                    if kseg <= iseg:
                        d_pos_x[:] -= L * np.sin(
                            np.sum(x[:, 1 : 2 * iseg + 2 : 2], axis=1)
                        )
                        d_pos_y[:] += L * np.cos(
                            np.sum(x[:, 1 : 2 * iseg + 2 : 2], axis=1)
                        )
                y[:, 0] += pos_x / (pos_x ** 2 + pos_y ** 2) ** 0.5 * d_pos_x
                y[:, 0] += pos_y / (pos_x ** 2 + pos_y ** 2) ** 0.5 * d_pos_y

        return y
