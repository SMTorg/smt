"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang         <hwangjt@umich.edu>

Aircraft wing weight problem from:
Liu, H., Xu, S., & Wang, X. Sampling strategies and metamodeling techniques for engineering design: comparison and application. In ASME Turbo Expo 2016: Turbomachinery Technical Conference and Exposition. American Society of Mechanical Engineers. June, 2016.
Forrester, A., Sobester, A., and Keane, A., 2008,
Engineering Design Via Surrogate Modelling: A Practical Guide, John Wiley & Sons, United Kingdom.
"""
from __future__ import division
import numpy as np
from scipy.misc import derivative

from smt.problems.problem import Problem


class WingWeight(Problem):

    def _declare_options(self):
        self.options.declare('name', 'WingWeight', types=str)
        self.options['ndim'] = 10

    def _initialize(self):
        assert self.options['ndim'] == 10, 'ndim must be 10'

        self.xlimits[:, 0] = [150,220,6,-10,16,0.5,0.08,2.5,1700,0.025]
        self.xlimits[:, 1] = [200,300,10,10,45,1,0.18,6,2500,0.08]

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

        def partial_derivative(function, var=0, point=[]):
            args = point[:]
            def wraps(x):
                args[var] = x
                return func(*args)
            return derivative(wraps, point[var], dx = 1e-6)

        def func(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9):
            return 0.036*x0**0.758*x1**0.0035*(x2/np.cos(np.deg2rad(x3))**2) \
                *x4**0.006*x5**0.04*(100*x6/np.cos(np.deg2rad(x3)))**(-0.3)*(x7*x8)**0.49+x0*x9

        for i in range(ne):
            x0 = x[i,0]
            x1 = x[i,1]
            x2 = x[i,2]
            x3 = x[i,3]
            x4 = x[i,4]
            x5 = x[i,5]
            x6 = x[i,6]
            x7 = x[i,7]
            x8 = x[i,8]
            x9 = x[i,9]
            if kx is None:
                y[i,0] = func(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9)
            else:
                point = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9]
                y[i, 0] = partial_derivative(func, var=kx, point=point)

        return y
