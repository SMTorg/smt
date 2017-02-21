"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

Base class for benchmarking/test problems.
"""
import numpy as np

from smt.utils.options_dictionary import OptionsDictionary


class Problem(object):

    def __init__(self, **kwargs):
        self.options = OptionsDictionary()
        self.options.declare('ndim', 1, types=int)
        self._declare_options()
        self.options.update(kwargs)

        self.xlimits = np.zeros((self.options['ndim'], 2))

        self._initialize()

    def _declare_options(self):
        pass

    def _initialize(self):
        pass

    def __call__(self, x, kx=None):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        if not isinstance(x, np.ndarray) or len(x.shape) != 2:
            raise TypeError('x should be a rank-2 array.')
        elif x.shape[1] != self.options['ndim']:
            raise ValueError('The second dimension of x should be %i' % self.options['ndim'])

        if kx is not None:
            if not isinstance(kx, int) or kx < 0:
                raise TypeError('kx should be None or a non-negative int.')

        return self._evaluate(x, kx)
