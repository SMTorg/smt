"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

Full-factorial sampling.
"""
from __future__ import division
import numpy as np
from six.moves import range

from smt.sampling.sampling import Sampling


class FullFactorial(Sampling):

    def _initialize(self):
        self.options.declare('weights', values=[None], types=[list, np.ndarray])
        self.options.declare('clip', types=bool)

    def _compute(self, n):
        """
        Compute the requested number of sampling points.

        Arguments
        ---------
        n : int
            Number of points requested.

        Returns
        -------
        ndarray[n, nx]
            The sampling locations in the input space.
        """
        xlimits = self.options['xlimits']
        nx = xlimits.shape[0]

        if self.options['weights'] is None:
            weights = np.ones(nx) / nx
        else:
            weights = np.atleast_1d(self.options['weights'])
            weights /= numpy.sum(weights)

        num_list = np.ones(nx, int)
        while np.prod(num_list) < n:
            ind = np.argmax(weights - num_list / np.sum(num_list))
            num_list[ind] += 1

        lins_list = [np.linspace(0., 1., num_list[kx]) for kx in range(nx)]
        x_list = np.meshgrid(*lins_list, indexing='ij')

        if self.options['clip']:
            n = np.prod(num_list)

        x = np.zeros((n, nx))
        for kx in range(nx):
            x[:, kx] = x_list[kx].reshape(np.prod(num_list))[:n]

        return x
