"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

LHS sampling; uses the pyDOE package.
"""
from __future__ import division
import pyDOE
from six.moves import range

from smt.sampling.sampling import Sampling


class LHS(Sampling):

    def _declare_options(self):
        self.options.declare('criterion', 'c', values=['center', 'maximin', 'centermaximin',
                                                       'correlation', 'c', 'm', 'cm', 'corr'])

    def __call__(self, n):
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

        x = pyDOE.lhs(nx, samples=n, criterion=self.options['criterion'])
        for kx in range(nx):
            x[:, kx] = xlimits[kx, 0] + x[:, kx] * (xlimits[kx, 1] - xlimits[kx, 0])

        return x
