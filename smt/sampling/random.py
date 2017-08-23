"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under BSD license

Random sampling.
"""
from __future__ import division
import numpy as np
from six.moves import range

from smt.sampling.sampling import Sampling

class Random(Sampling):

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
        return np.random.rand(n, nx)
