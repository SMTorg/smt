"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

Random sampling.
"""
from smt.utils.misc import scale_to_xlimits
import numpy as np

from smt.sampling_methods.sampling_method import SamplingMethod


class Random(SamplingMethod):
    def _compute(self, nt):
        """
        Compute the requested number of sampling points.

        Arguments
        ---------
        nt : int
            Number of points requested.

        Returns
        -------
        ndarray[nt, nx]
            The sampling locations in the input space.
        """
        xlimits = self.options["xlimits"]
        nx = xlimits.shape[0]
        return scale_to_xlimits(np.random.rand(nt, nx), xlimits)
