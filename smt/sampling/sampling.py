"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

Base class for sampling algorithms.
"""
import numpy as np

from smt.utils.options_dictionary import OptionsDictionary


class Sampling(object):

    def __init__(self, **kwargs):
        self.options = OptionsDictionary()
        self.options.declare('xlimits', types=np.ndarray)
        self._declare_options()
        self.options.update(kwargs)

    def _declare_options(self):
        pass

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
        pass
