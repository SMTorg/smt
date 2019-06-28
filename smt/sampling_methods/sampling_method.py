"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

Base class for sampling algorithms.
"""
import numpy as np

from smt.utils.options_dictionary import OptionsDictionary


class SamplingMethod(object):
    def __init__(self, **kwargs):
        """
        Constructor where values of options can be passed in.

        For the list of options, see the documentation for the problem being used.

        Parameters
        ----------
        **kwargs : named arguments
            Set of options that can be optionally set; each option must have been declared.

        Examples
        --------
        >>> import numpy as np
        >>> from smt.sampling_methods import Random
        >>> sampling = Random(xlimits=np.arange(2).reshape((1, 2)))
        """
        self.options = OptionsDictionary()
        self.options.declare(
            "xlimits",
            types=np.ndarray,
            desc="The interval of the domain in each dimension with shape nx x 2 (required)",
        )
        self._initialize()
        self.options.update(kwargs)

    def _initialize(self):
        """
        Implemented by sampling methods to declare options (optional).

        Examples
        --------
        self.options.declare('option_name', default_value, types=(bool, int), desc='description')
        """
        pass

    def __call__(self, nt):
        """
        Compute the requested number of sampling points.

        The number of dimensions (nx) is determined based on `xlimits.shape[0]`.

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

        x = self._compute(nt)
        for kx in range(nx):
            x[:, kx] = xlimits[kx, 0] + x[:, kx] * (xlimits[kx, 1] - xlimits[kx, 0])

        return x

    def _compute(self, nt):
        """
        Implemented by sampling methods to compute the requested number of sampling points.

        The number of dimensions (nx) is determined based on `xlimits.shape[0]`.

        Arguments
        ---------
        nt : int
            Number of points requested.

        Returns
        -------
        ndarray[nt, nx]
            The sampling locations in the input space.
        """
        raise Exception("This sampling method has not been implemented correctly")
