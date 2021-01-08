"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

Base class for sampling algorithms.
"""
from collections.abc import Generator
import numpy as np

from smt.utils.options_dictionary import OptionsDictionary


class SamplingMethod(Generator, object):
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

    def send(self, value=None):
        """
        Sends a value to the sampling method (allowing for coroutines
        but may not be supported by the Sampler).
        See: https://www.python.org/dev/peps/pep-0342/
        See: https://docs.python.org/3/reference/expressions.html#generator-iterator-methods

        By default the value is ignored but subclasses may support this feature.

        Arguments
        ---------
        value : Any
            Number of points requested.

        Returns
        -------
        ndarray[1, nx]
            The next sample or raises StopIteration.
        """
        return self.__call__(1)

    def throw(self, exception_type, value=None, traceback=None):
        """Raises an exception of type type at the point where
        the generator was paused, and returns the next value
        yielded by the generator function. If the generator
        exits without yielding another value, a StopIteration
        exception is raised. If the generator function does not
        catch the passed-in exception, or raises a different exception,
        then that exception propagates to the caller.
        """
        super().throw(exception_type, value, traceback)
