"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

Base class for sampling algorithms.
"""
from abc import ABCMeta, abstractmethod
import numpy as np

from smt.utils.options_dictionary import OptionsDictionary


class SamplingMethod(object, metaclass=ABCMeta):
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

    def _initialize(self) -> None:
        """
        Implemented by sampling methods to declare options (optional).

        Examples
        --------
        self.options.declare('option_name', default_value, types=(bool, int), desc='description')
        """
        pass

    def __call__(self, nt: int) -> np.ndarray:
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
        return self._compute(nt)

    @abstractmethod
    def _compute(self, nt: int) -> np.ndarray:
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


class ScaledSamplingMethod(SamplingMethod):
    """This class describes an sample method which generates samples in the unit hypercube.

    The __call__ method does scale the generated samples accordingly to the defined xlimits.
    """

    def __call__(self, nt: int) -> np.ndarray:
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
        return _scale_to_xlimits(self._compute(nt), self.options["xlimits"])

    @abstractmethod
    def _compute(self, nt: int) -> np.ndarray:
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
            The sampling locations in the unit hypercube.
        """
        raise Exception("This sampling method has not been implemented correctly")


def _scale_to_xlimits(samples: np.ndarray, xlimits: np.ndarray) -> np.ndarray:
    """Scales the samples from the unit hypercube to the specified limits.

    Parameters
    ----------
    samples : np.ndarray
        The samples with coordinates in [0,1]
    xlimits : np.ndarray
        The xlimits

    Returns
    -------
    np.ndarray
        The scaled samples.
    """
    nx = xlimits.shape[0]
    for kx in range(nx):
        samples[:, kx] = xlimits[kx, 0] + samples[:, kx] * (
            xlimits[kx, 1] - xlimits[kx, 0]
        )
    return samples
