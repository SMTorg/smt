"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

Base class for sampling algorithms.
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from smt.utils.options_dictionary import OptionsDictionary


class SamplingMethod(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        """
        Constructor where values of options can be passed in.
        xlimits keyword argument is required.

        For the list of options, see the documentation for the sampling method being used.

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
        self._initialize(**kwargs)
        self.options.update(kwargs)
        if self.options["xlimits"] is None:
            raise ValueError("xlimits keyword argument is required")

    def _initialize(self, **kwargs) -> None:
        """
        Implemented by sampling methods to declare options
        and/or use these optional values for initialization (optional)

        Parameters
        ----------
        **kwargs : named arguments passed by the user
            Set of options that can be optionally set

        Examples
        --------
        self.options.declare('option_name', default_value, types=(bool, int), desc='description')
        """
        pass

    def __call__(self, nt: int = None) -> np.ndarray:
        """
        Compute the samples.
        Depending on the concrete sampling method the requested number of samples nt may not be enforced.

        The number of dimensions (nx) is determined based on `xlimits.shape[0]`.

        Parameters
        ----------
        nt : int
            Number of points hint.

        Returns
        -------
        ndarray[nt, nx]
            The sampling locations in the input space.
        """
        return self._compute(nt)

    @abstractmethod
    def _compute(self, nt: int = None) -> np.ndarray:
        """
        Implemented by sampling methods to compute the samples.
        Depending on the concrete sampling method the requested number of samples nt may not be enforced.

        The number of dimensions (nx) is determined based on `xlimits.shape[0]`.

        Parameters
        ----------
        nt : int
            Number of points requested.
            Depending on the concrete sampling method this requested number of samples may not be enforced.

        Returns
        -------
        ndarray[nt, nx]
            The sampling locations in the input space.
        """
        raise Exception("This sampling method has not been implemented correctly")


class ScaledSamplingMethod(SamplingMethod):
    """This class represents sample methods which generates samples in the unit hypercube [0, 1]^nx.

    The __call__ method does scale the generated samples accordingly to the defined xlimits.

    Implementation notes:

    * When nt is None, it defaults to 2 * nx.
    * xlimits is presence is checked. ValueError is raised if not specified.
    """

    def __call__(self, nt: int = None) -> np.ndarray:
        """
        Compute the samples.
        Depending on the concrete sampling method the requested number of samples nt may not be enforced.
        When nt is None, it defaults to 2 * nx.

        The number of dimensions (nx) is determined based on `xlimits.shape[0]`.

        Parameters
        ----------
        nt : int (optional, default 2*nx)
            Number of points requested.

        Returns
        -------
        ndarray[nt, nx]
            The sampling locations in the input space.
        """
        xlimits = self.options["xlimits"]
        if nt is None:
            nt = 2 * xlimits.shape[0]
        return _scale_to_xlimits(self._compute(nt), xlimits)

    @abstractmethod
    def _compute(self, nt: int = None) -> np.ndarray:
        """
        Implemented by sampling methods to compute the requested number of sampling points.

        The number of dimensions (nx) is determined based on `xlimits.shape[0]`.

        Parameters
        ----------
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
