"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

Random sampling.
"""

from warnings import warn
import numpy as np

from smt.sampling_methods.sampling_method import ScaledSamplingMethod

# Check NumPy version
numpy_version = tuple(
    map(int, np.__version__.split(".")[:2])
)  # Extract major and minor version


class Random(ScaledSamplingMethod):
    def _initialize(self, **kwargs):
        self.options.declare(
            "random_state",
            types=(type(None), int, np.random.RandomState),
            desc="DEPRECATED: Numpy RandomState object or seed number which controls random draws",
        )
        self.options.declare(
            "seed",
            types=(type(None), int, np.random.Generator),
            desc="Numpy Generator object or seed number which controls random draws",
        )

        # Update options values passed by the user here to get 'random_state' option
        self.options.update(kwargs)

        # Generator are and have to be initialized once at constructor time,
        # not in _compute to avoid yielding the same dataset again and again
        if self.options["random_state"] is None:
            self.random_state = np.random.default_rng()
        elif isinstance(self.options["random_state"], np.random.RandomState):
            raise ValueError(
                "np.random.RandomState object is not handled anymore. Please use seed and np.random.Generator"
            )
        elif isinstance(self.options["random_state"], int):
            warn(
                "Passing a seed or integer to random_state is deprecated "
                "and will raise an error in a future version. Please "
                "use seed parameter",
                DeprecationWarning,
                stacklevel=3,
            )
            self.random_state = np.random.default_rng(self.options["random_state"])

        if self.options["seed"] is not None:
            self.random_state = np.random.default_rng(self.options["seed"])

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
            The sampling locations in the unit hypercube.
        """
        xlimits = self.options["xlimits"]
        nx = xlimits.shape[0]
        # Create a Generator object with a specified seed (numpy.random_state.rand(nt, nx)
        # is being deprecated)
        return self.random_state.random((nt, nx))
