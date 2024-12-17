"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

Random sampling.
"""

import warnings

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
            types=(type(None), int, np.random.RandomState, np.random.Generator),
            desc="Numpy RandomState or Generator object or seed number which controls random draws",
        )

        # Update options values passed by the user here to get 'random_state' option
        self.options.update(kwargs)

        # RandomState and Generator are and have to be initialized once at constructor time,
        # not in _compute to avoid yielding the same dataset again and again
        if numpy_version < (2, 0):  # Version is below 2.0.0
            if isinstance(self.options["random_state"], np.random.RandomState):
                self.random_state = self.options["random_state"]
            elif isinstance(self.options["random_state"], np.random.Generator):
                self.random_state = np.random.RandomState()
                warnings.warn(
                    "numpy.random.Generator initialization of random_state is not implemented for numpy "
                    "versions < 2.0.0. Using the default np.random.RandomState() as random_state. "
                    "Please consider upgrading to numpy version > 2.0.0, or use the legacy numpy.random.RandomState "
                    "class in the future.",
                    FutureWarning,
                )
            elif isinstance(self.options["random_state"], int):
                self.random_state = np.random.RandomState(self.options["random_state"])
            else:
                self.random_state = np.random.RandomState()
        else:
            # Construct a new Generator with the default BitGenerator (PCG64).
            # If passed a Generator, it will be returned unaltered. When passed a legacy
            # RandomState instance it will be coerced to a Generator.
            self.random_state = np.random.default_rng(seed=self.options["random_state"])

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
        if numpy_version < (2, 0):  # Version is below 2.0.0
            return self.random_state.rand(nt, nx)
        else:
            # Create a Generator object with a specified seed (numpy.random_state.rand(nt, nx)
            # is being deprecated)
            return self.random_state.random((nt, nx))
