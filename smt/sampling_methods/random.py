"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

Random sampling.
"""

import numpy as np

from smt.sampling_methods.sampling_method import ScaledSamplingMethod

# Check NumPy version
numpy_version = tuple(
    map(int, np.__version__.split(".")[:2])
)  # Extract major and minor version


class Random(ScaledSamplingMethod):
    def _initialize(self, **kwargs):
        self.options.declare(
            "seed",
            types=(type(None), int, np.random.Generator),
            desc="Numpy Generator object or seed number which controls random draws",
        )

        # Update options values passed by the user here to get 'random_state' option
        self.options.update(kwargs)

        # Generator are and have to be initialized once at constructor time,
        # not in _compute to avoid yielding the same dataset again and again
        self.random_state = np.random.default_rng(seed=self.options["seed"])

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
