"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

Random sampling.
"""

import numpy as np

from smt.sampling_methods.sampling_method import ScaledSamplingMethod


class Random(ScaledSamplingMethod):
    def _initialize(self, **kwargs):
        self.options.declare(
            "random_state",
            types=(type(None), int, np.random.RandomState),
            desc="Numpy RandomState object or seed number which controls random draws",
        )

        # Update options values passed by the user here to get 'random_state' option
        self.options.update(kwargs)

        # RandomState is and has to be initialized once at constructor time,
        # not in _compute to avoid yielding the same dataset again and again
        if isinstance(self.options["random_state"], np.random.RandomState):
            self.random_state = self.options["random_state"]
        elif isinstance(self.options["random_state"], int):
            self.random_state = np.random.RandomState(self.options["random_state"])
        else:
            self.random_state = np.random.RandomState()

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
        # Create a Generator object with a specified seed (np.random.rand(nt, nx) is being deprecated)
        rng = np.random.default_rng(self.random_state)
        return rng.random((nt, nx))
