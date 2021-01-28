"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

Full-factorial sampling.
"""
import numpy as np

from smt.sampling_methods.sampling_method import ScaledSamplingMethod


class FullFactorial(ScaledSamplingMethod):
    def _initialize(self):
        self.options.declare(
            "weights",
            values=None,
            types=(list, np.ndarray),
            desc="relative sampling weights for each nx dimensions",
        )
        self.options.declare(
            "clip",
            default=False,
            types=bool,
            desc="round number of samples to the sampling number product of each nx dimensions (> asked nt)",
        )

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

        if self.options["weights"] is None:
            weights = np.ones(nx) / nx
        else:
            weights = np.atleast_1d(self.options["weights"])
            weights /= np.sum(weights)

        num_list = np.ones(nx, int)
        while np.prod(num_list) < nt:
            ind = np.argmax(weights - num_list / np.sum(num_list))
            num_list[ind] += 1

        lins_list = [np.linspace(0.0, 1.0, num_list[kx]) for kx in range(nx)]
        x_list = np.meshgrid(*lins_list, indexing="ij")

        if self.options["clip"]:
            nt = np.prod(num_list)

        x = np.zeros((nt, nx))
        for kx in range(nx):
            x[:, kx] = x_list[kx].reshape(np.prod(num_list))[:nt]

        return x
