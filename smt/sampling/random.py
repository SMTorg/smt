"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

Random sampling.
"""
from __future__ import division
import numpy as np


def random(xlimits, n):
    """
    Random sampling.

    Arguments
    ---------
    xlimits : ndarray[nx, 2]
        Array of lower and upper bounds for each of the nx dimensions.
    n : int
        Number of points requested.

    Returns
    -------
    ndarray[n, nx]
        The sampling locations in the input space.
    """
    nx = xlimits.shape[0]

    x = np.random.rand(n, nx)
    for kx in range(nx):
        x[:, kx] = xlimits[kx, 0] + x[:, kx] * (xlimits[kx, 1] - xlimits[kx, 0])

    return x
