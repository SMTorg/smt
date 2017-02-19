"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

Random sampling.
"""
from __future__ import division
import numpy as np


def random(xlimits, n):
    nx = xlimits.shape[0]

    x = np.random.rand(n, nx)
    for kx in range(nx):
        x[:, kx] = xlimits[kx, 0] + x[:, kx] * (xlimits[kx, 1] - xlimits[kx, 0])

    return x
