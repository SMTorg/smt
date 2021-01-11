"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
        
This package is distributed under New BSD license.
"""

import numpy as np


def ensure_2d_array(array, name):
    if not isinstance(array, np.ndarray):
        raise ValueError("{} must be a NumPy array".format(name))

    array = np.atleast_2d(array.T).T

    if len(array.shape) != 2:
        raise ValueError("{} must have a rank of 1 or 2".format(name))

    return array


def check_support(sm, name, fail=False):
    if not sm.supports[name] or fail:
        class_name = sm.__class__.__name__
        raise NotImplementedError("{} does not support {}".format(class_name, name))


def check_nx(nx, x):
    if x.shape[1] != nx:
        if nx == 1:
            raise ValueError("x should have shape [:, 1] or [:]")
        else:
            raise ValueError(
                "x should have shape [:, {}] and not {}".format(nx, x.shape)
            )
