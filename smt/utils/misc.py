"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

import numpy as np


def compute_rms_error(sm, xe=None, ye=None, kx=None):
    """
    Returns a normalized RMS error of the training points or the given points.

    Arguments
    ---------
    sm : Surrogate
        Surrogate model instance.
    xe : np.ndarray[ne, dim] or None
        Input values. If None, the input values at the training points are used instead.
    ye : np.ndarray[ne, 1] or None
        Output / deriv. values. If None, the training pt. outputs / derivs. are used.
    kx : int or None
        If None, we are checking the output values.
        If int, we are checking the derivs. w.r.t. the kx^{th} input variable (0-based).
    """

    if xe is not None and ye is not None:
        ye = ye.reshape((xe.shape[0], 1))
        if kx == None:
            ye2 = sm.predict_values(xe)
        else:
            ye2 = sm.predict_derivatives(xe, kx)
        return np.linalg.norm(ye2 - ye) / np.linalg.norm(ye)
    elif xe is None and ye is None:
        num = 0.0
        den = 0.0
        if kx is None:
            kx2 = 0
        else:
            kx2 += 1
        if kx2 not in sm.training_points[None]:
            raise ValueError(
                "There is no training point data available for kx %s" % kx2
            )
        xt, yt = sm.training_points[None][kx2]
        if kx == None:
            yt2 = sm.predict_values(xt)
        else:
            yt2 = sm.predict_derivatives(xt, kx)
        num = np.linalg.norm(yt2 - yt)
        den = np.linalg.norm(yt)
        return num / den
