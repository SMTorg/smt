"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""
import sys
import numpy as np
from bisect import bisect_left


def standardization(X, y):
    """

    We substract the mean from each variable. Then, we divide the values of each
    variable by its standard deviation. If scale_X_to_unit, we scale the input
    space X to the unit hypercube [0,1]^dim with dim the input dimension.

    Parameters
    ----------

    X: np.ndarray [n_obs, dim]
            - The input variables.

    y: np.ndarray [n_obs, 1]
            - The output variable.

    Returns
    -------

    X: np.ndarray [n_obs, dim]
          The standardized input matrix.

    y: np.ndarray [n_obs, 1]
          The standardized output vector.

    X_offset: list(dim)
            The mean (or the min if scale_X_to_unit=True) of each input variable.

    y_mean: list(1)
            The mean of the output variable.

    X_scale:  list(dim)
            The standard deviation of each input variable.

    y_std:  list(1)
            The standard deviation of the output variable.

    """

    X_offset = np.mean(X, axis=0)
    X_scale = X.std(axis=0, ddof=1)

    X_scale[np.abs(X_scale) < (100.0 * sys.float_info.epsilon)] = 1.0
    y_mean = np.mean(y, axis=0)
    y_std = y.std(axis=0, ddof=1)
    y_std[y_std == 0.0] = 1.0

    # scale X and y
    X = (X - X_offset) / X_scale
    y = (y - y_mean) / y_std
    return X, y, X_offset, y_mean, X_scale, y_std


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


def take_closest_number(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def take_closest_in_list(myList, x):
    vfunc = np.vectorize(take_closest_number, excluded=["myList"])
    return vfunc(myList=myList, myNumber=x)
