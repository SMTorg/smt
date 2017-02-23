"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

LHS sampling; uses the pyDOE package.
"""
from __future__ import division
import pyDOE


def lhs_center(xlimits, n):
    """
    Latin hypercube sampling from pyDOE.

    'center' criterion: center the points within the sampling intervals.

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
    return _lhs(xlimits, n, 'center')


def lhs_maximin(xlimits, n):
    """
    Latin hypercube sampling from pyDOE.

    'maximin' criterion: maximize the minimum distance between points,
    but place the point in a randomized location within its interval.

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
    return _lhs(xlimits, n, 'maximin')


def lhs_centermaximin(xlimits, n):
    """
    Latin hypercube sampling from pyDOE.

    'centermaximin' criterion: same as 'maximin', but centered within the intervals.

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
    return _lhs(xlimits, n, 'centermaximin')


def lhs_correlation(xlimits, n):
    """
    Latin hypercube sampling from pyDOE.

    'correlation' criterion: minimize the maximum correlation coefficient.

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
    return _lhs(xlimits, n, 'correlation')


def _lhs(xlimits, n, criterion):
    nx = xlimits.shape[0]

    x = pyDOE.lhs(nx, samples=n, criterion=criterion)
    for kx in range(nx):
        x[:, kx] = xlimits[kx, 0] + x[:, kx] * (xlimits[kx, 1] - xlimits[kx, 0])

    return x
