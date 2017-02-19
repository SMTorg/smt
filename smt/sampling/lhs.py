"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

LHS sampling; uses the pyDOE package.
"""
from __future__ import division
import pyDOE


def _lhs(xlimits, n, criterion):
    nx = xlimits.shape[0]

    x = pyDOE.lhs(nx, samples=n, criterion=criterion)
    for kx in range(nx):
        x[:, kx] = xlimits[kx, 0] + x[:, kx] * (xlimits[kx, 1] - xlimits[kx, 0])

    return x


def lhs_center(xlimits, n):
    return _lhs(xlimits, n, 'center')


def lhs_maximin(xlimits, n):
    return _lhs(xlimits, n, 'maximin')


def lhs_centermaximin(xlimits, n):
    return _lhs(xlimits, n, 'centermaximin')


def lhs_correlation(xlimits, n):
    return _lhs(xlimits, n, 'correlation')
