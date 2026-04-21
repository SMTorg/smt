"""
Distance computation, PLS projection, and regression basis functions
for Kriging-based surrogate models.

This module groups the pure mathematical helper functions that were
previously in :mod:`smt.utils.kriging`. They have no dependency on the
``krg_based`` subpackage itself, making this a leaf module in the import
graph.

Functions — distances
---------------------
cross_distances
    Pairwise componentwise cross-distances.
differences
    Componentwise difference ``X - Y`` via broadcasting.
componentwise_distance
    Transform raw distances to power-exponential correlation distances.

Functions — PLS
---------------
componentwise_distance_PLS
    PLS-projected componentwise correlation distance.
ge_compute_pls
    Gradient-enhanced PLS coefficients.

Functions — regression basis
----------------------------
constant, linear, quadratic
    Zero / first / second order polynomial basis functions.

Functions — infrastructure
--------------------------
njit_use
    Numba JIT decorator factory (respects ``USE_NUMBA_JIT`` env var).
"""

import os

import numpy as np
from pyDOE3 import bbdesign
from sklearn.cross_decomposition import PLSRegression as pls
from sklearn.metrics.pairwise import check_pairwise_arrays

USE_NUMBA_JIT = int(os.getenv("USE_NUMBA_JIT", 0))
prange = range
if USE_NUMBA_JIT:
    from numba import njit, prange

"""
Quick benchmarking with the mixed-integer hierarchical Goldstein function indicates the following:

| Scenario                 | No numba | Numba   | Numba with caching | Speedup | Overhead |
|--------------------------|----------|---------|--------------------|---------|----------|
| HGoldstein 15 pt DoE     | 1.3 sec  | ~25 sec | 1.1 sec            | 15%     | 24 sec   |
| HGoldstein 150 pt DoE    | 38 sec   | ~29 sec | 7.4 sec            | 80%     | 23 sec   |

Important to note: caching is only needed once after installation of smt, so users will only
experience this overhead ONCE --> the rest of the time they use smt it will be faster than without numba!
"""


# ======================================================================
# Infrastructure
# ======================================================================


def njit_use(parallel=False):
    if USE_NUMBA_JIT:
        # njit:          https://numba.readthedocs.io/en/stable/user/jit.html#nopython
        # cache=True:    https://numba.readthedocs.io/en/stable/user/jit.html#cache
        # parallel=True: https://numba.readthedocs.io/en/stable/user/parallel.html
        return njit(parallel=parallel, cache=True)

    return lambda func: func


# ======================================================================
# Distance functions
# ======================================================================


def cross_distances(X, y=None):
    """
    Computes the nonzero componentwise cross-distances between the vectors
    in X or between the vectors in X and the vectors in y.

    Parameters
    ----------

    X: np.ndarray [n_obs, dim]
            - The input variables.
    y: np.ndarray [n_y, dim]
            - The training data.
    Returns
    -------

    D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The cross-distances between the vectors in X.

    ij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
            - The indices i and j of the vectors in X associated to the cross-
              distances in D.
    """
    n_samples, n_features = X.shape
    if y is None:
        n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
        ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int32)
        D = np.zeros((n_nonzero_cross_dist, n_features))

        _cross_dist_mat(n_samples, ij, X, D)
    else:
        n_y, n_features = y.shape
        X, y = check_pairwise_arrays(X, y)
        n_nonzero_cross_dist = n_samples * n_y
        ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int32)
        D = np.zeros((n_nonzero_cross_dist, n_features))

        _cross_dist_mat_y(n_nonzero_cross_dist, n_y, X, y, D, ij)

    return D, ij.astype(np.int32)


@njit_use()
def _cross_dist_mat(n_samples, ij, X, D):
    ll_1 = 0
    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = X[k] - X[(k + 1) : n_samples]


@njit_use()
def _cross_dist_mat_y(n_nonzero_cross_dist, n_y, X, y, D, ij):
    for k in prange(n_nonzero_cross_dist):
        xk = k // n_y
        yk = k % n_y
        D[k] = X[xk] - y[yk]
        ij[k, 0] = xk
        ij[k, 1] = yk


def differences(X, Y):
    "compute the componentwise difference between X and Y"
    X, Y = check_pairwise_arrays(X, Y)
    D = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    return D.reshape((-1, X.shape[1]))


def componentwise_distance(
    D, corr, dim, power=None, theta=None, return_derivative=False
):
    """
    Computes the nonzero componentwise cross-spatial-correlation-distance
    between the vectors in X.

    Parameters
    ----------

    D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The cross-distances between the vectors in X depending of the correlation function.

    corr: str
            - Name of the correlation function used.
              squar_exp or abs_exp.

    dim: int
            - Number of dimension.

    theta: np.ndarray [n_comp]
            - The theta values associated to the coeff_pls.

    return_derivative: boolean
            - Return d/dx derivative of theta*cross-spatial-correlation-distance

    Returns
    -------

    D_corr: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The componentwise cross-spatial-correlation-distance between the
              vectors in X.

    """
    if power is None and corr != "act_exp":
        raise ValueError(
            "Missing power initialization to compute cross-spatial correlation distance"
        )

    if not return_derivative:
        if corr == "act_exp":
            return _comp_dist_act_exp(D, dim)
        return _comp_dist(D, dim, power)
    else:
        if theta is None:
            raise ValueError(
                "Missing theta to compute spatial derivative of theta cross-spatial correlation distance"
            )
        if corr == "act_exp":
            raise ValueError("this option is not implemented for active learning")
        der = _comp_dist_derivative(D, power)
        D_corr = power * np.einsum("j,ij->ij", theta.T, der)
        return D_corr


@njit_use(parallel=True)
def _comp_dist_act_exp(D, dim):
    D_corr = np.zeros((D.shape[0], dim))
    i, nb_limit = 0, 1000
    for i in prange((D_corr.shape[0] // nb_limit) + 1):
        D_corr[i * nb_limit : (i + 1) * nb_limit, :] = D[
            i * nb_limit : (i + 1) * nb_limit, :
        ]
    return D_corr


@njit_use()
def _comp_dist(D, dim, power):
    D_corr = np.zeros((D.shape[0], dim))
    i, nb_limit = 0, 1000
    for i in range((D_corr.shape[0] // nb_limit) + 1):
        D_corr[i * nb_limit : (i + 1) * nb_limit, :] = (
            np.abs(D[i * nb_limit : (i + 1) * nb_limit, :]) ** power
        )
    return D_corr


@njit_use()
def _comp_dist_derivative(D, power):
    der = np.ones(D.shape)
    for i, j in np.ndindex(D.shape):
        der[i][j] = np.abs(D[i][j]) ** (power - 1)
        if D[i][j] < 0:
            der[i][j] = -der[i][j]
    return der


# ======================================================================
# PLS helpers
# ======================================================================


def componentwise_distance_PLS(
    D, corr, n_comp, coeff_pls, power=2.0, theta=None, return_derivative=False
):
    """
    Computes the nonzero componentwise cross-spatial-correlation-distance
    between the vectors in X.

    Parameters
    ----------

    D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The L1 cross-distances between the vectors in X.

    corr: str
            - Name of the correlation function used.
              squar_exp or abs_exp.

    n_comp: int
            - Number of principal components used.

    coeff_pls: np.ndarray [dim, n_comp]
            - The PLS-coefficients.

    theta: np.ndarray [n_comp]
            - The theta values associated to the coeff_pls.

    return_derivative: boolean
            - Return d/dx derivative of theta*cross-spatial-correlation-distance
    Returns
    -------

    D_corr: np.ndarray [n_obs * (n_obs - 1) / 2, n_comp]
            - The componentwise cross-spatial-correlation-distance between the
              vectors in X.

    """
    # Fit the matrix iteratively: avoid some memory troubles .
    limit = int(1e4)

    if corr == "squar_exp":
        power = 2.0
        # assert power == 2.0, "The power coefficient for the squar exp should be 2.0"
    elif corr in ["abs_exp", "matern32", "matern52"]:
        power = 1.0

    D_corr = np.zeros((D.shape[0], n_comp))
    i, nb_limit = 0, int(limit)
    if not return_derivative:
        while True:
            if i * nb_limit > D_corr.shape[0]:
                return D_corr
            else:
                if corr == "squar_exp":
                    D_corr[i * nb_limit : (i + 1) * nb_limit, :] = np.dot(
                        D[i * nb_limit : (i + 1) * nb_limit, :] ** 2, coeff_pls**2
                    )
                elif corr == "pow_exp":
                    D_corr[i * nb_limit : (i + 1) * nb_limit, :] = np.dot(
                        np.abs(D[i * nb_limit : (i + 1) * nb_limit, :]) ** power,
                        np.abs(coeff_pls) ** power,
                    )
                else:
                    # abs_exp
                    D_corr[i * nb_limit : (i + 1) * nb_limit, :] = np.dot(
                        np.abs(D[i * nb_limit : (i + 1) * nb_limit, :]),
                        np.abs(coeff_pls),
                    )
                i += 1

    else:
        if theta is None:
            raise ValueError(
                "Missing theta to compute spatial derivative of theta cross-spatial correlation distance"
            )

        if corr == "squar_exp":
            D_corr = np.zeros(np.shape(D))
            for i, j in np.ndindex(D.shape):
                coef = 0
                for ll in range(n_comp):
                    coef = coef + theta[ll] * coeff_pls[j][ll] ** 2
                coef = 2 * coef
                D_corr[i][j] = coef * D[i][j]
            return D_corr

        elif corr == "pow_exp":
            D_corr = np.zeros(np.shape(D))
            der = np.ones(np.shape(D))
            for i, j in np.ndindex(D.shape):
                coef = 0
                for ll in range(n_comp):
                    coef = coef + theta[ll] * np.abs(coeff_pls[j][ll]) ** power
                coef = power * coef
                D_corr[i][j] = coef * np.abs(D[i][j]) ** (power - 1) * der[i][j]
            return D_corr

        else:
            # abs_exp
            D_corr = np.zeros(np.shape(D))
            der = np.ones(np.shape(D))
            for i, j in np.ndindex(D.shape):
                if D[i][j] < 0:
                    der[i][j] = -1
                coef = 0
                for ll in range(n_comp):
                    coef = coef + theta[ll] * np.abs(coeff_pls[j][ll])
                D_corr[i][j] = coef * der[i][j]

            return D_corr


def ge_compute_pls(X, y, n_comp, pts, delta_x, xlimits, extra_points):
    """
    Gradient-enhanced PLS-coefficients.

    Parameters
    ----------

    X: np.ndarray [n_obs,dim]
            - - The input variables.

    y: np.ndarray [n_obs,ny]
            - The output variable

    n_comp: int
            - Number of principal components used.

    pts: dict()
            - The gradient values.

    delta_x: real
            - The step used in the FOTA.

    xlimits: np.ndarray[dim, 2]
            - The upper and lower var bounds.

    extra_points: int
            - The number of extra points per each training point.

    Returns
    -------

    Coeff_pls: np.ndarray[dim, n_comp]
            - The PLS-coefficients.

    XX: np.ndarray[extra_points*nt, dim]
            - Extra points added (when extra_points > 0)

    yy: np.ndarray[extra_points*nt, 1]
            - Extra points added (when extra_points > 0)

    """
    nt, dim = X.shape
    XX = np.empty(shape=(0, dim))
    yy = np.empty(shape=(0, y.shape[1]))
    _pls = pls(n_comp)

    coeff_pls = np.zeros((nt, dim, n_comp))
    for i in range(nt):
        if dim >= 3:
            sign = np.roll(bbdesign(int(dim), center=1), 1, axis=0)
            _X = np.zeros((sign.shape[0], dim))
            _y = np.zeros((sign.shape[0], 1))
            sign = sign * delta_x * (xlimits[:, 1] - xlimits[:, 0])
            _X = X[i, :] + sign
            for j in range(1, dim + 1):
                sign[:, j - 1] = sign[:, j - 1] * pts[None][j][1][i, 0]
            _y = y[i, :] + np.sum(sign, axis=1).reshape((sign.shape[0], 1))
        else:
            _X = np.zeros((9, dim))
            _y = np.zeros((9, 1))
            # center
            _X[:, :] = X[i, :].copy()
            _y[0, 0] = y[i, 0].copy()
            # right
            _X[1, 0] += delta_x * (xlimits[0, 1] - xlimits[0, 0])
            _y[1, 0] = _y[0, 0].copy() + pts[None][1][1][i, 0] * delta_x * (
                xlimits[0, 1] - xlimits[0, 0]
            )
            # up
            _X[2, 1] += delta_x * (xlimits[1, 1] - xlimits[1, 0])
            _y[2, 0] = _y[0, 0].copy() + pts[None][2][1][i, 0] * delta_x * (
                xlimits[1, 1] - xlimits[1, 0]
            )
            # left
            _X[3, 0] -= delta_x * (xlimits[0, 1] - xlimits[0, 0])
            _y[3, 0] = _y[0, 0].copy() - pts[None][1][1][i, 0] * delta_x * (
                xlimits[0, 1] - xlimits[0, 0]
            )
            # down
            _X[4, 1] -= delta_x * (xlimits[1, 1] - xlimits[1, 0])
            _y[4, 0] = _y[0, 0].copy() - pts[None][2][1][i, 0] * delta_x * (
                xlimits[1, 1] - xlimits[1, 0]
            )
            # right up
            _X[5, 0] += delta_x * (xlimits[0, 1] - xlimits[0, 0])
            _X[5, 1] += delta_x * (xlimits[1, 1] - xlimits[1, 0])
            _y[5, 0] = (
                _y[0, 0].copy()
                + pts[None][1][1][i, 0] * delta_x * (xlimits[0, 1] - xlimits[0, 0])
                + pts[None][2][1][i, 0] * delta_x * (xlimits[1, 1] - xlimits[1, 0])
            )
            # left up
            _X[6, 0] -= delta_x * (xlimits[0, 1] - xlimits[0, 0])
            _X[6, 1] += delta_x * (xlimits[1, 1] - xlimits[1, 0])
            _y[6, 0] = (
                _y[0, 0].copy()
                - pts[None][1][1][i, 0] * delta_x * (xlimits[0, 1] - xlimits[0, 0])
                + pts[None][2][1][i, 0] * delta_x * (xlimits[1, 1] - xlimits[1, 0])
            )
            # left down
            _X[7, 0] -= delta_x * (xlimits[0, 1] - xlimits[0, 0])
            _X[7, 1] -= delta_x * (xlimits[1, 1] - xlimits[1, 0])
            _y[7, 0] = (
                _y[0, 0].copy()
                - pts[None][1][1][i, 0] * delta_x * (xlimits[0, 1] - xlimits[0, 0])
                - pts[None][2][1][i, 0] * delta_x * (xlimits[1, 1] - xlimits[1, 0])
            )
            # right down
            _X[8, 0] += delta_x * (xlimits[0, 1] - xlimits[0, 0])
            _X[8, 1] -= delta_x * (xlimits[1, 1] - xlimits[1, 0])
            _y[8, 0] = (
                _y[0, 0].copy()
                + pts[None][1][1][i, 0] * delta_x * (xlimits[0, 1] - xlimits[0, 0])
                - pts[None][2][1][i, 0] * delta_x * (xlimits[1, 1] - xlimits[1, 0])
            )

        # As of sklearn 0.24.1 a zeroed _y raises an exception while sklearn 0.23 returns zeroed x_rotations
        # For now the try/except below is a workaround to restore the 0.23 behaviour
        try:
            _pls.fit(_X.copy(), _y.copy())
            coeff_pls[i, :, :] = _pls.x_rotations_
        except StopIteration:
            coeff_pls[i, :, :] = 0

        # Add additional points
        if extra_points != 0:
            max_coeff = np.argsort(np.abs(coeff_pls[i, :, 0]))[-extra_points:]
            for ii in max_coeff:
                XX = np.vstack((XX, X[i, :]))
                XX[-1, ii] += delta_x * (xlimits[ii, 1] - xlimits[ii, 0])
                yy = np.vstack((yy, y[i]))
                yy[-1] += (
                    pts[None][1 + ii][1][i]
                    * delta_x
                    * (xlimits[ii, 1] - xlimits[ii, 0])
                )
    return np.abs(coeff_pls).mean(axis=0), XX, yy


# ======================================================================
# Regression basis functions
# ======================================================================

# sklearn.gaussian_process.regression_models
# Copied from sklearn as it is deprecated since 0.19.1 and will be removed in sklearn 0.22
# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         (mostly translation, see implementation details)
# License: BSD 3 clause

"""
The built-in regression models submodule for the gaussian_process module.
"""


def constant(x):
    """
    Zero order polynomial (constant, p = 1) regression model.

    x --> f(x) = 1

    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    """
    x = np.asarray(x, dtype=np.float64)
    n_eval = x.shape[0]
    f = np.ones([n_eval, 1])
    return f


def linear(x):
    """
    First order polynomial (linear, p = n+1) regression model.

    x --> f(x) = [ 1, x_1, ..., x_n ].T

    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    """
    x = np.asarray(x, dtype=np.float64)
    n_eval = x.shape[0]
    f = np.hstack([np.ones([n_eval, 1]), x])
    return f


def quadratic(x):
    """
    Second order polynomial (quadratic, p = n*(n-1)/2+n+1) regression model.

    x --> f(x) = [ 1, { x_i, i = 1,...,n }, { x_i * x_j,  (i,j) = 1,...,n } ].T
                                                          i > j

    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    """

    x = np.asarray(x, dtype=np.float64)
    n_eval, n_features = x.shape
    f = np.hstack([np.ones([n_eval, 1]), x])
    for k in range(n_features):
        f = np.hstack([f, x[:, k, np.newaxis] * x[:, k:]])

    return f
