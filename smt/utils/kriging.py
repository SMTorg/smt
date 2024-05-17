"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

import os
from enum import Enum

import numpy as np
from pyDOE3 import bbdesign
from sklearn.cross_decomposition import PLSRegression as pls
from sklearn.metrics.pairwise import check_pairwise_arrays

from smt.utils.design_space import CategoricalVariable

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


def njit_use(parallel=False):
    if USE_NUMBA_JIT:
        # njit:          https://numba.readthedocs.io/en/stable/user/jit.html#nopython
        # cache=True:    https://numba.readthedocs.io/en/stable/user/jit.html#cache
        # parallel=True: https://numba.readthedocs.io/en/stable/user/parallel.html
        return njit(parallel=parallel, cache=True)

    return lambda func: func


class MixHrcKernelType(Enum):
    ARC_KERNEL = "ARC_KERNEL"
    ALG_KERNEL = "ALG_KERNEL"


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


def cross_levels(X, ij, design_space, y=None):
    """
    Returns the levels corresponding to the indices i and j of the vectors in X and the number of levels.
    Parameters
    ----------

    X: np.ndarray [n_obs, dim]
            - The input variables.
    y: np.ndarray [n_y, dim]
            - The training data.
    ij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
            - The indices i and j of the vectors in X associated to the cross-
              distances in D.
    design_space: BaseDesignSpace
        - The design space definition
    Returns
    -------

     Lij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
            - The levels corresponding to the indices i and j of the vectors in X.
     n_levels: np.ndarray
            - The number of levels for every categorical variable.
    """

    n_levels = []
    for dv in design_space.design_variables:
        if isinstance(dv, CategoricalVariable):
            n_levels.append(dv.n_values)
    n_levels = np.array(n_levels)
    n_var = n_levels.shape[0]
    n, _ = ij.shape
    X_cont, cat_features = compute_X_cont(X, design_space)
    X_cat = X[:, cat_features]

    if y is None:
        Lij = _cross_levels_mat(n_var, n, X_cat, ij)
    else:
        Lij = _cross_levels_mat_y(n_var, n, X_cat, ij, y, cat_features)

    return Lij, n_levels


@njit_use(parallel=True)
def _cross_levels_mat(n_var, n, X_cat, ij):
    Lij = np.zeros((n_var, n, 2))
    for k in prange(n_var):
        for ll in prange(n):
            i, j = ij[ll]
            Lij[k][ll][0] = X_cat[i, k]
            Lij[k][ll][1] = X_cat[j, k]
    return Lij


@njit_use(parallel=True)
def _cross_levels_mat_y(n_var, n, X_cat, ij, y, cat_features):
    Lij = np.zeros((n_var, n, 2))
    y_cat = y[:, cat_features]
    for k in prange(n_var):
        for ll in prange(n):
            i, j = ij[ll]
            Lij[k][ll][0] = X_cat[i, k]
            Lij[k][ll][1] = y_cat[j, k]
    return Lij


def cross_levels_homo_space(X, ij, y=None):
    """
    Computes the nonzero componentwise (or Hadamard) product between the vectors in X
    Parameters
    ----------

    X: np.ndarray [n_obs, dim]
            - The input variables.
    y: np.ndarray [n_y, dim]
            - The training data.
    ij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
            - The indices i and j of the vectors in X associated to the cross-
              distances in D.

    Returns
    -------
     dx: np.ndarray [n_obs * (n_obs - 1) / 2,dim]
            - The Hadamard product between the vectors in X.
    """
    dim = np.shape(X)[1]
    n, _ = ij.shape
    dx = np.zeros((n, dim))
    for ll in range(n):
        i, j = ij[ll]
        if y is None:
            dx[ll] = X[i] * X[j]
        else:
            dx[ll] = X[i] * y[j]

    return dx


def compute_X_cont(x, design_space):
    """
    Gets the X_cont part of a vector x for mixed integer
    Parameters
    ----------
    x: np.ndarray [n_obs, dim]
            - The input variables.
    design_space : BaseDesignSpace
        - The design space definition
    Returns
    -------
    X_cont: np.ndarray [n_obs, dim_cont]
         - The non categorical values of the input variables.
    cat_features: np.ndarray [dim]
        -  Indices of the categorical input dimensions.

    """
    is_cat_mask = design_space.is_cat_mask
    return x[:, ~is_cat_mask], is_cat_mask


def gower_componentwise_distances(
    X, x_is_acting, design_space, hierarchical_kernel, y=None, y_is_acting=None
):
    """
    Computes the nonzero Gower-distances componentwise between the vectors
    in X.
    Parameters
    ----------
    X: np.ndarray [n_obs, dim]
        - The input variables.
    x_is_acting: np.ndarray [n_obs, dim]
        - is_acting matrix for the inputs
    design_space : BaseDesignSpace
        - The design space definition
    y: np.ndarray [n_y, dim]
        - The training data
    y_is_acting: np.ndarray [n_y, dim]
        - is_acting matrix for the training points
    Returns
    -------
    D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The gower distances between the vectors in X.
    ij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
            - The indices i and j of the vectors in X associated to the cross-
              distances in D.
    X_cont: np.ndarray [n_obs, dim_cont]
         - The non categorical values of the input variables.
    """
    X = X.astype(np.float64)
    Xt = X
    X_cont, cat_features = compute_X_cont(Xt, design_space)
    is_decreed = design_space.is_conditionally_acting

    # function checks
    if y is None:
        Y = X
        y_is_acting = x_is_acting
    else:
        Y = y
        if y_is_acting is None:
            raise ValueError("Expected y_is_acting because y is given")

    if not isinstance(X, np.ndarray):
        if not np.array_equal(X.columns, Y.columns):
            raise TypeError("X and Y must have same columns!")
    else:
        if not X.shape[1] == Y.shape[1]:
            raise TypeError("X and Y must have same y-dim!")

    if x_is_acting.shape != X.shape or y_is_acting.shape != Y.shape:
        raise ValueError("is_acting matrices must have same shape as X!")

    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)

    Z = np.concatenate((X, Y))
    z_is_acting = np.concatenate((x_is_acting, y_is_acting))
    Z_cat = Z[:, cat_features]

    x_index = range(0, x_n_rows)
    y_index = range(x_n_rows, x_n_rows + y_n_rows)
    X_cat = Z_cat[x_index,]
    Y_cat = Z_cat[y_index,]

    # This is to normalize the numeric values between 0 and 1.
    Z_num = Z[:, ~cat_features]
    z_num_is_acting = z_is_acting[:, ~cat_features]
    num_is_decreed = is_decreed[~cat_features]

    num_bounds = design_space.get_num_bounds()[~cat_features, :]
    if num_bounds.shape[0] > 0:
        Z_offset = num_bounds[:, 0]
        Z_max = num_bounds[:, 1]
        Z_scale = Z_max - Z_offset
        Z_num = (Z_num - Z_offset) / Z_scale
    X_num = Z_num[x_index,]
    Y_num = Z_num[y_index,]
    x_num_is_acting = z_num_is_acting[x_index,]
    y_num_is_acting = z_num_is_acting[y_index,]

    # x_cat_is_acting : activeness vector delta
    # X_cat( not(x_cat_is_acting)) = 0 ###IMPUTED TO FIRST VALUE IN LIST (index 0)
    D_cat = compute_D_cat(X_cat, Y_cat, y)
    D_num, ij = compute_D_num(
        X_num,
        Y_num,
        x_num_is_acting,
        y_num_is_acting,
        num_is_decreed,
        y,
        hierarchical_kernel,
    )
    D = np.concatenate((D_cat, D_num), axis=1) * 0
    D[:, np.logical_not(cat_features)] = D_num
    D[:, cat_features] = D_cat
    if y is not None:
        return D
    else:
        return D, ij.astype(np.int32), X_cont


@njit_use(parallel=True)
def compute_D_cat(X_cat, Y_cat, y):
    nx_samples, n_features = X_cat.shape
    ny_samples, n_features = Y_cat.shape
    n_nonzero_cross_dist = nx_samples * ny_samples
    if y is None:
        n_nonzero_cross_dist = nx_samples * (nx_samples - 1) // 2
    D_cat = np.zeros((n_nonzero_cross_dist, n_features))
    indD = 0
    k1max = nx_samples
    if y is None:
        k1max = nx_samples - 1
    for k1 in range(k1max):
        k2max = ny_samples
        if y is None:
            k2max = ny_samples - k1 - 1
        for k2 in prange(k2max):
            l2 = k2
            if y is None:
                l2 = k2 + k1 + 1
            D_cat[indD + k2] = X_cat[k1] != Y_cat[l2]
        indD += k2max
    return D_cat


@njit_use()  # setting parallel=True results in a stack overflow
def compute_D_num(
    X_num,
    Y_num,
    x_num_is_acting,
    y_num_is_acting,
    num_is_decreed,
    y,
    hierarchical_kernel,
):
    nx_samples, n_features = X_num.shape
    ny_samples, n_features = Y_num.shape
    n_nonzero_cross_dist = nx_samples * ny_samples
    if y is None:
        n_nonzero_cross_dist = nx_samples * (nx_samples - 1) // 2
    D_num = np.zeros((n_nonzero_cross_dist, n_features))
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int32)
    ll_1 = 0
    indD = 0
    k1max = nx_samples
    if y is None:
        k1max = nx_samples - 1
    for k1 in range(k1max):
        k2max = ny_samples
        if y is None:
            k2max = ny_samples - k1 - 1
            ll_0 = ll_1
            ll_1 = ll_0 + nx_samples - k1 - 1
            ij[ll_0:ll_1, 0] = k1
            ij[ll_0:ll_1, 1] = np.arange(k1 + 1, nx_samples)
        for k2 in range(k2max):
            l2 = k2
            if y is None:
                l2 = k2 + k1 + 1
            D_num[indD] = np.abs(X_num[k1] - Y_num[l2])
            indD += 1

    if np.any(num_is_decreed):
        D_num = apply_the_algebraic_distance_to_the_decreed_variable(
            X_num,
            Y_num,
            x_num_is_acting,
            y_num_is_acting,
            num_is_decreed,
            y,
            D_num,
            hierarchical_kernel,
        )

    return D_num, ij


@njit_use()  # setting parallel=True results in a stack overflow
def apply_the_algebraic_distance_to_the_decreed_variable(
    X_num,
    Y_num,
    x_num_is_acting,
    y_num_is_acting,
    num_is_decreed,
    y,
    D_num,
    hierarchical_kernel,
):
    nx_samples, n_features = X_num.shape
    ny_samples, n_features = Y_num.shape

    indD = 0
    k1max = nx_samples
    if y is None:
        k1max = nx_samples - 1
    for k1 in range(k1max):
        k2max = ny_samples
        if y is None:
            k2max = ny_samples - k1 - 1
        x_k1_acting = x_num_is_acting[k1]
        for k2 in range(k2max):
            l2 = k2
            if y is None:
                l2 = k2 + k1 + 1
            abs_delta = np.abs(X_num[k1] - Y_num[l2])
            y_l2_acting = y_num_is_acting[l2]

            # Calculate the distances between the decreed (aka conditionally acting) variables
            if hierarchical_kernel == MixHrcKernelType.ALG_KERNEL:
                abs_delta[num_is_decreed] = (
                    2
                    * np.abs(X_num[k1][num_is_decreed] - Y_num[l2][num_is_decreed])
                    / (
                        np.sqrt(1 + X_num[k1][num_is_decreed] ** 2)
                        * np.sqrt(1 + Y_num[l2][num_is_decreed] ** 2)
                    )
                )
            elif hierarchical_kernel == MixHrcKernelType.ARC_KERNEL:
                abs_delta[num_is_decreed] = np.sqrt(2) * np.sqrt(
                    1
                    - np.cos(
                        np.pi
                        * np.abs(X_num[k1][num_is_decreed] - Y_num[l2][num_is_decreed])
                    )
                )

            # Set distances for non-acting variables: 0 if both are non-acting, 1 if only one is non-acting
            both_non_acting = num_is_decreed & ~(x_k1_acting | y_l2_acting)
            abs_delta[both_non_acting] = 0.0

            either_acting = num_is_decreed & (x_k1_acting != y_l2_acting)
            abs_delta[either_acting] = 1.0

            D_num[indD] = abs_delta
            indD += 1
    return D_num


def differences(X, Y):
    "compute the componentwise difference between X and Y"
    X, Y = check_pairwise_arrays(X, Y)
    D = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    return D.reshape((-1, X.shape[1]))


def compute_X_cross(X, n_levels):
    """
    Computes the full space cross-relaxation of the input X for
    the homoscedastic hypersphere kernel.
    Parameters
    ----------
    X: np.ndarray [n_obs, 1]
            - The input variables.
    n_levels: np.ndarray
            - The number of levels for the categorical variable.
    Returns
    -------
    Zeta: np.ndarray [n_obs, n_levels * (n_levels - 1) / 2]
         - The non categorical values of the input variables.
    """

    dim = int(n_levels * (n_levels - 1) / 2)
    nt = len(X)
    Zeta = np.zeros((nt, dim))
    k = 0
    for i in range(n_levels):
        for j in range(n_levels):
            if j > i:
                s = 0
                for x in X:
                    if int(x) == i or int(x) == j:
                        Zeta[s, k] = 1
                    s += 1
                k += 1

    return Zeta


def abs_exp(theta, d, grad_ind=None, hess_ind=None, derivative_params=None):
    """
    Absolute exponential autocorrelation model.
    (Ornstein-Uhlenbeck stochastic process)::

    Parameters
    ----------

    theta : list[small_d * n_comp]
        Hyperparameters of the correlation model
    d: np.ndarray[n_obs * (n_obs - 1) / 2, n_comp]
        d_i otherwise
    grad_ind : int, optional
        Indice for which component the gradient dr/dtheta must be computed. The default is None.
    hess_ind : int, optional
        Indice for which component the hessian  d²r/d²(theta) must be computed. The default is None.
    derivative_paramas : dict, optional
        List of arguments mandatory to compute the gradient dr/dx. The default is None.

    Raises
    ------
    Exception
        Assure that theta is of the good length

    Returns
    -------
    r: np.ndarray[n_obs * (n_obs - 1) / 2,1]
         An array containing the values of the autocorrelation model.
    """

    return pow_exp(
        theta,
        d,
        grad_ind=grad_ind,
        hess_ind=hess_ind,
        derivative_params=derivative_params,
    )


def squar_exp(theta, d, grad_ind=None, hess_ind=None, derivative_params=None):
    """
    Squared exponential autocorrelation model.

    Parameters
    ----------

    theta : list[small_d * n_comp]
        Hyperparameters of the correlation model
    d: np.ndarray[n_obs * (n_obs - 1) / 2, n_comp]
        d_i otherwise
    grad_ind : int, optional
        Indice for which component the gradient dr/dtheta must be computed. The default is None.
    hess_ind : int, optional
        Indice for which component the hessian  d²r/d²(theta) must be computed. The default is None.
    derivative_paramas : dict, optional
        List of arguments mandatory to compute the gradient dr/dx. The default is None.

    Raises
    ------
    Exception
        Assure that theta is of the good length

    Returns
    -------
    r: np.ndarray[n_obs * (n_obs - 1) / 2,1]
         An array containing the values of the autocorrelation model.
    """

    return pow_exp(
        theta,
        d,
        grad_ind=grad_ind,
        hess_ind=hess_ind,
        derivative_params=derivative_params,
    )


def pow_exp(theta, d, grad_ind=None, hess_ind=None, derivative_params=None):
    """
    Generative exponential autocorrelation model.

    Parameters
    ----------

    theta : list[small_d * n_comp]
        Hyperparameters of the correlation model
    d: np.ndarray[n_obs * (n_obs - 1) / 2, n_comp]
        d_i otherwise
    grad_ind : int, optional
        Indice for which component the gradient dr/dtheta must be computed. The default is None.
    hess_ind : int, optional
        Indice for which component the hessian  d²r/d²(theta) must be computed. The default is None.
    derivative_paramas : dict, optional
        List of arguments mandatory to compute the gradient dr/dx. The default is None.

    Raises
    ------
    Exception
        Assure that theta is of the good length

    Returns
    -------
    r: np.ndarray[n_obs * (n_obs - 1) / 2,1]
         An array containing the values of the autocorrelation model.
    """

    r = np.zeros((d.shape[0], 1))
    n_components = d.shape[1]

    # Construct/split the correlation matrix
    i, nb_limit = 0, int(1e4)
    while i * nb_limit <= d.shape[0]:
        r[i * nb_limit : (i + 1) * nb_limit, 0] = np.exp(
            -np.sum(
                theta.reshape(1, n_components)
                * d[i * nb_limit : (i + 1) * nb_limit, :],
                axis=1,
            )
        )
        i += 1

    i = 0
    if grad_ind is not None:
        while i * nb_limit <= d.shape[0]:
            r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                -d[i * nb_limit : (i + 1) * nb_limit, grad_ind]
                * r[i * nb_limit : (i + 1) * nb_limit, 0]
            )
            i += 1

    i = 0
    if hess_ind is not None:
        while i * nb_limit <= d.shape[0]:
            r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                -d[i * nb_limit : (i + 1) * nb_limit, hess_ind]
                * r[i * nb_limit : (i + 1) * nb_limit, 0]
            )
            i += 1

    if derivative_params is not None:
        dd = derivative_params["dd"]
        r = r.T
        dr = -np.einsum("i,ij->ij", r[0], dd)
        return r.T, dr

    return r


def squar_sin_exp(theta, d, grad_ind=None, hess_ind=None, derivative_params=None):
    """
    Generative exponential autocorrelation model.

    Parameters
    ----------

    theta : list[small_d * n_comp]
        Hyperparameters of the correlation model
    d: np.ndarray[n_obs * (n_obs - 1) / 2, n_comp]
        d_i otherwise
    grad_ind : int, optional
        Indice for which component the gradient dr/dtheta must be computed. The default is None.
    hess_ind : int, optional
        Indice for which component the hessian  d²r/d²(theta) must be computed. The default is None.
    derivative_paramas : dict, optional
        List of arguments mandatory to compute the gradient dr/dx. The default is None.

    Raises
    ------
    Exception
        Assure that theta is of the good length

    Returns
    -------
    r: np.ndarray[n_obs * (n_obs - 1) / 2,1]
         An array containing the values of the autocorrelation model.
    """

    r = np.zeros((d.shape[0], 1))
    # Construct/split the correlation matrix
    i, nb_limit = 0, int(1e4)
    while i * nb_limit <= d.shape[0]:
        theta_array = theta.reshape(1, len(theta))
        r[i * nb_limit : (i + 1) * nb_limit, 0] = np.exp(
            -np.sum(
                np.atleast_2d(theta_array[0][0 : int(len(theta) / 2)])
                * np.sin(
                    np.atleast_2d(theta_array[0][int(len(theta) / 2) : int(len(theta))])
                    * d[i * nb_limit : (i + 1) * nb_limit, :]
                )
                ** 2,
                axis=1,
            )
        )
        i += 1
    kernel = r.copy()

    i = 0
    if grad_ind is not None:
        cut = int(len(theta) / 2)
        if (
            hess_ind is not None and grad_ind >= cut and hess_ind < cut
        ):  # trick to use the symetry of the hessian when the hessian is asked
            grad_ind, hess_ind = hess_ind, grad_ind

        if grad_ind < cut:
            grad_ind2 = cut + grad_ind
            while i * nb_limit <= d.shape[0]:
                r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                    -(
                        np.sin(
                            theta_array[0][grad_ind2]
                            * d[i * nb_limit : (i + 1) * nb_limit, grad_ind]
                        )
                        ** 2
                    )
                    * r[i * nb_limit : (i + 1) * nb_limit, 0]
                )
                i += 1
        else:
            grad_ind2 = grad_ind - cut
            while i * nb_limit <= d.shape[0]:
                r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                    -theta_array[0][grad_ind2]
                    * d[i * nb_limit : (i + 1) * nb_limit, grad_ind2]
                    * np.sin(
                        2
                        * d[i * nb_limit : (i + 1) * nb_limit, grad_ind2]
                        * theta_array[0][grad_ind]
                    )
                    * r[i * nb_limit : (i + 1) * nb_limit, 0]
                )
                i += 1

    i = 0
    if hess_ind is not None:
        cut = int(len(theta) / 2)
        if grad_ind < cut and hess_ind < cut:
            hess_ind2 = cut + hess_ind
            while i * nb_limit <= d.shape[0]:
                r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                    -(
                        np.sin(
                            theta_array[0][hess_ind2]
                            * d[i * nb_limit : (i + 1) * nb_limit, hess_ind]
                        )
                        ** 2
                    )
                    * r[i * nb_limit : (i + 1) * nb_limit, 0]
                )
                i += 1
        elif grad_ind >= cut and hess_ind >= cut:
            hess_ind2 = hess_ind - cut
            if grad_ind == hess_ind:
                while i * nb_limit <= d.shape[0]:
                    r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                        -2
                        * theta_array[0][hess_ind2]
                        * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2] ** 2
                        * np.cos(
                            2
                            * theta_array[0][grad_ind]
                            * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                        )
                        * kernel[i * nb_limit : (i + 1) * nb_limit, 0]
                        - theta_array[0][hess_ind2]
                        * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                        * np.sin(
                            2
                            * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                            * theta_array[0][hess_ind]
                        )
                        * r[i * nb_limit : (i + 1) * nb_limit, 0]
                    )
                    i += 1
            else:
                while i * nb_limit <= d.shape[0]:
                    r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                        -theta_array[0][hess_ind2]
                        * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                        * np.sin(
                            2
                            * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                            * theta_array[0][hess_ind]
                        )
                        * r[i * nb_limit : (i + 1) * nb_limit, 0]
                    )
                    i += 1
        elif grad_ind < cut and hess_ind >= cut:
            hess_ind2 = hess_ind - cut
            while i * nb_limit <= d.shape[0]:
                r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                    -theta_array[0][hess_ind2]
                    * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                    * np.sin(
                        2
                        * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                        * theta_array[0][hess_ind]
                    )
                    * r[i * nb_limit : (i + 1) * nb_limit, 0]
                )
                if hess_ind2 == grad_ind:
                    r[i * nb_limit : (i + 1) * nb_limit, 0] += (
                        -d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                        * np.sin(
                            2
                            * d[i * nb_limit : (i + 1) * nb_limit, hess_ind2]
                            * theta_array[0][hess_ind]
                        )
                        * kernel[i * nb_limit : (i + 1) * nb_limit, 0]
                    )

                i += 1
        i = 0

    if derivative_params is not None:
        raise ValueError(
            "Spatial derivatives for ExpSinSquared not available yet (to implement)."
        )
    return r


def matern52(theta, d, grad_ind=None, hess_ind=None, derivative_params=None):
    """
    Matern 5/2 correlation model.

     Parameters
    ----------
    theta : list[small_d * n_comp]
        Hyperparameters of the correlation model
    d: np.ndarray[n_obs * (n_obs - 1) / 2, n_comp]
        d_i otherwise
    grad_ind : int, optional
        Indice for which component the gradient dr/dtheta must be computed. The default is None.
    hess_ind : int, optional
        Indice for which component the hessian  d²r/d²(theta) must be computed. The default is None.
    derivative_params : dict, optional
        List of arguments mandatory to compute the gradient dr/dx. The default is None.

    Raises
    ------
    Exception
        Assure that theta is of the good length

    Returns
    -------
    r: np.ndarray[n_obs * (n_obs - 1) / 2,1]
        An array containing the values of the autocorrelation model.
    """

    r = np.zeros((d.shape[0], 1))
    n_components = d.shape[1]

    # Construct/split the correlation matrix
    i, nb_limit = 0, int(1e4)

    while i * nb_limit <= d.shape[0]:
        ll = theta.reshape(1, n_components) * d[i * nb_limit : (i + 1) * nb_limit, :]
        r[i * nb_limit : (i + 1) * nb_limit, 0] = (
            1.0 + np.sqrt(5.0) * ll + 5.0 / 3.0 * ll**2.0
        ).prod(axis=1) * np.exp(-np.sqrt(5.0) * (ll.sum(axis=1)))
        i += 1
    i = 0

    M52 = r.copy()

    if grad_ind is not None:
        theta_r = theta.reshape(1, n_components)
        while i * nb_limit <= d.shape[0]:
            fact_1 = (
                np.sqrt(5) * d[i * nb_limit : (i + 1) * nb_limit, grad_ind]
                + 10.0
                / 3.0
                * theta_r[0, grad_ind]
                * d[i * nb_limit : (i + 1) * nb_limit, grad_ind] ** 2.0
            )
            fact_2 = (
                1.0
                + np.sqrt(5)
                * theta_r[0, grad_ind]
                * d[i * nb_limit : (i + 1) * nb_limit, grad_ind]
                + 5.0
                / 3.0
                * (theta_r[0, grad_ind] ** 2)
                * (d[i * nb_limit : (i + 1) * nb_limit, grad_ind] ** 2)
            )
            fact_3 = np.sqrt(5) * d[i * nb_limit : (i + 1) * nb_limit, grad_ind]

            r[i * nb_limit : (i + 1) * nb_limit, 0] = (fact_1 / fact_2 - fact_3) * r[
                i * nb_limit : (i + 1) * nb_limit, 0
            ]
            i += 1
    i = 0

    if hess_ind is not None:
        while i * nb_limit <= d.shape[0]:
            fact_1 = (
                np.sqrt(5) * d[i * nb_limit : (i + 1) * nb_limit, hess_ind]
                + 10.0
                / 3.0
                * theta_r[0, hess_ind]
                * d[i * nb_limit : (i + 1) * nb_limit, hess_ind] ** 2.0
            )
            fact_2 = (
                1.0
                + np.sqrt(5)
                * theta_r[0, hess_ind]
                * d[i * nb_limit : (i + 1) * nb_limit, hess_ind]
                + 5.0
                / 3.0
                * (theta_r[0, hess_ind] ** 2)
                * (d[i * nb_limit : (i + 1) * nb_limit, hess_ind] ** 2)
            )
            fact_3 = np.sqrt(5) * d[i * nb_limit : (i + 1) * nb_limit, hess_ind]

            r[i * nb_limit : (i + 1) * nb_limit, 0] = (fact_1 / fact_2 - fact_3) * r[
                i * nb_limit : (i + 1) * nb_limit, 0
            ]

            if hess_ind == grad_ind:
                fact_4 = (
                    10.0
                    / 3.0
                    * d[i * nb_limit : (i + 1) * nb_limit, hess_ind] ** 2.0
                    * fact_2
                )
                r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                    (fact_4 - fact_1**2) / (fact_2) ** 2
                ) * M52[i * nb_limit : (i + 1) * nb_limit, 0] + r[
                    i * nb_limit : (i + 1) * nb_limit, 0
                ]

            i += 1
    if derivative_params is not None:
        dx = derivative_params["dx"]

        abs_ = abs(dx)
        sqr = np.square(dx)
        abs_0 = np.dot(abs_, theta)

        dr = np.zeros(dx.shape)

        A = np.zeros((dx.shape[0], 1))
        for i in range(len(abs_0)):
            A[i][0] = np.exp(-np.sqrt(5) * abs_0[i])

        der = np.ones(dx.shape)
        for i in range(len(der)):
            for j in range(n_components):
                if dx[i][j] < 0:
                    der[i][j] = -1

        dB = np.zeros((dx.shape[0], n_components))
        for j in range(dx.shape[0]):
            for k in range(n_components):
                coef = 1
                for ll in range(n_components):
                    if ll != k:
                        coef = coef * (
                            1
                            + np.sqrt(5) * abs_[j][ll] * theta[ll]
                            + (5.0 / 3) * sqr[j][ll] * theta[ll] ** 2
                        )
                dB[j][k] = (
                    np.sqrt(5) * theta[k] * der[j][k]
                    + 2 * (5.0 / 3) * der[j][k] * abs_[j][k] * theta[k] ** 2
                ) * coef

        for j in range(dx.shape[0]):
            for k in range(n_components):
                dr[j][k] = (
                    -np.sqrt(5) * theta[k] * der[j][k] * r[j] + A[j][0] * dB[j][k]
                ).item()

        return r, dr

    return r


def matern32(theta, d, grad_ind=None, hess_ind=None, derivative_params=None):
    """
    Matern 3/2 correlation model.

     Parameters
    ----------
    theta : list[small_d * n_comp]
        Hyperparameters of the correlation model
    d: np.ndarray[n_obs * (n_obs - 1) / 2, n_comp]
        d_i otherwise
    grad_ind : int, optional
        Indice for which component the gradient dr/dtheta must be computed. The default is None.
    hess_ind : int, optional
        Indice for which component the hessian  d²r/d²(theta) must be computed. The default is None.
    derivative_paramas : dict, optional
        List of arguments mandatory to compute the gradient dr/dx. The default is None.

    Raises
    ------
    Exception
        Assure that theta is of the good length

    Returns
    -------
    r: np.ndarray[n_obs * (n_obs - 1) / 2,1]
        An array containing the values of the autocorrelation model.
    """

    r = np.zeros((d.shape[0], 1))
    n_components = d.shape[1]

    # Construct/split the correlation matrix
    i, nb_limit = 0, int(1e4)

    theta_r = theta.reshape(1, n_components)

    while i * nb_limit <= d.shape[0]:
        ll = theta_r * d[i * nb_limit : (i + 1) * nb_limit, :]
        r[i * nb_limit : (i + 1) * nb_limit, 0] = (1.0 + np.sqrt(3.0) * ll).prod(
            axis=1
        ) * np.exp(-np.sqrt(3.0) * (ll.sum(axis=1)))
        i += 1
    i = 0

    M32 = r.copy()

    if grad_ind is not None:
        while i * nb_limit <= d.shape[0]:
            fact_1 = (
                1.0
                / (
                    1.0
                    + np.sqrt(3.0)
                    * theta_r[0, grad_ind]
                    * d[i * nb_limit : (i + 1) * nb_limit, grad_ind]
                )
                - 1.0
            )
            r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                fact_1
                * r[i * nb_limit : (i + 1) * nb_limit, 0]
                * np.sqrt(3.0)
                * d[i * nb_limit : (i + 1) * nb_limit, grad_ind]
            )

            i += 1
        i = 0

    if hess_ind is not None:
        while i * nb_limit <= d.shape[0]:
            fact_2 = (
                1.0
                / (
                    1.0
                    + np.sqrt(3.0)
                    * theta_r[0, hess_ind]
                    * d[i * nb_limit : (i + 1) * nb_limit, hess_ind]
                )
                - 1.0
            )
            r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                r[i * nb_limit : (i + 1) * nb_limit, 0]
                * fact_2
                * np.sqrt(3.0)
                * d[i * nb_limit : (i + 1) * nb_limit, hess_ind]
            )
            if grad_ind == hess_ind:
                fact_3 = (
                    3.0 * d[i * nb_limit : (i + 1) * nb_limit, hess_ind] ** 2.0
                ) / (
                    1.0
                    + np.sqrt(3.0)
                    * theta_r[0, hess_ind]
                    * d[i * nb_limit : (i + 1) * nb_limit, hess_ind]
                ) ** 2.0
                r[i * nb_limit : (i + 1) * nb_limit, 0] = (
                    r[i * nb_limit : (i + 1) * nb_limit, 0]
                    - fact_3 * M32[i * nb_limit : (i + 1) * nb_limit, 0]
                )
            i += 1
    if derivative_params is not None:
        dx = derivative_params["dx"]

        abs_ = abs(dx)
        abs_0 = np.dot(abs_, theta)
        dr = np.zeros(dx.shape)

        A = np.zeros((dx.shape[0], 1))
        for i in range(len(abs_0)):
            A[i][0] = np.exp(-np.sqrt(3) * abs_0[i])

        der = np.ones(dx.shape)
        for i in range(len(der)):
            for j in range(n_components):
                if dx[i][j] < 0:
                    der[i][j] = -1

        dB = np.zeros((dx.shape[0], n_components))
        for j in range(dx.shape[0]):
            for k in range(n_components):
                coef = 1
                for ll in range(n_components):
                    if ll != k:
                        coef = coef * (1 + np.sqrt(3) * abs_[j][ll] * theta[ll])
                dB[j][k] = np.sqrt(3) * theta[k] * der[j][k] * coef

        for j in range(dx.shape[0]):
            for k in range(n_components):
                dr[j][k] = (
                    -np.sqrt(3) * theta[k] * der[j][k] * r[j] + A[j][0] * dB[j][k]
                ).item()
        return r, dr

    return r


def act_exp(theta, d, grad_ind=None, hess_ind=None, d_x=None, derivative_params=None):
    """
    Active learning exponential correlation model

    Parameters
    ----------
    theta : list[small_d * n_comp]
        Hyperparameters of the correlation model
    d: np.ndarray[n_obs * (n_obs - 1) / 2, n_comp]
        d_i otherwise
    grad_ind : int, optional
        Indice for which component the gradient dr/dtheta must be computed. The default is None.
    hess_ind : int, optional
        Indice for which component the hessian  d²r/d²(theta) must be computed. The default is None.
    derivative_paramas : dict, optional
        List of arguments mandatory to compute the gradient dr/dx. The default is None.

    Raises
    ------
    Exception
        Assure that theta is of the good length

    Returns
    -------
    r: np.ndarray[n_obs * (n_obs - 1) / 2,1]
        An array containing the values of the autocorrelation model.
    """

    r = np.zeros((d.shape[0], 1))
    n_components = d.shape[1]

    if len(theta) % n_components != 0:
        raise Exception("Length of theta must be a multiple of n_components")

    n_small_components = len(theta) // n_components

    A = np.reshape(theta, (n_small_components, n_components)).T

    d_A = d.dot(A)

    # Necessary when working in embeddings space
    if d_x is not None:
        d = d_x
        n_components = d.shape[1]

    r[:, 0] = np.exp(-(1 / 2) * np.sum(d_A**2.0, axis=1))

    if grad_ind is not None:
        d_grad_ind = grad_ind % n_components
        d_A_grad_ind = grad_ind // n_components

        if hess_ind is None:
            r[:, 0] = -d[:, d_grad_ind] * d_A[:, d_A_grad_ind] * r[:, 0]

        elif hess_ind is not None:
            d_hess_ind = hess_ind % n_components
            d_A_hess_ind = hess_ind // n_components
            fact = -d_A[:, d_A_grad_ind] * d_A[:, d_A_hess_ind]
            if d_A_hess_ind == d_A_grad_ind:
                fact = 1 + fact
            r[:, 0] = -d[:, d_grad_ind] * d[:, d_hess_ind] * fact * r[:, 0]

    if derivative_params is not None:
        raise ValueError("Jacobians are not available for this correlation kernel")

    return r


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
        if corr == "squar_sin_exp":
            raise ValueError(
                "Spatial derivatives for ExpSinSquared not available yet (to implement)."
            )
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


@njit_use(parallel=True)
def matrix_data_corr_levels_cat_matrix(
    i, n_levels, theta_cat, theta_bounds, is_ehh: bool
):
    Theta_mat = np.zeros((n_levels[i], n_levels[i]), dtype=np.float64)
    L = np.zeros((n_levels[i], n_levels[i]))
    v = 0
    for j in range(n_levels[i]):
        for k in range(n_levels[i] - j):
            if j == k + j:
                Theta_mat[j, k + j] = 1.0
            else:
                Theta_mat[j, k + j] = theta_cat[v].item()
                Theta_mat[k + j, j] = theta_cat[v].item()
                v = v + 1

    for j in range(n_levels[i]):
        for k in range(n_levels[i] - j):
            if j == k + j:
                if j == 0:
                    L[j, k + j] = 1

                else:
                    L[j, k + j] = 1
                    for ll in range(j):
                        L[j, k + j] = L[j, k + j] * np.sin(Theta_mat[j, ll])

            else:
                if j == 0:
                    L[k + j, j] = np.cos(Theta_mat[k, 0])
                else:
                    L[k + j, j] = np.cos(Theta_mat[k + j, j])
                    for ll in range(j):
                        L[k + j, j] = L[k + j, j] * np.sin(Theta_mat[k + j, ll])

    T = np.dot(L, L.T)

    if is_ehh:
        T = (T - 1) * theta_bounds[1] / 2
        T = np.exp(2 * T)
    k = (1 + np.exp(-theta_bounds[1])) / np.exp(-theta_bounds[0])
    T = (T + np.exp(-theta_bounds[1])) / (k)
    return T


@njit_use()
def matrix_data_corr_levels_cat_mod(i, Lij, r_cat, T, has_cat_kernel):
    for k in range(np.shape(Lij[i])[0]):
        indi = int(Lij[i][k][0])
        indj = int(Lij[i][k][1])

        if indi == indj:
            r_cat[k] = 1.0
        else:
            if has_cat_kernel:
                r_cat[k] = T[indi, indj]


@njit_use()
def matrix_data_corr_levels_cat_mod_comps(
    i, Lij, r_cat, n_levels, T, d_cat_i, has_cat_kernel
):
    for k in range(np.shape(Lij[i])[0]):
        indi = int(Lij[i][k][0])
        indj = int(Lij[i][k][1])

        if indi == indj:
            r_cat[k] = 1.0
        else:
            if has_cat_kernel:
                Theta_i_red = np.zeros(int((n_levels[i] - 1) * n_levels[i] / 2))
                indmatvec = 0
                for j in range(n_levels[i]):
                    for ll in range(n_levels[i]):
                        if ll > j:
                            Theta_i_red[indmatvec] = T[j, ll]
                            indmatvec += 1
                kval_cat = 0
                for indijk in range(len(Theta_i_red)):
                    kval_cat += np.multiply(
                        Theta_i_red[indijk], d_cat_i[k : k + 1][0][indijk]
                    )
                r_cat[k] = kval_cat
