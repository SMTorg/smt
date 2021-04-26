"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

import numpy as np
from packaging import version
from sklearn import __version__ as sklversion

if version.parse(sklversion) < version.parse("0.22"):
    from sklearn.cross_decomposition.pls_ import PLSRegression as pls
else:
    from sklearn.cross_decomposition import PLSRegression as pls

from pyDOE2 import bbdesign
from sklearn.metrics.pairwise import check_pairwise_arrays

# TODO: Create hyperclass Kernels and a class for each kernel


def standardization(X, y, scale_X_to_unit=False):

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

    scale_X_to_unit: bool
            - We substract the mean from each variable and then divide the values
              of each variable by its standard deviation (scale_X_to_unit=False).
            - We scale X to the unit hypercube [0,1]^dim (scale_X_to_unit=True).

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
            The standard deviation (or the difference between the max and the
            min if scale_X_to_unit=True) of each input variable.

    y_std:  list(1)
            The standard deviation of the output variable.

    """

    if scale_X_to_unit:
        X_offset = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_scale = X_max - X_offset
    else:
        X_offset = np.mean(X, axis=0)
        X_scale = X.std(axis=0, ddof=1)
        X_scale[X_scale == 0.0] = 1.0

    y_mean = np.mean(y, axis=0)
    y_std = y.std(axis=0, ddof=1)
    y_std[y_std == 0.0] = 1.0

    # scale X and y
    X = (X - X_offset) / X_scale
    y = (y - y_mean) / y_std
    return X, y, X_offset, y_mean, X_scale, y_std


def cross_distances(X):

    """
    Computes the nonzero componentwise cross-distances between the vectors
    in X.

    Parameters
    ----------

    X: np.ndarray [n_obs, dim]
            - The input variables.

    Returns
    -------

    D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The cross-distances between the vectors in X.

    ij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
            - The indices i and j of the vectors in X associated to the cross-
              distances in D.
    """

    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int32)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0

    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = X[k] - X[(k + 1) : n_samples]

    return D, ij.astype(np.int32)


def gower_distances(X, y=None):
    """
    Some parts were extracted from gower 0.0.5 library
    Computes the nonzero Gower-distances between the vectors
    in X.
    Parameters
    ----------
    X: np.ndarray [n_obs, dim]
            - The input variables.
    Returns
    -------
    D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The gower distances between the vectors in X.
    ij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
            - The indices i and j of the vectors in X associated to the cross-
              distances in D.
    """
    # function checks
    if y is None:
        Y = X
    else:
        Y = y
    if not isinstance(X, np.ndarray):
        if not np.array_equal(X.columns, Y.columns):
            raise TypeError("X and Y must have same columns!")
    else:
        if not X.shape[1] == Y.shape[1]:
            raise TypeError("X and Y must have same y-dim!")

    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape

    if not isinstance(X, np.ndarray):
        is_number = np.vectorize(lambda x: not np.issubdtype(x, np.number))
        cat_features = is_number(X.dtypes)
    else:
        cat_features = np.zeros(x_n_cols, dtype=bool)
        for col in range(x_n_cols):
            if not np.issubdtype(type(X[0, col]), np.number):
                cat_features[col] = True

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)

    Z = np.concatenate((X, Y))

    x_index = range(0, x_n_rows)
    y_index = range(x_n_rows, x_n_rows + y_n_rows)

    Z_num = Z[:, np.logical_not(cat_features)]

    num_cols = Z_num.shape[1]
    num_ranges = np.zeros(num_cols)
    num_max = np.zeros(num_cols)

    for col in range(num_cols):
        col_array = Z_num[:, col].astype(np.float32)
        max = np.nanmax(col_array)
        min = np.nanmin(col_array)

        if np.isnan(max):
            max = 0.0
        if np.isnan(min):
            min = 0.0
        num_max[col] = max
        num_ranges[col] = (1 - min / max) if (max != 0) else 0.0

    # This is to normalize the numeric values between 0 and 1.
    Z_num = np.divide(Z_num, num_max, out=np.zeros_like(Z_num), where=num_max != 0)

    Z_cat = Z[:, cat_features]

    X_cat = Z_cat[
        x_index,
    ]
    X_num = Z_num[
        x_index,
    ]
    Y_cat = Z_cat[
        y_index,
    ]
    Y_num = Z_num[
        y_index,
    ]

    n_samples, n_features = X_num.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D_num = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0

    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        abs_delta = np.abs(X_num[k] - Y_num[(k + 1) : n_samples])
        try:
            D_num[ll_0:ll_1] = np.divide(
                abs_delta,
                num_ranges,
                out=np.zeros_like(abs_delta),
                where=num_ranges != 0,
            )
        except:
            pass

    n_samples, n_features = X_cat.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
    D_cat = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0

    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        D_cat[ll_0:ll_1] = np.where(
            X_cat[k] == Y_cat[(k + 1) : n_samples],
            np.zeros_like(X_cat[k]),
            np.ones_like(X_cat[k]),
        )

    D = np.concatenate((D_cat, D_num), axis=1)

    return D, ij.astype(np.int)


def gower_corr(theta, d):

    """
    Gower autocorrelation model.
    Parameters
    ----------
    theta : list[ncomp]
        the autocorrelation parameter(s).
    d: np.ndarray[n_obs * (n_obs - 1) / 2, n_comp]
        |d_i * coeff_pls_i| if PLS is used, |d_i| otherwise
    Returns
    -------
    r : np.ndarray[n_obs * (n_obs - 1) / 2,1]
        An array containing the values of the autocorrelation model.
    """

    r = np.zeros((d.shape[0], 1))
    n_components = d.shape[1]

    # Construct/split the correlation matrix
    i, nb_limit = 0, int(1e4)
    while True:
        if i * nb_limit > d.shape[0]:
            return r
        else:
            r[i * nb_limit : (i + 1) * nb_limit, 0] = np.exp(
                np.divide(
                    -np.sum(
                        theta.reshape(1, n_components)
                        * d[i * nb_limit : (i + 1) * nb_limit, :],
                        axis=1,
                    ),
                    theta.sum(),
                )
            )
            i += 1


def gower_matrix(data_x, data_y=None, weight=None, cat_features=None):
    "this function was copied from gower 0.0.5 code"
    # function checks
    X = data_x
    if data_y is None:
        Y = data_x
    else:
        Y = data_y
    if not isinstance(X, np.ndarray):
        if not np.array_equal(X.columns, Y.columns):
            raise TypeError("X and Y must have same columns!")
    else:
        if not X.shape[1] == Y.shape[1]:
            raise TypeError("X and Y must have same y-dim!")

    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape

    if cat_features is None:
        if not isinstance(X, np.ndarray):
            is_number = np.vectorize(lambda x: not np.issubdtype(x, np.number))
            cat_features = is_number(X.dtypes)
        else:
            cat_features = np.zeros(x_n_cols, dtype=bool)
            for col in range(x_n_cols):
                if not np.issubdtype(type(X[0, col]), np.number):
                    cat_features[col] = True
    else:
        cat_features = np.array(cat_features)

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)

    Z = np.concatenate((X, Y))

    x_index = range(0, x_n_rows)
    y_index = range(x_n_rows, x_n_rows + y_n_rows)

    Z_num = Z[:, np.logical_not(cat_features)]

    num_cols = Z_num.shape[1]
    num_ranges = np.zeros(num_cols)
    num_max = np.zeros(num_cols)

    for col in range(num_cols):
        col_array = Z_num[:, col].astype(np.float32)
        max = np.nanmax(col_array)
        min = np.nanmin(col_array)

        if np.isnan(max):
            max = 0.0
        if np.isnan(min):
            min = 0.0
        num_max[col] = max
        num_ranges[col] = (1 - min / max) if (max != 0) else 0.0

    # This is to normalize the numeric values between 0 and 1.
    Z_num = np.divide(Z_num, num_max, out=np.zeros_like(Z_num), where=num_max != 0)
    Z_cat = Z[:, cat_features]

    if weight is None:
        weight = np.ones(Z.shape[1])

    weight_cat = weight[cat_features]
    weight_num = weight[np.logical_not(cat_features)]

    out = np.zeros((x_n_rows, y_n_rows), dtype=np.float32)

    weight_sum = weight.sum()

    X_cat = Z_cat[
        x_index,
    ]
    X_num = Z_num[
        x_index,
    ]
    Y_cat = Z_cat[
        y_index,
    ]
    Y_num = Z_num[
        y_index,
    ]

    for i in range(x_n_rows):
        j_start = i
        if x_n_rows != y_n_rows:
            j_start = 0
        # call the main function
        res = gower_get(
            X_cat[i, :],
            X_num[i, :],
            Y_cat[j_start:y_n_rows, :],
            Y_num[j_start:y_n_rows, :],
            weight_cat,
            weight_num,
            weight_sum,
            cat_features,
            num_ranges,
            num_max,
        )
        out[i, j_start:] = res
        if x_n_rows == y_n_rows:
            out[i:, j_start] = res

    return out


def gower_get(
    xi_cat,
    xi_num,
    xj_cat,
    xj_num,
    feature_weight_cat,
    feature_weight_num,
    feature_weight_sum,
    categorical_features,
    ranges_of_numeric,
    max_of_numeric,
):
    "this function was copied from gower 0.0.5 code"
    # categorical columns
    sij_cat = np.where(xi_cat == xj_cat, np.zeros_like(xi_cat), np.ones_like(xi_cat))
    sum_cat = np.multiply(feature_weight_cat, sij_cat).sum(axis=1)

    # numerical columns
    abs_delta = np.absolute(xi_num - xj_num)
    sij_num = np.divide(
        abs_delta,
        ranges_of_numeric,
        out=np.zeros_like(abs_delta),
        where=ranges_of_numeric != 0,
    )

    sum_num = np.multiply(feature_weight_num, sij_num).sum(axis=1)
    sums = np.add(sum_cat, sum_num)
    sum_sij = np.divide(sums, feature_weight_sum)

    return sum_sij


def differences(X, Y):
    X, Y = check_pairwise_arrays(X, Y)
    D = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    return D.reshape((-1, X.shape[1]))


def abs_exp(theta, d, grad_ind=None, hess_ind=None, derivative_params=None):
    """
    Absolute exponential autocorrelation model.
    (Ornstein-Uhlenbeck stochastic process)::

    Parameters
    ----------
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


def squar_exp(theta, d, grad_ind=None, hess_ind=None, derivative_params=None):

    """
    Squared exponential correlation model.

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
            1.0 + np.sqrt(5.0) * ll + 5.0 / 3.0 * ll ** 2.0
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
                    ((fact_4 - fact_1 ** 2) / (fact_2) ** 2)
                    * M52[i * nb_limit : (i + 1) * nb_limit, 0]
                    + r[i * nb_limit : (i + 1) * nb_limit, 0]
                )

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
                for l in range(n_components):
                    if l != k:
                        coef = coef * (
                            1
                            + np.sqrt(5) * abs_[j][l] * theta[l]
                            + (5.0 / 3) * sqr[j][l] * theta[l] ** 2
                        )
                dB[j][k] = (
                    np.sqrt(5) * theta[k] * der[j][k]
                    + 2 * (5.0 / 3) * der[j][k] * abs_[j][k] * theta[k] ** 2
                ) * coef

        for j in range(dx.shape[0]):
            for k in range(n_components):
                dr[j][k] = (
                    -np.sqrt(5) * theta[k] * der[j][k] * r[j] + A[j][0] * dB[j][k]
                )

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
                    (3.0 * d[i * nb_limit : (i + 1) * nb_limit, hess_ind] ** 2.0)
                    / (
                        1.0
                        + np.sqrt(3.0)
                        * theta_r[0, hess_ind]
                        * d[i * nb_limit : (i + 1) * nb_limit, hess_ind]
                    )
                    ** 2.0
                )
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
                for l in range(n_components):
                    if l != k:
                        coef = coef * (1 + np.sqrt(3) * abs_[j][l] * theta[l])
                dB[j][k] = np.sqrt(3) * theta[k] * der[j][k] * coef

        for j in range(dx.shape[0]):
            for k in range(n_components):
                dr[j][k] = (
                    -np.sqrt(3) * theta[k] * der[j][k] * r[j] + A[j][0] * dB[j][k]
                )
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

    r[:, 0] = np.exp(-(1 / 2) * np.sum(d_A ** 2.0, axis=1))

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

    y: np.ndarray [n_obs,1]
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
    yy = np.empty(shape=(0, 1))
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
                yy = np.vstack((yy, y[i, 0]))
                yy[-1, 0] += (
                    pts[None][1 + ii][1][i, 0]
                    * delta_x
                    * (xlimits[ii, 1] - xlimits[ii, 0])
                )
    return np.abs(coeff_pls).mean(axis=0), XX, yy


def componentwise_distance(D, corr, dim, theta=None, return_derivative=False):

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
    # Fit the matrix iteratively: avoid some memory troubles .
    limit = int(1e4)

    D_corr = np.zeros((D.shape[0], dim))
    i, nb_limit = 0, int(limit)
    if return_derivative == False:
        while True:
            if i * nb_limit > D_corr.shape[0]:
                return D_corr
            else:
                if corr == "squar_exp":
                    D_corr[i * nb_limit : (i + 1) * nb_limit, :] = (
                        D[i * nb_limit : (i + 1) * nb_limit, :] ** 2
                    )
                elif corr == "act_exp":
                    D_corr[i * nb_limit : (i + 1) * nb_limit, :] = D[
                        i * nb_limit : (i + 1) * nb_limit, :
                    ]
                else:
                    # abs_exp or matern
                    D_corr[i * nb_limit : (i + 1) * nb_limit, :] = np.abs(
                        D[i * nb_limit : (i + 1) * nb_limit, :]
                    )
                i += 1
    else:
        if theta is None:
            raise ValueError(
                "Missing theta to compute spatial derivative of theta cross-spatial correlation distance"
            )
        if corr == "squar_exp":
            D_corr = 2 * np.einsum("j,ij->ij", theta.T, D)
            return D_corr
        elif corr == "act_exp":
            raise ValueError("this option is not implemented for active learning")
        else:
            # abs_exp or matern
            # derivative of absolute value : +1/-1
            der = np.ones(D.shape)
            for i, j in np.ndindex(D.shape):
                if D[i][j] < 0:
                    der[i][j] = -1

            D_corr = np.einsum("j,ij->ij", theta.T, der)
            return D_corr

        i += 1


def componentwise_distance_PLS(
    D, corr, n_comp, coeff_pls, theta=None, return_derivative=False
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

    D_corr = np.zeros((D.shape[0], n_comp))
    i, nb_limit = 0, int(limit)
    if return_derivative == False:
        while True:
            if i * nb_limit > D_corr.shape[0]:
                return D_corr
            else:
                if corr == "squar_exp":
                    D_corr[i * nb_limit : (i + 1) * nb_limit, :] = np.dot(
                        D[i * nb_limit : (i + 1) * nb_limit, :] ** 2, coeff_pls ** 2
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
                for l in range(n_comp):
                    coef = coef + theta[l] * coeff_pls[j][l] ** 2
                coef = 2 * coef
                D_corr[i][j] = coef * D[i][j]
            return D_corr

        else:
            # abs_exp
            D_corr = np.zeros(np.shape(D))
            der = np.ones(np.shape(D))
            for i, j in np.ndindex(D.shape):
                if D[i][j] < 0:
                    der[i][j] = -1
                coef = 0
                for l in range(n_comp):
                    coef = coef + theta[l] * np.abs(coeff_pls[j][l])
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
