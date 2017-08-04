import numpy as np

"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>

The kriging-correlation model functions.
"""


def standardization(X,y,copy=False):

    """
    We substract the mean from each variable. Then, we divide the values of each
    variable by its standard deviation.

    Parameters
    ----------

    X: np.ndarray [n_obs, dim]
            - The input variables.

    y: np.ndarray [n_obs, 1]
            - The output variable.

    copy: bool
            - A copy of matrices X and y will be used (copy = True).
            - Matrices X and y will be used. The matrices X and y will be
              normalized (copy = False).
            - (copy = False by default).

    Returns
    -------

    X: np.ndarray [n_obs, dim]
          The standardized input matrix.

    y: np.ndarray [n_obs, 1]
          The standardized output vector.

    X_mean: list(dim)
            The mean of each input variable.

    y_mean: list(1)
            The mean of the output variable.

    X_std:  list(dim)
            The standard deviation of each input variable.

    y_std:  list(1)
            The standard deviation of the output variable.

    """
    X_mean = np.mean(X, axis=0)
    X_std = X.std(axis=0,ddof=1)
    y_mean = np.mean(y, axis=0)
    y_std = y.std(axis=0,ddof=1)
    X_std[X_std == 0.] = 1.
    y_std[y_std == 0.] = 1.

    # center and scale X
    if copy:
        Xr = (X.copy() - X_mean) / X_std
        yr = (y.copy() - y_mean) / y_std
        return Xr, yr, X_mean, y_mean, X_std, y_std

    else:
        X = (X - X_mean) / X_std
        y = (y - y_mean) / y_std
        return X, y, X_mean, y_mean, X_std, y_std


def l1_cross_distances(X):

    """
    Computes the nonzero componentwise L1 cross-distances between the vectors
    in X.

    Parameters
    ----------

    X: np.ndarray [n_obs, dim]
            - The input variables.

    Returns
    -------

    D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The L1 cross-distances between the vectors in X.

    ij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
            - The indices i and j of the vectors in X associated to the cross-
              distances in D.
    """

    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0

    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = np.abs(X[k] - X[(k + 1):n_samples])

    return D, ij.astype(np.int)


def abs_exp(theta, d):

    """
    Absolute exponential autocorrelation model.
    (Ornstein-Uhlenbeck stochastic process)::

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

    r = np.zeros((d.shape[0],1))
    n_components = d.shape[1]

    # Construct/split the correlation matrix
    i,nb_limit  = 0,int(1e4)
    while True:
        if i * nb_limit > d.shape[0]:
            return r
        else:
            r[i*nb_limit:(i+1)*nb_limit,0] = np.exp(-np.sum(theta.reshape(1,
                    n_components) * d[i*nb_limit:(i+1)*nb_limit,:], axis=1))
            i+=1


def squar_exp(theta, d):

    """
    Squared exponential correlation model.

    Parameters
    ----------
    theta : list[ncomp]
        the autocorrelation parameter(s).

    d: np.ndarray[n_obs * (n_obs - 1) / 2, n_comp]
        |d_i * coeff_pls_i| if PLS is used, |d_i| otherwise

    Returns
    -------
    r: np.ndarray[n_obs * (n_obs - 1) / 2,1]
        An array containing the values of the autocorrelation model.
    """

    r = np.zeros((d.shape[0],1))
    n_components = d.shape[1]

    # Construct/split the correlation matrix
    i,nb_limit  = 0,int(1e4)

    while True:
        if i * nb_limit > d.shape[0]:
            return r
        else:
            r[i*nb_limit:(i+1)*nb_limit,0] = np.exp(-np.sum(theta.reshape(1,
                    n_components) * d[i*nb_limit:(i+1)*nb_limit,:], axis=1))
            i+=1


"""
The built-in regression models subroutine for the KPLS module.
"""

def constant(x):

    """
    Zero order polynomial (constant, p = 1) regression model.

    x --> f(x) = 1

    Parameters
    ----------
    x: np.ndarray[n_obs,dim]
            - An array giving the locations x at which the regression model
              should be evaluated.

    Returns
    -------
    f: np.ndarray[n_obs,p]
            - An array with the values of the regression model.
    """

    x = np.asarray(x, dtype=np.float)
    n_eval = x.shape[0]
    f = np.ones([n_eval, 1])

    return f


def linear(x):
    """
    First order polynomial (linear, p = n+1) regression model.

    x --> f(x) = [ 1, x_1, ..., x_n ].T

    Parameters
    ----------
    x: np.ndarray[n_obs,dim]
            - An array giving the locations x at which the regression model
              should be evaluated.

    Returns
    -------
    f: np.ndarray[n_obs,p]
            - An array with the values of the regression model.
    """

    x = np.asarray(x, dtype=np.float)
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
    x: np.ndarray[n_obs,dim]
            - An array giving the locations x at which the regression model
              should be evaluated.

    Returns
    -------
    f: np.ndarray[n_obs,p]
            - An array with the values of the regression model.
    """

    x = np.asarray(x, dtype=np.float)
    n_eval, n_features = x.shape
    f = np.hstack([np.ones([n_eval, 1]), x])
    for k in range(n_features):
        f = np.hstack([f, x[:, k, np.newaxis] * x[:, k:]])

    return f
