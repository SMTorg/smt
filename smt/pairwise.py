# -*- coding: utf-8 -*-
# Mohamed Amine Bouhlel <mbouhlel@umich.edu>
# This submodule is from sklearn 0.14 toolbox

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Robert Layton <robertlayton@gmail.com>
#          Andreas Mueller <amueller@ais.uni-bonn.de>
# License: BSD 3 clause

import numpy as np
from scipy.sparse import issparse

from smt.validation import atleast2d_or_csr

def check_pairwise_arrays(X, Y):
    """ Set X and Y appropriately and checks inputs

    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats. Finally, the function checks that the size
    of the second dimension of the two arrays is equal.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples_a, n_features]

    Y : {array-like, sparse matrix}, shape = [n_samples_b, n_features]

    Returns
    -------
    safe_X : {array-like, sparse matrix}, shape = [n_samples_a, n_features]
        An array equal to X, guaranteed to be a numpy array.

    safe_Y : {array-like, sparse matrix}, shape = [n_samples_b, n_features]
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.

    """
    if Y is X or Y is None:
        X = Y = atleast2d_or_csr(X)
    else:
        X = atleast2d_or_csr(X)
        Y = atleast2d_or_csr(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))

    if not (X.dtype == Y.dtype == np.float32):
        if Y is X:
            X = Y = X.astype(np.float)
        else:
            X = X.astype(np.float)
            Y = Y.astype(np.float)
    return X, Y

def manhattan_distances(X, Y=None, sum_over_features=True,
                        size_threshold=5e8):
    """ Compute the L1 distances between the vectors in X and Y.

    With sum_over_features equal to False it returns the componentwise
    distances.

    Parameters
    ----------
    X : array_like
        An array with shape (n_samples_X, n_features).

    Y : array_like, optional
        An array with shape (n_samples_Y, n_features).

    sum_over_features : bool, default=True
        If True the function returns the pairwise distance matrix
        else it returns the componentwise L1 pairwise-distances.

    size_threshold : int, default=5e8
        Avoid creating temporary matrices bigger than size_threshold (in
        bytes). If the problem size gets too big, the implementation then
        breaks it down in smaller problems.

    Returns
    -------
    D : array
        If sum_over_features is False shape is
        (n_samples_X * n_samples_Y, n_features) and D contains the
        componentwise L1 pairwise-distances (ie. absolute difference),
        else shape is (n_samples_X, n_samples_Y) and D contains
        the pairwise l1 distances.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import manhattan_distances
    >>> manhattan_distances(3, 3)#doctest:+ELLIPSIS
    array([[ 0.]])
    >>> manhattan_distances(3, 2)#doctest:+ELLIPSIS
    array([[ 1.]])
    >>> manhattan_distances(2, 3)#doctest:+ELLIPSIS
    array([[ 1.]])
    >>> manhattan_distances([[1, 2], [3, 4]],\
         [[1, 2], [0, 3]])#doctest:+ELLIPSIS
    array([[ 0.,  2.],
           [ 4.,  4.]])
    >>> import numpy as np
    >>> X = np.ones((1, 2))
    >>> y = 2 * np.ones((2, 2))
    >>> manhattan_distances(X, y, sum_over_features=False)#doctest:+ELLIPSIS
    array([[ 1.,  1.],
           [ 1.,  1.]]...)
    """
    if issparse(X) or issparse(Y):
        raise ValueError("manhattan_distance does not support sparse"
                         " matrices.")
    X, Y = check_pairwise_arrays(X, Y)
    temporary_size = X.size * Y.shape[-1]
    # Convert to bytes
    temporary_size *= X.itemsize
    if temporary_size > size_threshold and sum_over_features:
        # Broadcasting the full thing would be too big: it's on the order
        # of magnitude of the gigabyte
        D = np.empty((X.shape[0], Y.shape[0]), dtype=X.dtype)
        index = 0
        increment = 1 + int(size_threshold / float(temporary_size) *
                            X.shape[0])
        while index < X.shape[0]:
            this_slice = slice(index, index + increment)
            tmp = X[this_slice, np.newaxis, :] - Y[np.newaxis, :, :]
            tmp = np.abs(tmp, tmp)
            tmp = np.sum(tmp, axis=2)
            D[this_slice] = tmp
            index += increment
    else:
        D = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
        D = np.abs(D, D)
        if sum_over_features:
            D = np.sum(D, axis=2)
        else:
            D = D.reshape((-1, X.shape[1]))
    return D
