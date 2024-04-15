"""
Authors : Morgane Menz / Alexandre Thouvenot
Some parts are copied from KrgBased SMT class
"""

import numpy as np
from scipy import linalg

from smt.utils.kriging import differences


def covariance_matrix(krg, X, conditioned=True):
    """
    This function computes the covariance matrix (with conditioned kernel or not) at point(s) X.

    Parameters
    ----------
    krg : SMT KRG
        SMT kriging model already trained
    X : array_like
        Array with shape (n_eval, n_feature) composed of the point(s) on which the
        covariance matrix will be computed
    conditioned : Boolean
        Option to whether computes covariance matrix with conditioned kernel or not.

    Returns
    -------
    cov_matrix : array_like
        if conditioned: Array with shape(n_eval, n_eval) composed of the value(s) of the
        conditioned kernel
    cov_matrix : array_like
        if not conditioned: Array with shape(n_eval, n_eval) composed of the value(s) of the
        non conditioned kernel
    """
    X_cont = (X - krg.X_offset) / krg.X_scale
    d = differences(X_cont, Y=krg.X_norma.copy())
    cross_d = differences(X_cont, Y=X_cont)

    C = krg.optimal_par["C"]
    theta = krg.optimal_theta
    n_eval = X.shape[0]

    k = krg._correlation_types[krg.options["corr"]](
        theta, krg._componentwise_distance(cross_d)
    ).reshape(n_eval, n_eval)
    if not conditioned:
        cov_matrix = krg.optimal_par["sigma2"] * k
        return cov_matrix

    r = krg._correlation_types[krg.options["corr"]](
        theta, krg._componentwise_distance(d)
    ).reshape(n_eval, -1)
    rt = linalg.solve_triangular(C, r.T, lower=True)

    u = linalg.solve_triangular(
        krg.optimal_par["G"].T,
        np.dot(krg.optimal_par["Ft"].T, rt)
        - krg._regression_types[krg.options["poly"]](X_cont).T,
    )

    cov_matrix = krg.optimal_par["sigma2"] * (k - rt.T.dot(rt) + u.T.dot(u))

    return cov_matrix


def sample_trajectory(krg, X, n_traj, method="eigen", eps=10 ** (-10)):
    """
    This function samples gaussian process trajectories with eigen decomposition or Cholesky decomposition.

    Parameters
    ----------
    krg : SMT KRG
        SMT kriging model already trained
    X : array_like
        Array with shape (n_eval, n_feature) composed of the point(s) on which the
        trajctories will be sampled
    n_traj : int
        Number of sampled trajectories
    method : string
        Option to whether samples trajectories with eigen or Cholesky method. Cholesky decomposition
        might not be possible because of bad conditioning
    eps : float
        Threshold used to floor negative eigen value

    Returns
    -------
    traj : array_like
        Array with shape(n_eval, n_traj) composed of the sampled trajectorie with a chosen
        decomposition method
    """
    n_eval = X.shape[0]
    cov = covariance_matrix(krg, X)
    if method == "eigen":
        v, w = np.linalg.eigh(cov)
        v[np.abs(v) < eps] = np.zeros_like(v[np.abs(v) < eps])
        C = w.dot(np.diagflat(np.sqrt(v)))
    if method == "cholesky":
        C = np.linalg.cholesky(cov)
    mean_ = krg._predict_values(X).reshape(-1, 1)
    traj = (mean_ + C.dot(np.random.randn(n_eval, n_traj))).reshape(n_eval, n_traj)

    return traj


def gauss_legendre_grid(bounds, n_point):
    """
    This function creates a grid to computes integrals with Gauss-Legendre quadrature method.

    Parameters
    ----------
    bounds : array_like
        Array with shape (2, int_dims) where dims is the number of integration dimension.
        It iscontaining the integration domain where bounds[0] is the inferior integration
        bound and bounds[1] is the superior integration bound.
    n_point : int
        Number of point in each dimension

    Returns
    -------
    x_grid : array_like
        Array with shape(n_point**int_dims, int_dims) composed of point(s) on which the
        integral will be computed
    weights_grid : array_like
        Array with shape(n_point**int_dims, 1) composed of weigths used to computed the
        integral
    """
    dims = bounds.shape[1]
    x, weights = np.polynomial.legendre.leggauss(n_point)
    x_list = [
        (bounds[1][i] - bounds[0][i]) / 2 * x + (bounds[1][i] + bounds[0][i]) / 2
        for i in range(dims)
    ]
    x_grid = np.meshgrid(*x_list)
    x_grid = np.concatenate([x_grid[i].reshape(-1, 1) for i in range(dims)], axis=1)
    weights_grid = np.meshgrid(*[weights] * dims)
    weights_grid = np.prod(weights_grid, axis=0).reshape(-1, 1)
    return x_grid, weights_grid


def rectangular_grid(bounds, n_point):
    """
    This function creates a grid to computes integrals with rectangular grid method.

    Parameters
    ----------
    bounds : array_like
        Array with shape (2, int_dims) where dims is the number of integration dimension.
        It iscontaining the integration domain where bounds[0] is the inferior integration
        bound and bounds[1] is the superior integration bound.
    n_point : int
        Number of point in each dimension

    Returns
    -------
    x_grid : array_like
        Array with shape(n_point**int_dims, int_dims) composed of point(s) on which the
        integral will be computed
    weights_grid : array_like
        Array with shape(n_point**int_dims, 1) composed of weigths used to computed the
        integral
    """
    dims = bounds.shape[1]
    x, weights = np.linspace(-1, 1, n_point), np.ones(n_point) / n_point
    x_list = [
        (bounds[1][i] - bounds[0][i]) / 2 * x + (bounds[1][i] + bounds[0][i]) / 2
        for i in range(dims)
    ]
    x_grid = np.meshgrid(*x_list)
    x_grid = np.concatenate([x_grid[i].reshape(-1, 1) for i in range(dims)], axis=1)
    weights_grid = np.meshgrid(*[weights] * dims)
    weights_grid = np.prod(weights_grid, axis=0).reshape(-1, 1)
    return x_grid, weights_grid


def simpson_weigths(n_points, h):
    """
    This function computes Simpson quadrature weigths in one dimension.

    Parameters
    ----------
    n_point : int
        Number of weigths
    h : float
        Scaling coefficient

    Returns
    -------
    weights : array_like
        Array with shape(n_point,) composed of weigths used to computed the
        integral
    """
    weights = np.zeros(n_points)
    for i in range((n_points + 1) // 2):
        weights[2 * i] = 2.0 * h / 3.0
        weights[2 * i - 1] = 4.0 * h / 3.0
    weights[0], weights[-1] = h / 3.0, h / 3.0
    return weights


def simpson_grid(bounds, n_point):
    """
    This function creates a grid to computes integrals with Simpson quadrature.

    Parameters
    ----------
    bounds : array_like
        Array with shape (2, int_dims) where dims is the number of integration dimension.
        It iscontaining the integration domain where bounds[0] is the inferior integration
        bound and bounds[1] is the superior integration bound.
    n_point : int
        Number of point in each dimension

    Returns
    -------
    x_grid : array_like
        Array with shape (n_point**int_dims, int_dims) composed of point(s) on which the
        integral will be computed
    weights_grid : array_like
        Array with shape (n_point**int_dims, 1) composed of weigths used to computed the
        integral
    """
    dims = bounds.shape[1]
    x = np.linspace(-1, 1, n_point)
    x_list = [
        (bounds[1][i] - bounds[0][i]) / 2 * x + (bounds[1][i] + bounds[0][i]) / 2
        for i in range(dims)
    ]
    x_grid = np.meshgrid(*x_list)
    x_grid = np.concatenate([x_grid[i].reshape(-1, 1) for i in range(dims)], axis=1)
    weights_grid = np.meshgrid(
        *[
            simpson_weigths(n_point, (bounds[1][i] - bounds[0][i]) / n_point)
            for i in range(dims)
        ]
    )
    weights_grid = np.prod(weights_grid, axis=0).reshape(-1, 1)
    return x_grid, weights_grid


def eig_grid(krg, x_grid, weights_grid):
    """
    This function computes eigen values and eigen vectors of Karhunen-Loève decomposition
    with Nyström method on a given interpolation.

    Parameters
    ----------
    krg : SMT KRG
        SMT kriging model already trained
    x_grid : array_like
        Array with shape (n_point, n_features) composed of point(s) on which the
        integral will be computed
    weights_grid : array_like
        Array with shape (n_point, 1) composed of weigths used to computed the
        integral

    Returns
    -------
    eig_val : array_like
        Eigen values
    eig_vec : array_like
        Eigen Vectors
    M : int
        Number of retained eigen values

    """
    C = covariance_matrix(krg, x_grid, conditioned=False)

    W_sqrt = np.sqrt(np.diagflat(weights_grid))
    B = W_sqrt.dot(C).dot(W_sqrt)
    eig_val, eig_vec = np.linalg.eigh(B)

    ind = (-eig_val).argsort()
    eig_val = eig_val[ind]
    eig_vec = eig_vec[:, ind]

    crit = 1.0 - np.cumsum(eig_val) / eig_val.sum()
    M = int(np.argwhere(crit > 10 ** (-8))[-1].item()) + 1

    return eig_val, eig_vec, M


def evaluate_eigen_function(krg, X, eig_val, eig_vec, x_grid, weights_grid, M):
    """
    This function evaluates eigen functions of  Karhunen-Loève decomposition
    with Nyström method on X.

    Parameters
    ----------
    krg : SMT KRG
        SMT kriging model already trained
    X : array_like
        Array with shape (n_eval, n_feature) containing point(s) on which the eigen
        function will be evaluated
    eig_val : array_like
        Eigen values
    eig_vec : array_like
        Eigen Vectors
    x_grid : array_like
        Array with shape (n_point, n_features) composed of point(s) on which the
        integral will be computed
    weights_grid : array_like
        Array with shape (n_point, 1) composed of weigths used to computed the
        integral
    M : int
        Number of retained eigen values

    Returns
    -------
    phi : array_like
        Value of retained eigen functions on X

    """
    W_sqrt = np.sqrt(np.diagflat(weights_grid))
    X_ = np.concatenate((krg.X_train, X), axis=0)
    n_X = X_.shape[0]
    n_grid = x_grid.shape[0]

    X_cont = (X_ - krg.X_offset) / krg.X_scale
    X_grid_cont = (x_grid - krg.X_offset) / krg.X_scale
    cross_d = (
        np.tile(X_cont, (n_grid, 1)) - X_grid_cont.repeat(repeats=n_X, axis=0)
    ) ** 2

    C = krg.optimal_par["sigma2"] * krg._correlation_types[krg.options["corr"]](
        krg.optimal_theta, cross_d
    ).reshape(n_grid, n_X)

    U = np.diagflat(np.sqrt(1 / eig_val[:M]))
    phi = U.dot(eig_vec[:, :M].T.dot(W_sqrt).dot(C))
    return phi


def sample_eigen(krg, X, eig_val, eig_vec, x_grid, weights_grid, M, n_traj):
    """
    This function samples trajectories of gaussian process with Karhunen-Loève decomposition
    with Nyström method.

    Parameters
    ----------
    krg : SMT KRG
        SMT kriging model already trained
    X : array_like
        Array with shape (n_eval, n_feature) containing point(s) on which the eigen
        functions will be evaluated and on which the trajectories will be sampled
    eig_val : array_like
        Eigen values
    eig_vec : array_like
        Eigen Vectors
    x_grid : array_like
        Array with shape (n_point, n_features) composed of point(s) on which the
        integral will be computed
    weights_grid : array_like
        Array with shape (n_point, 1) composed of weigths used to computed the
        integral
    M : int
        Number of retained eigen values
    n_traj : int
        Number trajectories

    Returns
    -------
    traj : array_like
        Trajectories sampled on X

    """
    phi = evaluate_eigen_function(krg, X, eig_val, eig_vec, x_grid, weights_grid, M)
    sample = np.random.randn(n_traj, M).dot(phi)
    Y = sample[:, : krg.X_train.shape[0]]
    Y_mean, Y_std = Y.mean(axis=1).reshape(-1, 1), Y.std(axis=1).reshape(-1, 1)
    Y_norma = ((Y - Y_mean) / Y_std).T

    C, Q, G, Ft = (
        krg.optimal_par["C"],
        krg.optimal_par["Q"],
        krg.optimal_par["G"],
        krg.optimal_par["Ft"],
    )
    Yt = linalg.solve_triangular(C, Y_norma, lower=True)
    beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))

    rho = Yt - np.dot(Ft, beta)
    gamma = linalg.solve_triangular(C.T, rho)

    X_cont = (X - krg.X_offset) / krg.X_scale
    dx = differences(X_cont, Y=krg.X_norma.copy())
    d = krg._componentwise_distance(dx)
    r = krg._correlation_types[krg.options["corr"]](krg.optimal_theta, d).reshape(
        X.shape[0], krg.nt
    )
    y = np.zeros(X.shape[0])
    f = krg._regression_types[krg.options["poly"]](X_cont)
    y_ = np.dot(f, beta) + np.dot(r, gamma)
    y = Y_mean.flatten() + y_ * Y_std.flatten()

    mean = krg._predict_values(X).reshape(-1, 1) - y
    traj = mean + sample[:, krg.X_train.shape[0] :].T

    return traj
