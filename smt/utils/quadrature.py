"""
Numerical quadrature grids and weights.

Pure mathematical utilities with no kriging dependency.

Functions
---------
gauss_legendre_grid
    Gauss-Legendre quadrature grid and weights.
rectangular_grid
    Rectangular (midpoint) quadrature grid and weights.
simpson_weigths
    Simpson quadrature weights in one dimension.
simpson_grid
    Simpson quadrature grid and weights.

Authors: Morgane Menz / Alexandre Thouvenot
"""

import numpy as np


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
    n_points : int
        Number of weigths
    h : float
        Scaling coefficient

    Returns
    -------
    weights : array_like
        Array with shape(n_points,) composed of weigths used to computed the
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
