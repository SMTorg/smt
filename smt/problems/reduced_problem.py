"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

Reduced problem class - selects a subset of input variables.
"""
import numpy as np

from smt.utils.options_dictionary import OptionsDictionary
from smt.problems.problem import Problem


class ReducedProblem(Problem):
    def __init__(self, problem, dims, w=0.2):
        """
        Arguments
        ---------
        problem : Problem
            Pointer to the Problem object being wrapped.
        dims : int or list/tuple of ints
            Either the number of dimensions or a list of the dimension indices that this
            problem uses.
        w : float
            The value to use for all unaccounted for inputs where 0/1 is lower/upper bound.
        """
        self.problem = problem
        self.w = w

        if isinstance(dims, int):
            self.dims = np.arange(dims)
            assert dims <= problem.options["ndim"]
        elif isinstance(dims, (list, tuple, np.ndarray)):
            self.dims = np.array(dims, int)
            assert np.max(dims) < problem.options["ndim"]
        else:
            raise ValueError("dims is invalid")

        self.options = OptionsDictionary()
        self.options.declare("ndim", len(self.dims), types=int)
        self.options.declare("return_complex", False, types=bool)
        self.options.declare("name", "R_" + self.problem.options["name"], types=str)

        self.xlimits = np.zeros((self.options["ndim"], 2))
        for idim, idim_reduced in enumerate(self.dims):
            self.xlimits[idim, :] = problem.xlimits[idim_reduced, :]

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape

        nx_prob = self.problem.options["ndim"]
        x_prob = np.zeros((ne, nx_prob), complex)
        for ix in range(nx_prob):
            x_prob[:, ix] = (1 - self.w) * self.problem.xlimits[
                ix, 0
            ] + self.w * self.problem.xlimits[ix, 1]

        for ix in range(nx):
            x_prob[:, self.dims[ix]] = x[:, ix]

        if kx is None:
            y = self.problem._evaluate(x_prob, None)
        else:
            y = self.problem._evaluate(x_prob, self.dims[kx])

        return y
