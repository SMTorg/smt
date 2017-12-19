"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

N-dimensional Rosenbrock problem.
"""
import numpy as np
from six.moves import range

from smt.utils.options_dictionary import OptionsDictionary
from smt.problem.problem import Problem
from smt.problem.reduced_problem import ReducedProblem
from smt.problem.rosenbrock import Rosenbrock


class NdimRosenbrock(Problem):

    def __init__(self, ndim=1, w=0.2):
        self.problem = ReducedProblem(Rosenbrock(ndim=ndim+1), np.arange(1, ndim+1), w=w)

        self.options = OptionsDictionary()
        self.options.declare('ndim', ndim, types=int)
        self.options.declare('return_complex', False, types=bool)
        self.options.declare('name', 'NdimRosenbrock', types=str)

        self.xlimits = self.problem.xlimits

    def _evaluate(self, x, kx):
        return self.problem._evaluate(x, kx)
