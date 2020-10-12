"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

N-dimensional cantilever beam problem.
"""
import numpy as np

from smt.utils.options_dictionary import OptionsDictionary
from smt.problems.problem import Problem
from smt.problems.reduced_problem import ReducedProblem
from smt.problems.cantilever_beam import CantileverBeam


class NdimCantileverBeam(Problem):
    def __init__(self, ndim=1, w=0.2):
        self.problem = ReducedProblem(
            CantileverBeam(ndim=3 * ndim), np.arange(1, 3 * ndim, 3), w=w
        )

        self.options = OptionsDictionary()
        self.options.declare("ndim", ndim, types=int)
        self.options.declare("return_complex", False, types=bool)
        self.options.declare("name", "NdimCantileverBeam", types=str)

        self.xlimits = self.problem.xlimits

    def _evaluate(self, x, kx):
        return self.problem._evaluate(x, kx)
