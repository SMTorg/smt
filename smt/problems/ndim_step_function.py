"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

N-dimensional step function problem.
"""
import numpy as np

from smt.utils.options_dictionary import OptionsDictionary
from smt.problems.problem import Problem
from smt.problems.tensor_product import TensorProduct


class NdimStepFunction(Problem):
    def __init__(self, ndim=1, width=10.0):
        self.problem = TensorProduct(ndim=ndim, func="tanh", width=width)

        self.options = OptionsDictionary()
        self.options.declare("ndim", ndim, types=int)
        self.options.declare("return_complex", False, types=bool)
        self.options.declare("name", "NdimStepFunction", types=str)

        self.xlimits = self.problem.xlimits

    def _evaluate(self, x, kx):
        return self.problem._evaluate(x, kx)
