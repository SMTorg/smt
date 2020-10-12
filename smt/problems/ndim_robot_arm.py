"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

N-dimensional robot arm problem.
"""
import numpy as np

from smt.utils.options_dictionary import OptionsDictionary
from smt.problems.problem import Problem
from smt.problems.reduced_problem import ReducedProblem
from smt.problems.robot_arm import RobotArm


class NdimRobotArm(Problem):
    def __init__(self, ndim=1, w=0.2):
        self.problem = ReducedProblem(
            RobotArm(ndim=2 * (ndim + 1)), np.arange(3, 2 * (ndim + 1), 2), w=w
        )

        self.options = OptionsDictionary()
        self.options.declare("ndim", ndim, types=int)
        self.options.declare("return_complex", False, types=bool)
        self.options.declare("name", "NdimRobotArm", types=str)

        self.xlimits = self.problem.xlimits

    def _evaluate(self, x, kx):
        return self.problem._evaluate(x, kx)
