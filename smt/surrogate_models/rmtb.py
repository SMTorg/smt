"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""
import numpy as np
import scipy.sparse
from numbers import Integral

from smt.utils.linear_solvers import get_solver
from smt.utils.line_search import get_line_search_class
from smt.surrogate_models.rmts import RMTS

from smt.surrogate_models.rmtsclib import PyRMTB


class RMTB(RMTS):
    """
    Regularized Minimal-energy Tensor-product B-spline (RMTB) interpolant.

    RMTB builds an approximation from a tensor product of B-spline curves.
    In 1-D it is a B-spline curve, in 2-D it is a B-spline surface, in 3-D
    it is a B-spline volume, and so on - it works for any arbitrary number
    of dimensions. However, the training points should preferably be
    arranged in a structured fashion.

    Advantages:
    - Evaluation time is independent of the number of training points
    - The smoothness can be tuned by adjusting the B-spline order and the
    number of B-spline control points (the latter also affects performance)

    Disadvantages:
    - Training time scales poorly with the # dimensions
    - The data should be structured - RMTB does not handle track data well
    - RMTB approximates, not interpolates - it does not pass through the
    training points
    """

    name = "RMTB"

    def _initialize(self):
        super(RMTB, self)._initialize()
        declare = self.options.declare

        declare(
            "order",
            3,
            types=(Integral, tuple, list, np.ndarray),
            desc="B-spline order in each dimension - length [nx]",
        )
        declare(
            "num_ctrl_pts",
            15,
            types=(Integral, tuple, list, np.ndarray),
            desc="# B-spline control points in each dimension - length [nx]",
        )

    def _setup(self):
        options = self.options
        nx = self.training_points[None][0][0].shape[1]

        for name in ["smoothness", "num_ctrl_pts", "order"]:
            if isinstance(options[name], (int, float)):
                options[name] = [options[name]] * nx
            options[name] = np.atleast_1d(options[name])

        self.printer.max_print_depth = options["max_print_depth"]

        num = {}
        # number of inputs and outputs
        num["x"] = self.training_points[None][0][0].shape[1]
        num["y"] = self.training_points[None][0][1].shape[1]
        num["order_list"] = np.array(options["order"], int)
        num["order"] = np.prod(num["order_list"])
        num["ctrl_list"] = np.array(options["num_ctrl_pts"], int)
        num["ctrl"] = np.prod(num["ctrl_list"])
        num["elem_list"] = np.array(num["ctrl_list"] - num["order_list"] + 1, int)
        num["elem"] = np.prod(num["elem_list"])
        num["knots_list"] = num["order_list"] + num["ctrl_list"]
        num["knots"] = np.sum(num["knots_list"])
        # total number of training points (function values and derivatives)
        num["t"] = 0
        for kx in self.training_points[None]:
            num["t"] += self.training_points[None][kx][0].shape[0]
        # for RMT
        num["coeff"] = num["ctrl"]
        num["support"] = num["order"]
        num["dof"] = num["ctrl"]

        self.num = num

        self.rmtsc = PyRMTB()
        self.rmtsc.setup(
            num["x"],
            np.array(self.options["xlimits"][:, 0]),
            np.array(self.options["xlimits"][:, 1]),
            np.array(num["order_list"], np.int32),
            np.array(num["ctrl_list"], np.int32),
        )

    def _compute_jac_raw(self, ix1, ix2, x):
        xlimits = self.options["xlimits"]

        t = np.zeros(x.shape)
        for kx in range(self.num["x"]):
            t[:, kx] = (x[:, kx] - xlimits[kx, 0]) / (xlimits[kx, 1] - xlimits[kx, 0])
        t = np.maximum(t, 0.0 + 1e-15)
        t = np.minimum(t, 1.0 - 1e-15)

        n = x.shape[0]
        nnz = n * self.num["order"]
        # data, rows, cols = RMTBlib.compute_jac(ix1, ix2, self.num['x'], n, nnz,
        #     self.num['order_list'], self.num['ctrl_list'], t)

        data = np.empty(nnz)
        rows = np.empty(nnz, dtype=np.int32)
        cols = np.empty(nnz, dtype=np.int32)
        self.rmtsc.compute_jac(ix1 - 1, ix2 - 1, n, t.flatten(), data, rows, cols)

        if ix1 != 0:
            data /= xlimits[ix1 - 1, 1] - xlimits[ix1 - 1, 0]
        if ix2 != 0:
            data /= xlimits[ix2 - 1, 1] - xlimits[ix2 - 1, 0]

        return data, rows, cols

    def _compute_dof2coeff(self):
        return None
