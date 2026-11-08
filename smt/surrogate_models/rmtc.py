"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

from numbers import Integral

import numpy as np
import scipy.sparse

from smt.surrogate_models.rmts import RMTS
from smt.surrogate_models.rmts_backend import PureRMTC


class RMTC(RMTS):
    """
    Regularized Minimal-energy Tensor-product Cubic hermite spline (RMTC) interpolant.

    RMTC divides the n-dimensional space using n-dimensional box elements.
    Each n-D box is represented using a tensor-product of cubic functions,
    one in each dimension. The coefficients of the cubic functions are
    computed by minimizing the second derivatives of the interpolant under
    the condition that it interpolates or approximates the training points.

    Advantages:
    - Extremely fast to evaluate
    - Evaluation/training time are relatively insensitive to the number of
    training points
    - Avoids oscillations

    Disadvantages:
    - Training time scales poorly with the # dimensions (too slow beyond 4-D)
    - The user must choose the number of elements in each dimension
    """

    name = "RMTC"

    def _initialize(self):
        super(RMTC, self)._initialize()
        declare = self.options.declare

        declare(
            "num_elements",
            4,
            types=(Integral, list, np.ndarray),
            desc="# elements in each dimension - ndarray [nx]",
        )

    def _setup(self):
        options = self.options
        nx = self.training_points[None][0][0].shape[1]

        for name in ["smoothness", "num_elements"]:
            if isinstance(options[name], (int, float)):
                options[name] = [options[name]] * nx
            options[name] = np.atleast_1d(options[name])

        self.printer.max_print_depth = options["max_print_depth"]

        num = {}
        # number of inputs and outputs
        num["x"] = self.training_points[None][0][0].shape[1]
        num["y"] = self.training_points[None][0][1].shape[1]
        # number of elements
        num["elem_list"] = np.array(options["num_elements"], int)
        num["elem"] = np.prod(num["elem_list"])
        # number of terms/coefficients per element
        num["term_list"] = 4 * np.ones(num["x"], int)
        num["term"] = np.prod(num["term_list"])
        # number of nodes
        num["uniq_list"] = num["elem_list"] + 1
        num["uniq"] = np.prod(num["uniq_list"])
        # total number of training points (function values and derivatives)
        num["t"] = 0
        for kx in self.training_points[None]:
            num["t"] += self.training_points[None][kx][0].shape[0]
        # for RMT
        num["coeff"] = num["term"] * num["elem"]
        num["support"] = num["term"]
        num["dof"] = num["uniq"] * 2 ** num["x"]

        self.num = num

        self._backend = PureRMTC()
        self._backend.setup(
            num["x"],
            np.array(self.options["xlimits"][:, 0], dtype=np.float64),
            np.array(self.options["xlimits"][:, 1], dtype=np.float64),
            np.array(num["elem_list"], np.int32),
            np.array(num["term_list"], np.int32),
        )

    def __getstate__(self):
        state = super().__getstate__()
        state["_backend"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        if None in self.training_points and 0 in self.training_points[None]:
            self._setup()

    def _compute_jac_raw(self, ix1, ix2, x):
        return self._backend.compute_jac(ix1 - 1, ix2 - 1, x)

    def _compute_dof2coeff(self):
        num = self.num

        # This computes a num['term'] x num['term'] matrix called coeff2nodal.
        # Multiplying this matrix with the list of coefficients for an element
        # yields the list of function and derivative values at the element nodes.
        # We need the inverse, but the matrix size is small enough to invert since
        # RMTC is normally only used for 1 <= nx <= 4 in most cases.
        elem_coeff2nodal = self._backend.compute_coeff2nodal()
        elem_nodal2coeff = np.linalg.inv(elem_coeff2nodal)

        # This computes a num_coeff_elem x num_coeff_uniq permutation matrix called
        # uniq2elem. This sparse matrix maps the unique list of nodal function and
        # derivative values to the same function and derivative values, but ordered
        # by element, with repetition.
        data, rows, cols = self._backend.compute_uniq2elem()

        num_coeff_elem = num["term"] * num["elem"]
        num_coeff_uniq = num["uniq"] * 2 ** num["x"]
        full_uniq2elem = scipy.sparse.csc_matrix(
            (data, (rows, cols)), shape=(num_coeff_elem, num_coeff_uniq)
        )

        # This computes the matrix full_dof2coeff, which maps the unique
        # degrees of freedom to the list of coefficients ordered by element.
        full_nodal2coeff = scipy.sparse.kron(
            scipy.sparse.eye(num["elem"], format="csc"),
            scipy.sparse.csc_matrix(elem_nodal2coeff),
            format="csc",
        )

        full_dof2coeff = (full_nodal2coeff * full_uniq2elem).tocsc()

        return full_dof2coeff
