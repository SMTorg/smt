"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

from numbers import Integral

import numpy as np

from smt.surrogate_models.rmts import RMTS
from smt.surrogate_models.rmts_backend import PureRMTB


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

        self._backend = PureRMTB()
        self._backend.setup(
            num["x"],
            np.array(self.options["xlimits"][:, 0], dtype=np.float64),
            np.array(self.options["xlimits"][:, 1], dtype=np.float64),
            np.array(num["order_list"], np.int32),
            np.array(num["ctrl_list"], np.int32),
        )
        self.knots_list = [
            np.array(k, dtype=np.float64, copy=True) for k in self._backend.knots_list
        ]

    def __getstate__(self):
        state = super().__getstate__()
        state["_backend"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        if None in self.training_points and 0 in self.training_points[None]:
            self._setup()

    def _compute_jac_raw(self, ix1, ix2, x):
        xlimits = self.options["xlimits"]

        t = np.zeros(x.shape, dtype=np.float64)
        for kx in range(self.num["x"]):
            t[:, kx] = (x[:, kx] - xlimits[kx, 0]) / (xlimits[kx, 1] - xlimits[kx, 0])
        t = np.maximum(t, 0.0 + 1e-15)
        t = np.minimum(t, 1.0 - 1e-15)

        data, rows, cols = self._backend.compute_jac(ix1 - 1, ix2 - 1, t)

        if ix1 != 0:
            data /= xlimits[ix1 - 1, 1] - xlimits[ix1 - 1, 0]
        if ix2 != 0:
            data /= xlimits[ix2 - 1, 1] - xlimits[ix2 - 1, 0]

        return data, rows, cols

    def _compute_dof2coeff(self):
        return None

    def get_bspline(self, output=0, physical=True):
        """
        Return the trained tensor-product B-spline representation for one output.

        Parameters
        ----------
        output : int, optional
            0-based output column index (default 0).
        physical : bool, optional
            If True (default), rescale knot vectors to physical ``xlimits``;
            if False, return normalized knot vectors on ``[0, 1]``.

        Returns
        -------
        dict
            Dictionary with keys ``'knots'``, ``'degree'``, ``'coefficients'``,
            and ``'domain'``.
        """
        if not hasattr(self, "sol_coeff") or self.sol_coeff is None:
            raise RuntimeError(
                "RMTB model has not been trained yet. Call train() first."
            )
        ny = self.num["y"]
        if (
            isinstance(output, bool)
            or not isinstance(output, Integral)
            or not (0 <= int(output) < ny)
        ):
            raise ValueError(f"output must be an integer in [0, {ny}), got {output!r}.")

        xlimits = np.asarray(self.options["xlimits"], dtype=np.float64)
        if physical:
            knots = tuple(
                np.array(
                    xlimits[kx, 0] + k * (xlimits[kx, 1] - xlimits[kx, 0]),
                    dtype=np.float64,
                )
                for kx, k in enumerate(self.knots_list)
            )
            domain = xlimits.copy()
        else:
            knots = tuple(
                np.array(k, dtype=np.float64, copy=True) for k in self.knots_list
            )
            domain = np.tile(np.array([0.0, 1.0], dtype=np.float64), (self.num["x"], 1))

        degree = tuple(int(o) - 1 for o in self.num["order_list"])
        ctrl_shape = tuple(int(c) for c in self.num["ctrl_list"])
        coefficients = np.array(
            self.sol_coeff[:, int(output)].reshape(ctrl_shape),
            dtype=np.float64,
            copy=True,
        )
        return {
            "knots": knots,
            "degree": degree,
            "coefficients": coefficients,
            "domain": domain,
        }

    def to_scipy(self, output=0, physical=True, as_1d=False):
        """
        Convert the trained RMTB model for one output to a SciPy spline object.

        Parameters
        ----------
        output : int, optional
            0-based output column index (default 0).
        physical : bool, optional
            If True (default), construct the spline over physical ``xlimits``;
            if False, construct it over normalized ``[0, 1]^nx``.
        as_1d : bool, optional
            If True and ``nx == 1``, return a ``scipy.interpolate.BSpline`` instance
            (which accepts 1D array evaluations ``(n,)``) instead of ``NdBSpline``.

        Returns
        -------
        scipy.interpolate.NdBSpline or scipy.interpolate.BSpline
            Equivalent SciPy B-spline object.
        """
        bs = self.get_bspline(output=output, physical=physical)
        if as_1d:
            if self.num["x"] != 1:
                raise ValueError("as_1d=True is only supported when nx == 1.")
            from scipy.interpolate import BSpline

            return BSpline(bs["knots"][0], bs["coefficients"], bs["degree"][0])

        try:
            from scipy.interpolate import NdBSpline
        except ImportError as err:
            raise ImportError(
                "RMTB.to_scipy() requires SciPy >= 1.12.0 for scipy.interpolate.NdBSpline."
            ) from err

        return NdBSpline(bs["knots"], bs["coefficients"], bs["degree"])
