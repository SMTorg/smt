"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

import numpy as np
from scipy.sparse import csc_matrix

from smt.surrogate_models.rbf_idw_backend import rbf_jac
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils import persistence
from smt.utils.caching import cached_operation
from smt.utils.linear_solvers import get_solver


class RBF(SurrogateModel):
    """
    Radial basis function interpolant with global polynomial trend.
    """

    name = "RBF"

    def _initialize(self):
        super(RBF, self)._initialize()
        declare = self.options.declare
        supports = self.supports

        declare(
            "d0",
            1.0,
            types=(int, float, np.number, list, tuple, np.ndarray),
            desc="basis function scaling parameter in exp(-d^2 / d0^2)",
        )
        declare(
            "poly_degree",
            -1,
            types=int,
            values=(-1, 0, 1),
            desc="-1 means no global polynomial, 0 means constant, 1 means linear trend",
        )
        declare(
            "data_dir",
            values=None,
            types=str,
            desc="Directory for loading / saving cached data; None means do not save or load",
        )
        declare("reg", 1e-10, types=(int, float), desc="Regularization coeff.")
        declare(
            "max_print_depth",
            5,
            types=int,
            desc="Maximum depth (level of nesting) to print operation descriptions and times",
        )

        supports["derivatives"] = True
        supports["output_derivatives"] = True

    def _setup(self):
        options = self.options

        nx = self.training_points[None][0][0].shape[1]
        d0 = np.asarray(options["d0"], dtype=float)
        if d0.ndim == 0 or d0.size == 1:
            d0 = np.full(nx, float(d0.reshape(-1)[0]), dtype=float)
        elif d0.shape != (nx,):
            raise ValueError(
                f"Expected scalar or length-{nx} d0, got shape {d0.shape}."
            )
        if np.any(d0 <= 0.0) or not np.all(np.isfinite(d0)):
            raise ValueError(f"Expected strictly positive finite d0, got {d0!r}.")
        options["d0"] = d0

        self.printer.max_print_depth = options["max_print_depth"]

        num = {}
        # number of inputs and outputs
        num["x"] = self.training_points[None][0][0].shape[1]
        num["y"] = self.training_points[None][0][1].shape[1]
        # number of radial function terms
        num["radial"] = self.training_points[None][0][0].shape[0]
        # number of polynomial terms
        if options["poly_degree"] == -1:
            num["poly"] = 0
        elif options["poly_degree"] == 0:
            num["poly"] = 1
        elif options["poly_degree"] == 1:
            num["poly"] = 1 + num["x"]
        num["dof"] = num["radial"] + num["poly"]

        self.num = num

    def __setstate__(self, state):
        # models pickled before the C++ extension was removed carry an
        # ``rbfc`` entry (always None); drop it
        state.pop("rbfc", None)
        self.__dict__.update(state)

        self._setup()

    def _jac(self, x, kx=None):
        xt = self.training_points[None][0][0]
        return rbf_jac(x, xt, self.options["d0"], self.options["poly_degree"], kx)

    def _new_train(self):
        num = self.num

        xt, yt = self.training_points[None][0]
        jac = self._jac(xt)

        mtx = np.zeros((num["dof"], num["dof"]))
        mtx[: num["radial"], :] = jac
        mtx[:, : num["radial"]] = jac.T
        mtx[np.arange(num["radial"]), np.arange(num["radial"])] += self.options["reg"]

        self.mtx = mtx

        rhs = np.zeros((num["dof"], num["y"]))
        rhs[: num["radial"], :] = yt

        sol = np.zeros((num["dof"], num["y"]))

        solver = get_solver("dense-lu")
        with self.printer._timed_context("Initializing linear solver"):
            solver._setup(mtx, self.printer)

        for ind_y in range(rhs.shape[1]):
            with self.printer._timed_context("Solving linear system (col. %i)" % ind_y):
                solver._solve(rhs[:, ind_y], sol[:, ind_y], ind_y=ind_y)

        self.sol = sol

    def _train(self):
        """
        Train the model
        """
        self._setup()

        inputs = {"self": self}
        with cached_operation(inputs, self.options["data_dir"]) as outputs:
            if outputs:
                self.sol = outputs["sol"]
            else:
                self._new_train()
                outputs["sol"] = self.sol

    def _predict_values(self, x):
        """
        Evaluates the model at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        y : np.ndarray
            Evaluation point output variable values
        """
        jac = self._jac(x)

        y = jac.dot(self.sol)
        return y

    def _predict_derivatives(self, x, kx):
        """
        Evaluates the derivatives at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values
        kx : int
            The 0-based index of the input variable with respect to which derivatives are desired.

        Returns
        -------
        dy_dx : np.ndarray
            Derivative values.
        """
        jac = self._jac(x, kx)

        dy_dx = jac.dot(self.sol)
        return dy_dx

    def _predict_output_derivatives(self, x):
        _nt = self.nt
        ny = self.training_points[None][0][1].shape[1]
        num = self.num

        dy_dstates = self._jac(x)

        dstates_dytl = np.linalg.inv(self.mtx)

        ones = np.ones(self.nt)
        arange = np.arange(self.nt)
        dytl_dyt = csc_matrix((ones, (arange, arange)), shape=(num["dof"], self.nt))

        dy_dyt = (dytl_dyt.T.dot(dstates_dytl.T).dot(dy_dstates.T)).T
        dy_dyt = np.einsum("ij,k->ijk", dy_dyt, np.ones(ny))
        return {None: dy_dyt}

    def save(self, filename):
        persistence.save(self, filename)

    @staticmethod
    def load(filename):
        return persistence.load(filename)
