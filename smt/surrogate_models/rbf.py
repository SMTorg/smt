"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""
import numpy as np
from scipy.sparse import csc_matrix
from smt.surrogate_models.surrogate_model import SurrogateModel

from smt.utils.linear_solvers import get_solver
from smt.utils.caching import cached_operation

from smt.surrogate_models.rbfclib import PyRBF


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
            types=(int, float, list, np.ndarray),
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
        if isinstance(options["d0"], (int, float)):
            options["d0"] = [options["d0"]] * nx
        options["d0"] = np.array(np.atleast_1d(options["d0"]), dtype=float)

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

        nt = self.training_points[None][0][0].shape[0]
        xt, yt = self.training_points[None][0]

        self.rbfc = PyRBF()
        self.rbfc.setup(
            num["x"],
            nt,
            num["dof"],
            options["poly_degree"],
            options["d0"],
            xt.flatten(),
        )

    def _new_train(self):
        num = self.num

        xt, yt = self.training_points[None][0]
        jac = np.empty(num["radial"] * num["dof"])
        self.rbfc.compute_jac(num["radial"], xt.flatten(), jac)
        jac = jac.reshape((num["radial"], num["dof"]))

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

        tmp = self.rbfc
        self.rbfc = None

        inputs = {"self": self}
        with cached_operation(inputs, self.options["data_dir"]) as outputs:
            self.rbfc = tmp

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
        n = x.shape[0]
        num = self.num

        jac = np.empty(n * num["dof"])
        self.rbfc.compute_jac(n, x.flatten(), jac)
        jac = jac.reshape((n, num["dof"]))

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
        n = x.shape[0]
        num = self.num

        jac = np.empty(n * num["dof"])
        self.rbfc.compute_jac_derivs(n, kx, x.flatten(), jac)
        jac = jac.reshape((n, num["dof"]))

        dy_dx = jac.dot(self.sol)
        return dy_dx

    def _predict_output_derivatives(self, x):
        n = x.shape[0]
        nt = self.nt
        ny = self.training_points[None][0][1].shape[1]
        num = self.num

        dy_dstates = np.empty(n * num["dof"])
        self.rbfc.compute_jac(n, x.flatten(), dy_dstates)
        dy_dstates = dy_dstates.reshape((n, num["dof"]))

        dstates_dytl = np.linalg.inv(self.mtx)

        ones = np.ones(self.nt)
        arange = np.arange(self.nt)
        dytl_dyt = csc_matrix((ones, (arange, arange)), shape=(num["dof"], self.nt))

        dy_dyt = (dytl_dyt.T.dot(dstates_dytl.T).dot(dy_dstates.T)).T
        dy_dyt = np.einsum("ij,k->ijk", dy_dyt, np.ones(ny))
        return {None: dy_dyt}
