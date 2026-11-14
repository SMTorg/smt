"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>, Paul Saves <paul.saves@ird.fr>

This package is distributed under New BSD license.
"""

from numbers import Integral

import numpy as np
import scipy.sparse

from smt.surrogate_models.rmts_backend import (
    PureRMTB,
    PureRMTC,
    _opt_func_jax,
    _opt_grad_jax,
    _opt_hess_jax,
    _prepare_jax_opt_cache,
    _use_jax_opt,
    compute_ext_dist,
    compute_quadrature_points,
)
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils import persistence
from smt.utils.caching import cached_operation
from smt.utils.line_search import VALID_LINE_SEARCHES, LineSearch, get_line_search_class
from smt.utils.linear_solvers import VALID_SOLVERS, LinearSolver, get_solver


def _normalize_spline_type(spline_type):
    st = str(spline_type).strip().lower().replace("-", "_").replace(" ", "_")
    if st in ("b_spline", "b_spine", "bspline", "b"):
        return "B-Spline"
    if st in ("cubic_hermite_spline", "cubic_hermite", "hermite", "c"):
        return "Cubic_Hermite_Spline"
    raise ValueError(
        f"Unsupported spline_type: {spline_type!r}. Expected 'B-Spline' or 'Cubic_Hermite_Spline'."
    )


class RMTS(SurrogateModel):
    """
    Regularized Minimal-energy Tensor-product Spline interpolant base class for RMTC and RMTB.
    """

    name = "RMTS"

    def _initialize(self):
        super(RMTS, self)._initialize()
        declare = self.options.declare
        supports = self.supports

        declare(
            "spline_type",
            "B-Spline",
            values=[
                "B-Spline",
                "Cubic_Hermite_Spline",
                "bspline",
                "cubic_hermite",
                "b_spline",
                "cubic_hermite_spline",
                "Cubic_Hermite",
            ],
            types=str,
            desc="Spline representation type: 'B-Spline' or 'Cubic_Hermite_Spline'",
        )
        declare(
            "order",
            3,
            types=(Integral, tuple, list, np.ndarray),
            desc="B-spline order in each dimension - length [nx] (for B-Splines)",
        )
        declare(
            "num_ctrl_pts",
            15,
            types=(Integral, tuple, list, np.ndarray),
            desc="# B-spline control points in each dimension - length [nx] (for B-Splines)",
        )
        declare(
            "num_elements",
            4,
            types=(Integral, tuple, list, np.ndarray),
            desc="# elements in each dimension - length [nx] (for Cubic_Hermite_Spline)",
        )
        declare(
            "xlimits",
            types=np.ndarray,
            desc="Lower/upper bounds in each dimension - ndarray [nx, 2]",
        )
        declare(
            "smoothness",
            1.0,
            types=(Integral, float, tuple, list, np.ndarray),
            desc="Smoothness parameter in each dimension - length nx. None implies uniform",
        )
        declare(
            "regularization_weight",
            1e-14,
            types=(Integral, float),
            desc="Weight of the term penalizing the norm of the spline coefficients."
            + " This is useful as an alternative to energy minimization "
            + " when energy minimization makes the training time too long.",
        )
        declare(
            "energy_weight",
            1e-4,
            types=(Integral, float),
            desc="The weight of the energy minimization terms",
        )
        declare(
            "extrapolate",
            False,
            types=bool,
            desc="Whether to perform linear extrapolation for external evaluation points",
        )
        declare(
            "min_energy",
            True,
            types=bool,
            desc="Whether to perform energy minimization",
        )
        declare(
            "approx_order", 4, types=Integral, desc="Exponent in the approximation term"
        )
        declare(
            "solver",
            "krylov",
            values=VALID_SOLVERS,
            types=LinearSolver,
            desc="Linear solver",
        )
        declare(
            "derivative_solver",
            "krylov",
            values=VALID_SOLVERS,
            types=LinearSolver,
            desc="Linear solver used for computing output derivatives (dy_dyt)",
        )
        declare(
            "grad_weight",
            0.5,
            types=(Integral, float),
            desc="Weight on gradient training data",
        )
        declare(
            "solver_tolerance",
            1e-12,
            types=(Integral, float),
            desc="Convergence tolerance for the nonlinear solver",
        )
        declare(
            "nonlinear_maxiter",
            10,
            types=Integral,
            desc="Maximum number of nonlinear solver iterations",
        )
        declare(
            "line_search",
            "backtracking",
            values=VALID_LINE_SEARCHES,
            types=LineSearch,
            desc="Line search algorithm",
        )
        declare(
            "save_energy_terms",
            False,
            types=bool,
            desc="Whether to cache energy terms in the data_dir directory",
        )
        declare(
            "data_dir",
            None,
            values=(None,),
            types=str,
            desc="Directory for loading / saving cached data; None means do not save or load",
        )
        declare(
            "max_print_depth",
            5,
            types=Integral,
            desc="Maximum depth (level of nesting) to print operation descriptions and times",
        )

        declare(
            "jax_dtype",
            "float64",
            values=["float64", "float32"],
            types=str,
            desc="Computation precision for JAX acceleration ('float64' or 'float32')",
        )

        supports["training_derivatives"] = True
        supports["derivatives"] = True
        supports["output_derivatives"] = True
        supports["jax"] = True

    def _final_initialize(self):
        super()._final_initialize()
        _normalize_spline_type(self.options["spline_type"])

    def _check_approx_order(self):
        # Validated at training time (not at construction) so that subclasses
        # such as RMTBUQ can apply their own, stricter checks in ``_train``.
        approx_order = int(self.options["approx_order"])
        if approx_order < 2 or approx_order % 2 != 0:
            raise ValueError(
                f"approx_order must be a positive even integer (>= 2), got {approx_order}."
            )

    def _setup(self):
        self._check_approx_order()
        self._jax_opt_cache = None
        options = self.options
        nx = self.training_points[None][0][0].shape[1]
        st = _normalize_spline_type(options["spline_type"])

        if st == "B-Spline":
            for name in ["smoothness", "num_ctrl_pts", "order"]:
                if isinstance(options[name], (int, float)):
                    options[name] = [options[name]] * nx
                options[name] = np.atleast_1d(options[name])

            self.printer.max_print_depth = options["max_print_depth"]

            num = {}
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
            num["t"] = 0
            for kx in self.training_points[None]:
                num["t"] += self.training_points[None][kx][0].shape[0]
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
                np.array(k, dtype=np.float64, copy=True)
                for k in self._backend.knots_list
            ]
        else:
            for name in ["smoothness", "num_elements"]:
                if isinstance(options[name], (int, float)):
                    options[name] = [options[name]] * nx
                options[name] = np.atleast_1d(options[name])

            self.printer.max_print_depth = options["max_print_depth"]

            num = {}
            num["x"] = self.training_points[None][0][0].shape[1]
            num["y"] = self.training_points[None][0][1].shape[1]
            num["elem_list"] = np.array(options["num_elements"], int)
            num["elem"] = np.prod(num["elem_list"])
            num["term_list"] = 4 * np.ones(num["x"], int)
            num["term"] = np.prod(num["term_list"])
            num["uniq_list"] = num["elem_list"] + 1
            num["uniq"] = np.prod(num["uniq_list"])
            num["t"] = 0
            for kx in self.training_points[None]:
                num["t"] += self.training_points[None][kx][0].shape[0]
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

    def _compute_jac_raw(self, ix1, ix2, x):
        st = _normalize_spline_type(self.options["spline_type"])
        if st == "B-Spline":
            xlimits = self.options["xlimits"]
            t = np.zeros(x.shape, dtype=np.float64)
            for kx in range(self.num["x"]):
                t[:, kx] = (x[:, kx] - xlimits[kx, 0]) / (
                    xlimits[kx, 1] - xlimits[kx, 0]
                )
            t = np.maximum(t, 0.0 + 1e-15)
            t = np.minimum(t, 1.0 - 1e-15)

            data, rows, cols = self._backend.compute_jac(ix1 - 1, ix2 - 1, t)

            if ix1 != 0:
                data /= xlimits[ix1 - 1, 1] - xlimits[ix1 - 1, 0]
            if ix2 != 0:
                data /= xlimits[ix2 - 1, 1] - xlimits[ix2 - 1, 0]

            return data, rows, cols
        else:
            return self._backend.compute_jac(ix1 - 1, ix2 - 1, x)

    def _compute_dof2coeff(self):
        st = _normalize_spline_type(self.options["spline_type"])
        if st == "B-Spline":
            return None
        else:
            num = self.num
            elem_coeff2nodal = self._backend.compute_coeff2nodal()
            elem_nodal2coeff = np.linalg.inv(elem_coeff2nodal)

            data, rows, cols = self._backend.compute_uniq2elem()
            num_coeff_elem = num["term"] * num["elem"]
            num_coeff_uniq = num["uniq"] * 2 ** num["x"]
            full_uniq2elem = scipy.sparse.csc_matrix(
                (data, (rows, cols)), shape=(num_coeff_elem, num_coeff_uniq)
            )

            full_nodal2coeff = scipy.sparse.kron(
                scipy.sparse.eye(num["elem"], format="csc"),
                scipy.sparse.csc_matrix(elem_nodal2coeff),
                format="csc",
            )
            full_dof2coeff = (full_nodal2coeff * full_uniq2elem).tocsc()
            return full_dof2coeff

    def _setup_hessian(self):
        diag = np.ones(self.num["dof"])
        arange = np.arange(self.num["dof"])
        full_hess = scipy.sparse.csc_matrix((diag, (arange, arange)))
        return full_hess

    def _compute_jac(self, ix1, ix2, x):
        data, rows, cols = self._compute_jac_raw(ix1, ix2, x)
        n = x.shape[0]
        full_jac = scipy.sparse.csc_matrix(
            (data, (rows, cols)), shape=(n, self.num["coeff"])
        )
        if self.full_dof2coeff is not None:
            full_jac = full_jac * self.full_dof2coeff
        return full_jac

    def _compute_approx_terms(self):
        # This computes the approximation terms for the training points.
        # We loop over kx: 0 is for values and kx>0 represents.
        # the 1-based index of the derivative given by the training point data.
        _num = self.num
        xlimits = self.options["xlimits"]

        full_jac_dict = {}
        for kx in self.training_points[None]:
            xt, yt = self.training_points[None][kx]

            xmin = np.min(xt, axis=0)
            xmax = np.max(xt, axis=0)
            assert np.all(xlimits[:, 0] <= xmin), (
                "Training points below min for %s" % kx
            )
            assert np.all(xlimits[:, 1] >= xmax), (
                "Training points above max for %s" % kx
            )

            if kx == 0:
                c = 1.0
            else:
                c = self.options["grad_weight"] / xlimits.shape[0]

            full_jac = self._compute_jac(kx, 0, xt)
            full_jac_dict[kx] = (full_jac, full_jac.T.tocsc(), c)

        return full_jac_dict

    def _compute_energy_terms(self):
        # This computes the energy terms that are to be minimized.
        # The quadrature points are the centroids of the multi-dimensional elements.
        num = self.num
        xlimits = self.options["xlimits"]

        inputs = {}
        inputs["nx"] = xlimits.shape[0]
        inputs["elem_list"] = num["elem_list"]
        st = _normalize_spline_type(self.options["spline_type"])
        if st == "B-Spline":
            inputs["num_ctrl_list"] = num["ctrl_list"]
            inputs["order_list"] = num["order_list"]

        if self.options["save_energy_terms"]:
            cache_dir = self.options["data_dir"]
        else:
            cache_dir = None
        with cached_operation(inputs, cache_dir) as outputs:
            if outputs:
                sq_mtx = outputs["sq_mtx"]
            else:
                x = compute_quadrature_points(
                    2 * num["elem_list"], xlimits[:, 0], xlimits[:, 1]
                )

                sq_mtx = [None] * num["x"]
                for kx in range(num["x"]):
                    mtx = self._compute_jac(kx + 1, kx + 1, x)
                    sq_mtx[kx] = (
                        mtx.T.tocsc() * mtx * (xlimits[kx, 1] - xlimits[kx, 0]) ** 4
                    )

                outputs["sq_mtx"] = sq_mtx

        elem_vol = np.prod((xlimits[:, 1] - xlimits[:, 0]) / (2 * num["elem_list"]))
        total_vol = np.prod(xlimits[:, 1] - xlimits[:, 0])

        full_hess = scipy.sparse.csc_matrix((num["dof"], num["dof"]))
        for kx in range(num["x"]):
            full_hess += sq_mtx[kx] * (
                elem_vol
                / total_vol
                * self.options["smoothness"][kx]
                / (xlimits[kx, 1] - xlimits[kx, 0]) ** 4
            )

        return full_hess

    def __getstate__(self):
        state = super().__getstate__()
        state["_backend"] = None
        state["_jax_opt_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if None in self.training_points and 0 in self.training_points[None]:
            self._setup()

    def _get_jax_opt_data(self):
        cache = getattr(self, "_jax_opt_cache", None)
        if (
            cache is None
            or cache["hess_id"] != id(self.full_hess)
            or cache["jac_dict_id"] != id(self.full_jac_dict)
            or cache["hess_nnz"] != self.full_hess.nnz
        ):
            cache = _prepare_jax_opt_cache(self.full_hess, self.full_jac_dict)
            self._jax_opt_cache = cache
        return cache

    def _opt_func(self, sol, p, yt_dict):
        if _use_jax_opt():
            cache = self._get_jax_opt_data()
            return _opt_func_jax(sol, p, yt_dict, cache)

        full_hess = self.full_hess
        full_jac_dict = self.full_jac_dict

        func = 0.5 * np.dot(sol, full_hess * sol)
        for kx in self.training_points[None]:
            full_jac, full_jac_T, c = full_jac_dict[kx]
            yt = yt_dict[kx]
            func += 0.5 * c * np.sum((full_jac * sol - yt) ** p)

        return func

    def _opt_grad(self, sol, p, yt_dict):
        if _use_jax_opt():
            cache = self._get_jax_opt_data()
            return _opt_grad_jax(sol, p, yt_dict, cache)

        full_hess = self.full_hess
        full_jac_dict = self.full_jac_dict

        grad = full_hess * sol
        for kx in self.training_points[None]:
            full_jac, full_jac_T, c = full_jac_dict[kx]
            yt = yt_dict[kx]
            grad += 0.5 * c * full_jac_T * p * (full_jac * sol - yt) ** (p - 1)

        return grad

    def _opt_dgrad_dyt(self, sol, p, yt_dict, kx):
        full_jac_dict = self.full_jac_dict

        full_jac, full_jac_T, c = full_jac_dict[kx]
        yt = yt_dict[kx]

        if p in (0, 1):
            diag_vec = np.zeros(full_jac.shape[0], dtype=np.float64)
        else:
            diag_vec = p * (p - 1) * (full_jac * sol - yt) ** (p - 2)
        diag_mtx = scipy.sparse.diags(diag_vec, format="csc")
        mtx = 0.5 * c * full_jac_T.dot(diag_mtx)

        return -mtx.toarray()

    def _opt_hess(self, sol, p, yt_dict):
        if _use_jax_opt():
            cache = self._get_jax_opt_data()
            return _opt_hess_jax(
                sol, p, yt_dict, cache, self.full_hess, self.full_jac_dict
            )

        full_hess = self.full_hess
        full_jac_dict = self.full_jac_dict

        hess = scipy.sparse.csc_matrix(full_hess)
        for kx in self.training_points[None]:
            full_jac, full_jac_T, c = full_jac_dict[kx]
            yt = yt_dict[kx]

            if p in (0, 1):
                diag_vec = np.zeros(full_jac.shape[0], dtype=np.float64)
            else:
                diag_vec = p * (p - 1) * (full_jac * sol - yt) ** (p - 2)
            diag_mtx = scipy.sparse.diags(diag_vec, format="csc")
            hess += 0.5 * c * full_jac_T * diag_mtx * full_jac

        return hess

    def _opt_norm(self, sol, p, yt_dict):
        grad = self._opt_grad(sol, p, yt_dict)
        return np.linalg.norm(grad)

    def _get_yt_dict(self, ind_y):
        yt_dict = {}
        for kx in self.training_points[None]:
            xt, yt = self.training_points[None][kx]
            yt_dict[kx] = yt[:, ind_y]
        return yt_dict

    def _run_newton_solver(self, sol):
        num = self.num
        options = self.options

        solver = get_solver(options["solver"])
        ls_class = get_line_search_class(options["line_search"])

        total_size = int(num["dof"])
        rhs = np.zeros((total_size, num["y"]))
        d_sol = np.zeros((total_size, num["y"]))

        p = self.options["approx_order"]
        for ind_y in range(rhs.shape[1]):
            with self.printer._timed_context("Solving for output %i" % ind_y):
                yt_dict = self._get_yt_dict(ind_y)

                norm = self._opt_norm(sol[:, ind_y], p, yt_dict)
                fval = self._opt_func(sol[:, ind_y], p, yt_dict)
                self.printer(
                    "Iteration (num., iy, grad. norm, func.) : %3i %3i %15.9e %15.9e"
                    % (0, ind_y, norm, fval)
                )

                iter_count = 0
                while (
                    iter_count < options["nonlinear_maxiter"]
                    and norm > options["solver_tolerance"]
                ):
                    with self.printer._timed_context():
                        with self.printer._timed_context("Assembling linear system"):
                            mtx = self._opt_hess(sol[:, ind_y], p, yt_dict)
                            rhs[:, ind_y] = -self._opt_grad(sol[:, ind_y], p, yt_dict)

                        with self.printer._timed_context("Initializing linear solver"):
                            solver._setup(mtx, self.printer)

                        with self.printer._timed_context("Solving linear system"):
                            d_sol[:, ind_y] = 0.0
                            solver._solve(rhs[:, ind_y], d_sol[:, ind_y], ind_y=ind_y)

                        def func(x):
                            return self._opt_func(x, p, yt_dict)

                        def grad(x):
                            return self._opt_grad(x, p, yt_dict)

                        # sol[:, ind_y] += d_sol[:, ind_y]

                        ls = ls_class(sol[:, ind_y], d_sol[:, ind_y], func, grad)
                        with self.printer._timed_context("Performing line search"):
                            sol[:, ind_y] = ls(1.0)

                    norm = self._opt_norm(sol[:, ind_y], p, yt_dict)
                    fval = self._opt_func(sol[:, ind_y], p, yt_dict)
                    self.printer(
                        "Iteration (num., iy, grad. norm, func.) : %3i %3i %15.9e %15.9e"
                        % (iter_count, ind_y, norm, fval)
                    )

                    self.mtx = mtx

                    iter_count += 1

                self.mtx = self._opt_hess(sol[:, ind_y], p, yt_dict)

    def _solve(self):
        num = self.num
        options = self.options

        total_size = int(num["dof"])
        sol = np.zeros((total_size, num["y"]))

        with self.printer._timed_context(
            "Solving initial startup problem (n=%i)" % total_size
        ):
            approx_order = options["approx_order"]
            nonlinear_maxiter = options["nonlinear_maxiter"]
            options["approx_order"] = 2
            options["nonlinear_maxiter"] = 1

            self._run_newton_solver(sol)

            options["approx_order"] = approx_order
            options["nonlinear_maxiter"] = nonlinear_maxiter

        with self.printer._timed_context(
            "Solving nonlinear problem (n=%i)" % total_size
        ):
            self._run_newton_solver(sol)

        return sol

    def _new_train(self):
        """
        Train the model
        """
        with self.printer._timed_context("Pre-computing matrices", "assembly"):
            with self.printer._timed_context("Computing dof2coeff", "dof2coeff"):
                self.full_dof2coeff = self._compute_dof2coeff()

            with self.printer._timed_context("Initializing Hessian", "init_hess"):
                self.full_hess = (
                    self._setup_hessian() * self.options["regularization_weight"]
                )

            if self.options["min_energy"]:
                with self.printer._timed_context("Computing energy terms", "energy"):
                    self.full_hess += (
                        self._compute_energy_terms() * self.options["energy_weight"]
                    )

            with self.printer._timed_context("Computing approximation terms", "approx"):
                self.full_jac_dict = self._compute_approx_terms()

        with self.printer._timed_context(
            "Solving for degrees of freedom", "total_solution"
        ):
            self.sol = self._solve()

        if self.full_dof2coeff is not None:
            self.sol_coeff = self.full_dof2coeff * self.sol
        else:
            self.sol_coeff = self.sol

    def _train(self):
        """
        Train the model
        """
        self._setup()

        inputs = {"self": self}
        with cached_operation(inputs, self.options["data_dir"]) as outputs:
            if outputs:
                self.sol_coeff = outputs["sol_coeff"]
                self.sol = outputs["sol"]
                self.mtx = outputs["mtx"]
                self.full_dof2coeff = outputs["full_dof2coeff"]
                self.full_hess = outputs["full_hess"]
                self.full_jac_dict = outputs["full_jac_dict"]
            else:
                self._new_train()
                outputs["sol_coeff"] = self.sol_coeff
                outputs["sol"] = self.sol
                outputs["mtx"] = self.mtx
                outputs["full_dof2coeff"] = self.full_dof2coeff
                outputs["full_hess"] = self.full_hess
                outputs["full_jac_dict"] = self.full_jac_dict

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
        mtx = self._compute_prediction_mtx(x, 0)
        y = mtx.dot(self.sol_coeff)

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
        y : np.ndarray
            Derivative values.
        """
        mtx = self._compute_prediction_mtx(x, kx + 1)
        y = mtx.dot(self.sol_coeff)

        return y

    def _compute_prediction_mtx(self, x, kx):
        n = x.shape[0]

        num = self.num
        options = self.options

        data, rows, cols = self._compute_jac_raw(kx, 0, x)

        # In the explanation below, n is the number of dimensions, and
        # a_k and b_k are the lower and upper bounds for x_k.
        #
        # A C1 extrapolation can get very tricky, so we implement a simple C0
        # extrapolation. We basically linearly extrapolate from the nearest
        # domain point. For example, if n = 4 and x2 > b2 and x3 > b3:
        #    f(x1,x2,x3,x4) = f(x1,b2,b3,x4) + dfdx2 (x2-b2) + dfdx3 (x3-b3)
        #    where the derivatives are evaluated at x1,b2,b3,x4 (called b) and
        #    dfdx1|x = dfdx1|b + d2fdx1dx2|b (x2-b2) + d2fdx1dx3|b (x3-b3)
        #    dfdx2|x = dfdx2|b.
        # The dfdx2|x derivative is what it is because f and all derivatives
        # evaluated at x1,b2,b3,x4 are constant with respect to changes in x2.
        # On the other hand, the dfdx1|x derivative is what it is because
        # f and all derivatives evaluated at x1,b2,b3,x4 change with x1.
        # The extrapolation function is non-differentiable at boundaries:
        # i.e., where x_k = a_k or x_k = b_k for at least one k.
        if options["extrapolate"]:
            # First we evaluate the vector pointing to each evaluation points
            # from the nearest point on the domain, in a matrix called dx.
            # If the ith evaluation point is not external, dx[i, :] = 0.
            xlimits = self.options["xlimits"]
            dx = compute_ext_dist(x, xlimits[:, 0], xlimits[:, 1], nterm=num["support"])

            isexternal = np.array(np.array(dx, bool), float)

            for ix in range(num["x"]):
                # Now we compute the first order term where we have a
                # derivative times (x_k - b_k) or (x_k - a_k).
                data_tmp, rows, cols = self._compute_jac_raw(kx, ix + 1, x)
                data_tmp *= dx[:, ix]

                # If we are evaluating a derivative (with index kx),
                # we zero the first order terms for which dx_k = 0.
                if kx != 0:
                    data_tmp *= 1 - isexternal[:, kx - 1]

                data += data_tmp
        elif kx != 0:
            xlimits = self.options["xlimits"]
            is_outside = np.any((x < xlimits[:, 0]) | (x > xlimits[:, 1]), axis=1)
            if np.any(is_outside):
                data = np.where(is_outside[rows], 0.0, data)

        mtx = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(n, num["coeff"]))
        return mtx

    def _predict_output_derivatives(self, x):
        # dy_dyt = dy_dw * (dR_dw)^{-1} * dR_dyt

        nw = self.mtx.shape[0]
        ny = self.sol.shape[1]

        p = self.options["approx_order"]

        dy_dw = self._compute_prediction_mtx(x, 0)
        if self.full_dof2coeff is not None:
            dy_dw = dy_dw * self.full_dof2coeff
        dy_dw = dy_dw.toarray()

        dR_dw = self.mtx

        dy_dyt = {}
        for kx in self.training_points[None]:
            nt = self.training_points[None][kx][0].shape[0]

            dR_dyt = np.zeros((nw, nt, ny))
            for ind_y in range(ny):
                yt_dict = self._get_yt_dict(ind_y)
                dR_dyt[:, :, ind_y] = self._opt_dgrad_dyt(
                    self.sol[:, ind_y], p, yt_dict, kx
                )

            solver = get_solver(self.options["derivative_solver"])
            solver._setup(dR_dw, self.printer)

            dw_dyt = np.zeros((nw, nt, ny))
            for ind_t in range(nt):
                for ind_y in range(ny):
                    solver._solve(
                        dR_dyt[:, ind_t, ind_y], dw_dyt[:, ind_t, ind_y], ind_y=ind_y
                    )
                    dw_dyt[:, ind_t, ind_y] *= -1.0

            if kx == 0:
                dy_dyt[None] = np.einsum("ij,jkl->ikl", dy_dw, dw_dyt)
            else:
                dy_dyt[kx - 1] = np.einsum("ij,jkl->ikl", dy_dw, dw_dyt)

        return dy_dyt

    def save(self, filename):
        persistence.save(self, filename)

    @staticmethod
    def load(filename):
        return persistence.load(filename)

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
            Dictionary with keys:

            - ``'knots'``: tuple of ``nx`` 1D arrays, the per-dimension knot
              vectors (open uniform, ``num_ctrl_pts[i] + order[i]`` knots each),
              in physical units or on ``[0, 1]`` depending on ``physical``.
            - ``'degree'``: tuple of ``nx`` ints, ``degree[i] = order[i] - 1``.
            - ``'coefficients'``: array of shape ``tuple(num_ctrl_pts)`` holding
              the control-point values of the selected output. Indexing is
              ``coefficients[i_0, ..., i_{nx-1}]`` (C order, last input varying
              fastest), i.e. the layout expected by ``scipy.interpolate.NdBSpline``.
            - ``'domain'``: array of shape ``(nx, 2)``, the ``xlimits`` (or
              ``[0, 1]`` bounds when ``physical=False``).

        Examples
        --------
        >>> bs = sm.get_bspline(output=0)
        >>> from scipy.interpolate import NdBSpline
        >>> spl = NdBSpline(bs["knots"], bs["coefficients"], bs["degree"])
        """
        st = _normalize_spline_type(self.options["spline_type"])
        if st != "B-Spline":
            raise ValueError(
                f"get_bspline() is only supported for 'B-Spline' models, "
                f"got spline_type={self.options['spline_type']!r}."
            )
        if not hasattr(self, "sol_coeff") or self.sol_coeff is None:
            raise RuntimeError(
                "RMTS model has not been trained yet. Call train() first."
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
        Convert the trained RMTS model for one output to a SciPy spline object.

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
        st = _normalize_spline_type(self.options["spline_type"])
        if st != "B-Spline":
            raise ValueError(
                f"to_scipy() is only supported for 'B-Spline' models, "
                f"got spline_type={self.options['spline_type']!r}."
            )
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
                "RMTS.to_scipy() requires SciPy >= 1.12.0 for scipy.interpolate.NdBSpline."
            ) from err

        return NdBSpline(bs["knots"], bs["coefficients"], bs["degree"])

    def get_jax_predictor(self, dtype=None):
        """
        Return a pure JAX function predict(x) that is end-to-end differentiable.

        Parameters
        ----------
        dtype : DTypeLike or str, optional
            Computation precision, e.g. jnp.float64 (default) or jnp.float32 for
            maximum GPU throughput on consumer Tensor Cores. If None, checks
            self.options["jax_dtype"].

        Returns
        -------
        predict_fn : Callable
            Differentiable JAX prediction function with weights cached in device VRAM.
        """
        if not hasattr(self, "sol_coeff") or self.sol_coeff is None:
            raise RuntimeError(
                f"{self.__class__.__name__} model has not been trained yet. Call train() first."
            )
        import jax
        import jax.numpy as jnp

        opt_dtype = (
            self.options["jax_dtype"] if "jax_dtype" in self.options else "float64"
        )
        raw_dtype = dtype if dtype is not None else opt_dtype
        if isinstance(raw_dtype, str):
            want_64 = "32" not in raw_dtype
        else:
            want_64 = jnp.dtype(raw_dtype) == jnp.float64
        if dtype is None and want_64 and not jax.config.jax_enable_x64:
            use_dtype = jnp.float32
        else:
            use_dtype = jnp.float64 if want_64 else jnp.float32

        spline_type = _normalize_spline_type(self.options["spline_type"])
        extrapolate = bool(self.options["extrapolate"])

        lower_j = jnp.asarray(self.options["xlimits"][:, 0], dtype=use_dtype)
        upper_j = jnp.asarray(self.options["xlimits"][:, 1], dtype=use_dtype)
        coeff_arr = jnp.asarray(self.sol_coeff, dtype=use_dtype)
        if coeff_arr.ndim == 1:
            coeff_arr = coeff_arr[:, None]
        ny = coeff_arr.shape[1]
        squeeze_y = ny == 1

        if spline_type == "B-Spline":
            from smt.surrogate_models.rmts_backend import _get_predict_rmtb_jax_fn

            order_tuple = tuple(int(v) for v in self.num["order_list"])
            ncp_tuple = tuple(int(v) for v in self.num["ctrl_list"])
            knots_tuple = tuple(
                jnp.asarray(k, dtype=use_dtype) for k in self.knots_list
            )
            coeff_nd = coeff_arr.reshape(ncp_tuple + (ny,))

            fn = _get_predict_rmtb_jax_fn(order_tuple, ncp_tuple, extrapolate)

            def predict_fn(x):
                x_j = jnp.asarray(x, dtype=use_dtype)
                squeeze_x = x_j.ndim == 1
                if squeeze_x:
                    x_j = x_j[None, :]
                y = fn(x_j, coeff_nd, lower_j, upper_j, knots_tuple)
                if squeeze_x:
                    return y[0, 0] if squeeze_y else y[0]
                return y

            return predict_fn

        elif spline_type == "Cubic_Hermite_Spline":
            from smt.surrogate_models.rmts_backend import _get_predict_rmtc_jax_fn

            nelem_tuple = tuple(int(v) for v in self.num["elem_list"])
            nterm_tuple = (
                tuple(int(v) for v in self.num["term_list"])
                if "term_list" in self.num
                else (4,) * len(nelem_tuple)
            )
            nelem = 1
            for m in nelem_tuple:
                nelem *= m
            nterm = 1
            for nt in nterm_tuple:
                nterm *= nt
            coeff_nd = coeff_arr.reshape((nelem, nterm, ny))

            fn = _get_predict_rmtc_jax_fn(nelem_tuple, nterm_tuple, extrapolate)

            def predict_fn(x):
                x_j = jnp.asarray(x, dtype=use_dtype)
                squeeze_x = x_j.ndim == 1
                if squeeze_x:
                    x_j = x_j[None, :]
                y = fn(x_j, coeff_nd, lower_j, upper_j)
                if squeeze_x:
                    return y[0, 0] if squeeze_y else y[0]
                return y

            return predict_fn

        else:
            raise ValueError(f"Unknown spline_type: {spline_type}")


class RMTB(RMTS):
    """
    Regularized Minimal-energy Tensor-product B-spline (RMTB) interpolant.

    .. deprecated:: 2.x
       Use ``RMTS(spline_type='B-Spline')`` instead.
    """

    name = "RMTB"

    def __init__(self, **kwargs):
        # Only warn on direct use: subclasses (e.g. RMTBUQ) are not deprecated.
        if type(self) is RMTB:
            import warnings

            warnings.warn(
                "RMTB is deprecated, please use RMTS(spline_type='B-Spline') instead.",
                FutureWarning,
                stacklevel=2,
            )
        super().__init__(**kwargs)

    def _initialize(self):
        super()._initialize()
        self.options["spline_type"] = "B-Spline"

    def _final_initialize(self):
        st = _normalize_spline_type(self.options["spline_type"])
        if st != "B-Spline":
            raise ValueError(
                f"RMTB only supports spline_type='B-Spline', got {self.options['spline_type']!r}."
            )
        super()._final_initialize()


class RMTC(RMTS):
    """
    Regularized Minimal-energy Tensor-product Cubic Hermite spline (RMTC) interpolant.

    .. deprecated:: 2.x
       Use ``RMTS(spline_type='Cubic_Hermite_Spline')`` instead.
    """

    name = "RMTC"

    def __init__(self, **kwargs):
        # Only warn on direct use: subclasses are not deprecated.
        if type(self) is RMTC:
            import warnings

            warnings.warn(
                "RMTC is deprecated, please use RMTS(spline_type='Cubic_Hermite_Spline') instead.",
                FutureWarning,
                stacklevel=2,
            )
        super().__init__(**kwargs)

    def _initialize(self):
        super()._initialize()
        self.options["spline_type"] = "Cubic_Hermite_Spline"

    def _final_initialize(self):
        st = _normalize_spline_type(self.options["spline_type"])
        if st != "Cubic_Hermite_Spline":
            raise ValueError(
                "RMTC only supports spline_type='Cubic_Hermite_Spline', "
                f"got {self.options['spline_type']!r}."
            )
        super()._final_initialize()
