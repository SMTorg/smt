"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""
import numpy as np
import scipy.sparse
from numbers import Integral

from smt.utils.linear_solvers import get_solver, LinearSolver, VALID_SOLVERS
from smt.utils.line_search import get_line_search_class, LineSearch, VALID_LINE_SEARCHES
from smt.utils.caching import cached_operation
from smt.surrogate_models.surrogate_model import SurrogateModel


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

        supports["training_derivatives"] = True
        supports["derivatives"] = True
        supports["output_derivatives"] = True

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
        num = self.num
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
                self.options["grad_weight"] / xlimits.shape[0]

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
        if self.__class__.__name__ == "RMTB":
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
                n = np.prod(2 * num["elem_list"])
                x = np.empty(n * num["x"])
                self.rmtsc.compute_quadrature_points(
                    n, np.array(2 * num["elem_list"], dtype=np.int32), x
                )
                x = x.reshape((n, num["x"]))

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

    def _opt_func(self, sol, p, yt_dict):
        full_hess = self.full_hess
        full_jac_dict = self.full_jac_dict

        func = 0.5 * np.dot(sol, full_hess * sol)
        for kx in self.training_points[None]:
            full_jac, full_jac_T, c = full_jac_dict[kx]
            yt = yt_dict[kx]
            func += 0.5 * c * np.sum((full_jac * sol - yt) ** p)

        return func

    def _opt_grad(self, sol, p, yt_dict):
        full_hess = self.full_hess
        full_jac_dict = self.full_jac_dict

        grad = full_hess * sol
        for kx in self.training_points[None]:
            full_jac, full_jac_T, c = full_jac_dict[kx]
            yt = yt_dict[kx]
            grad += 0.5 * c * full_jac_T * p * (full_jac * sol - yt) ** (p - 1)

        return grad

    def _opt_dgrad_dyt(self, sol, p, yt_dict, kx):
        full_hess = self.full_hess
        full_jac_dict = self.full_jac_dict

        full_jac, full_jac_T, c = full_jac_dict[kx]
        yt = yt_dict[kx]

        diag_vec = p * (p - 1) * (full_jac * sol - yt) ** (p - 2)
        diag_mtx = scipy.sparse.diags(diag_vec, format="csc")
        mtx = 0.5 * c * full_jac_T.dot(diag_mtx)

        return -mtx.todense()

    def _opt_hess(self, sol, p, yt_dict):
        full_hess = self.full_hess
        full_jac_dict = self.full_jac_dict

        hess = scipy.sparse.csc_matrix(full_hess)
        for kx in self.training_points[None]:
            full_jac, full_jac_T, c = full_jac_dict[kx]
            yt = yt_dict[kx]

            diag_vec = p * (p - 1) * (full_jac * sol - yt) ** (p - 2)
            diag_mtx = scipy.sparse.diags(diag_vec, format="csc")
            hess += 0.5 * c * full_jac_T * diag_mtx * full_jac

        return hess

    def _opt_norm(self, sol, p, yt_dict):
        full_hess = self.full_hess
        full_jac_dict = self.full_jac_dict

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
                            solver._solve(rhs[:, ind_y], d_sol[:, ind_y], ind_y=ind_y)

                        func = lambda x: self._opt_func(x, p, yt_dict)
                        grad = lambda x: self._opt_grad(x, p, yt_dict)

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

    def _solve(self):
        num = self.num
        options = self.options

        solver = get_solver(options["solver"])
        ls_class = get_line_search_class(options["line_search"])

        total_size = int(num["dof"])
        rhs = np.zeros((total_size, num["y"]))
        sol = np.zeros((total_size, num["y"]))
        d_sol = np.zeros((total_size, num["y"]))

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

        tmp = self.rmtsc
        self.rmtsc = None

        inputs = {"self": self}
        with cached_operation(inputs, self.options["data_dir"]) as outputs:
            self.rmtsc = tmp

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
        # extrapolation. We basically linarly extrapolate from the nearest
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
            dx = np.empty(n * num["support"] * num["x"])
            self.rmtsc.compute_ext_dist(n, num["support"], x.flatten(), dx)
            dx = dx.reshape((n * num["support"], num["x"]))

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

        mtx = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(n, num["coeff"]))
        return mtx

    def _predict_output_derivatives(self, x):
        # dy_dyt = dy_dw * (dR_dw)^{-1} * dR_dyt

        n = x.shape[0]
        nw = self.mtx.shape[0]
        nx = x.shape[1]
        ny = self.sol.shape[1]

        p = self.options["approx_order"]

        dy_dw = self._compute_prediction_mtx(x, 0)
        if self.full_dof2coeff is not None:
            dy_dw = dy_dw * self.full_dof2coeff
        dy_dw = dy_dw.todense()

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
