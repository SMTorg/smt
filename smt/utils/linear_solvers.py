"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
        
This package is distributed under New BSD license.
"""

import numpy as np
import scipy.sparse.linalg
import scipy.linalg
import contextlib

from smt.utils.options_dictionary import OptionsDictionary

VALID_SOLVERS = (
    "krylov-dense",
    "dense-lu",
    "dense-chol",
    "lu",
    "ilu",
    "krylov",
    "krylov-lu",
    "krylov-mg",
    "gs",
    "jacobi",
    "mg",
    "null",
)


def get_solver(solver):
    if solver == "dense-lu":
        return DenseLUSolver()
    elif solver == "dense-chol":
        return DenseCholeskySolver()
    elif solver == "krylov-dense":
        return KrylovSolver(pc="dense")
    elif solver == "lu" or solver == "ilu":
        return DirectSolver(alg=solver)
    elif solver == "krylov":
        return KrylovSolver()
    elif solver == "krylov-lu":
        return KrylovSolver(pc="lu")
    elif solver == "krylov-mg":
        return KrylovSolver(pc="mg")
    elif solver == "gs" or solver == "jacobi":
        return StationarySolver(solver=solver)
    elif solver == "mg":
        return MultigridSolver()
    elif isinstance(solver, LinearSolver):
        return solver
    elif solver == "null":
        return NullSolver()
    elif solver == None:
        return None


class Callback(object):
    def __init__(self, size, string, interval, printer):
        self.size = size
        self.string = string
        self.interval = interval
        self.printer = printer

        self.counter = 0
        self.ind_y = 0
        self.mtx = None
        self.rhs = None
        self.norm0 = 1.0

    def _print_norm(self, norm):
        if self.counter == 0:
            self.norm0 = norm

        if self.counter % self.interval == 0:
            self.printer(
                "%s (%i x %i mtx), output %-3i : %3i  %15.9e  %15.9e"
                % (
                    self.string,
                    self.size,
                    self.size,
                    self.ind_y,
                    self.counter,
                    norm,
                    norm / self.norm0,
                )
            )
        self.counter += 1

    def _print_res(self, res):
        self._print_norm(res)

    def _print_sol(self, sol):
        res = self.mtx.dot(sol) - self.rhs
        norm = np.linalg.norm(res)
        self._print_norm(norm)


class LinearSolver(object):
    def __init__(self, **kwargs):
        self.mtx = None
        self.rhs = None

        self.options = OptionsDictionary()
        self.options.declare("print_init", True, types=bool)
        self.options.declare("print_solve", True, types=bool)
        self._initialize()
        self.options.update(kwargs)

    def _initialize(self):
        pass

    def _setup(self, mtx, printer, mg_matrices=[]):
        pass

    def _solve(self, rhs, sol=None, ind_y=0):
        pass

    def _clone(self):
        clone = self.__class__()
        clone.options.update(clone.options._dict)
        return clone

    @contextlib.contextmanager
    def _active(self, active):
        orig_active = self.printer.active

        self.printer.active = self.printer.active and active
        yield self.printer
        self.printer.active = orig_active


class NullSolver(LinearSolver):
    def solve(self, rhs, sol=None, ind_y=0):
        pass


class DenseCholeskySolver(LinearSolver):
    def _setup(self, mtx, printer, mg_matrices=[]):
        self.printer = printer
        with self._active(self.options["print_init"]) as printer:
            self.mtx = mtx.A
            assert isinstance(self.mtx, np.ndarray), "mtx is of type %s" % type(mtx)

            with printer._timed_context(
                "Performing Chol. fact. (%i x %i mtx)" % mtx.shape
            ):
                self.upper = scipy.linalg.cholesky(self.mtx)

    def _solve(self, rhs, sol=None, ind_y=0):
        with self._active(self.options["print_solve"]) as printer:
            self.rhs = rhs

            if sol is None:
                sol = np.array(rhs)

            with printer._timed_context("Back solving (%i x %i mtx)" % self.mtx.shape):
                sol[:] = rhs
                scipy.linalg.solve_triangular(
                    self.upper, sol, overwrite_b=True, trans="T"
                )
                scipy.linalg.solve_triangular(self.upper, sol, overwrite_b=True)

        return sol


class DenseLUSolver(LinearSolver):
    def _setup(self, mtx, printer, mg_matrices=[]):
        self.printer = printer
        with self._active(self.options["print_init"]) as printer:
            self.mtx = mtx
            assert isinstance(mtx, np.ndarray), "mtx is of type %s" % type(mtx)

            with printer._timed_context(
                "Performing LU fact. (%i x %i mtx)" % mtx.shape
            ):
                self.fact = scipy.linalg.lu_factor(mtx)

    def _solve(self, rhs, sol=None, ind_y=0):
        with self._active(self.options["print_solve"]) as printer:
            self.rhs = rhs

            if sol is None:
                sol = np.array(rhs)

            with printer._timed_context("Back solving (%i x %i mtx)" % self.mtx.shape):
                sol[:] = scipy.linalg.lu_solve(self.fact, rhs)

        return sol


class DirectSolver(LinearSolver):
    def _initialize(self):
        self.options.declare("alg", "lu", values=["lu", "ilu"])

    def _setup(self, mtx, printer, mg_matrices=[]):
        self.printer = printer
        with self._active(self.options["print_init"]) as printer:
            self.mtx = mtx
            assert isinstance(mtx, scipy.sparse.spmatrix), "mtx is of type %s" % type(
                mtx
            )

            with printer._timed_context(
                "Performing %s fact. (%i x %i mtx)"
                % ((self.options["alg"],) + mtx.shape)
            ):
                if self.options["alg"] == "lu":
                    self.fact = scipy.sparse.linalg.splu(mtx)
                elif self.options["alg"] == "ilu":
                    self.fact = scipy.sparse.linalg.spilu(
                        mtx, drop_rule="interp", drop_tol=1e-3, fill_factor=2
                    )

    def _solve(self, rhs, sol=None, ind_y=0):
        with self._active(self.options["print_solve"]) as printer:
            self.rhs = rhs

            if sol is None:
                sol = np.array(rhs)

            with printer._timed_context("Back solving (%i x %i mtx)" % self.mtx.shape):
                sol[:] = self.fact.solve(rhs)

        return sol


class KrylovSolver(LinearSolver):
    def _initialize(self):
        self.options.declare("interval", 10, types=int)
        self.options.declare("solver", "cg", values=["cg", "bicgstab", "gmres"])
        self.options.declare(
            "pc",
            None,
            values=[None, "ilu", "lu", "gs", "jacobi", "mg", "dense"],
            types=LinearSolver,
        )
        self.options.declare("ilimit", 100, types=int)
        self.options.declare("atol", 1e-15, types=(int, float))
        self.options.declare("rtol", 1e-15, types=(int, float))

    def _setup(self, mtx, printer, mg_matrices=[]):
        self.printer = printer
        with self._active(self.options["print_init"]) as printer:
            self.mtx = mtx

            pc_solver = get_solver(self.options["pc"])

            if pc_solver is not None:
                pc_solver._setup(mtx, printer, mg_matrices=mg_matrices)
                self.pc_solver = pc_solver
                self.pc_op = scipy.sparse.linalg.LinearOperator(
                    mtx.shape, matvec=pc_solver._solve
                )
            else:
                self.pc_solver = None
                self.pc_op = None

            self.callback = Callback(
                mtx.shape[0], "Krylov solver", self.options["interval"], printer
            )
            if self.options["solver"] == "cg":
                self.solver = scipy.sparse.linalg.cg
                self.callback_func = self.callback._print_sol
                self.solver_kwargs = {
                    "atol": "legacy",
                    "tol": self.options["atol"],
                    "maxiter": self.options["ilimit"],
                }
            elif self.options["solver"] == "bicgstab":
                self.solver = scipy.sparse.linalg.bicgstab
                self.callback_func = self.callback._print_sol
                self.solver_kwargs = {
                    "tol": self.options["atol"],
                    "maxiter": self.options["ilimit"],
                }
            elif self.options["solver"] == "gmres":
                self.solver = scipy.sparse.linalg.gmres
                self.callback_func = self.callback._print_res
                self.solver_kwargs = {
                    "tol": self.options["atol"],
                    "maxiter": self.options["ilimit"],
                    "restart": min(self.options["ilimit"], mtx.shape[0]),
                }

    def _solve(self, rhs, sol=None, ind_y=0):
        with self._active(self.options["print_solve"]) as printer:
            self.rhs = rhs

            if sol is None:
                sol = np.array(rhs)

            with printer._timed_context(
                "Running %s Krylov solver (%i x %i mtx)"
                % ((self.options["solver"],) + self.mtx.shape)
            ):
                self.callback.counter = 0
                self.callback.ind_y = ind_y
                self.callback.mtx = self.mtx
                self.callback.rhs = rhs

                self.callback._print_sol(sol)
                tmp, info = self.solver(
                    self.mtx,
                    rhs,
                    x0=sol,
                    M=self.pc_op,
                    callback=self.callback_func,
                    **self.solver_kwargs
                )

            sol[:] = tmp

        return sol


class StationarySolver(LinearSolver):
    def _initialize(self):
        self.options.declare("interval", 10, types=int)
        self.options.declare("solver", "gs", values=["gs", "jacobi"])
        self.options.declare("damping", 1.0, types=(int, float))
        self.options.declare("ilimit", 10, types=int)

    def _setup(self, mtx, printer, mg_matrices=[]):
        self.printer = printer
        with self._active(self.options["print_init"]) as printer:
            self.mtx = mtx

            self.callback = Callback(
                mtx.shape[0], "Stationary solver", self.options["interval"], printer
            )

            with printer._timed_context(
                "Initializing %s solver (%i x %i mtx)"
                % ((self.options["solver"],) + self.mtx.shape)
            ):
                if self.options["solver"] == "jacobi":
                    # A x = b
                    # x_{k+1} = x_k + w D^{-1} (b - A x_k)
                    self.d_inv = self.options["damping"] / self._split_mtx_diag()
                    self.iterate = self._jacobi

                elif self.options["solver"] == "gs":
                    # A x = b
                    # x_{k+1} = x_k + (1/w D + L)^{-1} (b - A x_k)
                    mtx_d = self._split_mtx("diag")
                    mtx_l = self._split_mtx("lower")
                    mtx_ldw = mtx_l + mtx_d / self.options["damping"]
                    self.inv = scipy.sparse.linalg.splu(mtx_ldw)
                    self.iterate = self._gs

    def _split_mtx_diag(self):
        shape = self.mtx.shape
        rows, cols, data = scipy.sparse.find(self.mtx)

        mask_d = rows == cols
        diag = np.zeros(shape[0])
        np.add.at(diag, rows[mask_d], data[mask_d])
        return diag

    def _split_mtx(self, part):
        shape = self.mtx.shape
        rows, cols, data = scipy.sparse.find(self.mtx)

        if part == "diag":
            mask = rows == cols
        elif part == "lower":
            mask = rows > cols
        elif part == "upper":
            mask = rows < cols

        return scipy.sparse.csc_matrix(
            (data[mask], (rows[mask], cols[mask])), shape=shape
        )

    def _jacobi(self, rhs, sol):
        # A x = b
        # x_{k+1} = x_k + w D^{-1} (b - A x_k)
        sol += self.d_inv * (rhs - self.mtx.dot(sol))

    def _gs(self, rhs, sol):
        # A x = b
        # x_{k+1} = x_k + (1/w D + L)^{-1} (b - A x_k)
        sol += self.inv.solve(rhs - self.mtx.dot(sol))

    def _solve(self, rhs, sol=None, ind_y=0):
        with self._active(self.options["print_solve"]) as printer:
            self.rhs = rhs

            if sol is None:
                sol = np.array(rhs)

            self.callback.counter = 0
            self.callback.ind_y = ind_y
            self.callback.mtx = self.mtx
            self.callback.rhs = rhs

            with printer._timed_context(
                "Running %s stationary solver (%i x %i mtx)"
                % ((self.options["solver"],) + self.mtx.shape)
            ):
                for ind in range(self.options["ilimit"]):
                    self.iterate(rhs, sol)
                    self.callback._print_sol(sol)

        return sol


class MultigridSolver(LinearSolver):
    def _initialize(self):
        self.options.declare("interval", 1, types=int)
        self.options.declare("mg_cycles", 0, types=int)
        self.options.declare(
            "solver",
            "null",
            values=["null", "gs", "jacobi", "krylov"],
            types=LinearSolver,
        )

    def _setup(self, mtx, printer, mg_matrices=[]):
        self.printer = printer
        with self._active(self.options["print_init"]) as printer:
            self.mtx = mtx

            solver = get_solver(self.options["solver"])
            mg_solver = solver._clone()
            mg_solver._setup(mtx, printer)

            self.mg_mtx = [mtx]
            self.mg_sol = [np.zeros(self.mtx.shape[0])]
            self.mg_rhs = [np.zeros(self.mtx.shape[0])]
            self.mg_ops = []
            self.mg_solvers = [mg_solver]

            for ind, mg_op in enumerate(mg_matrices):
                mg_mtx = mg_op.T.dot(self.mg_mtx[-1]).dot(mg_op).tocsc()
                mg_sol = mg_op.T.dot(self.mg_sol[-1])
                mg_rhs = mg_op.T.dot(self.mg_rhs[-1])

                mg_solver = solver._clone()
                mg_solver._setup(mg_mtx, printer)

                self.mg_mtx.append(mg_mtx)
                self.mg_sol.append(mg_sol)
                self.mg_rhs.append(mg_rhs)
                self.mg_ops.append(mg_op)
                self.mg_solvers.append(mg_solver)

            mg_mtx = self.mg_mtx[-1]
            mg_solver = DirectSolver()
            mg_solver._setup(mg_mtx, printer)
            self.mg_solvers[-1] = mg_solver

            self.callback = Callback(
                mtx.shape[0], "Multigrid solver", self.options["interval"], printer
            )

    def _restrict(self, ind_level):
        mg_op = self.mg_ops[ind_level]
        mtx = self.mg_mtx[ind_level]
        sol = self.mg_sol[ind_level]
        rhs = self.mg_rhs[ind_level]

        res = rhs - mtx.dot(sol)
        res_coarse = mg_op.T.dot(res)
        self.mg_rhs[ind_level + 1][:] = res_coarse

    def _smooth_and_restrict(self, ind_level, ind_cycle, ind_y):
        mg_op = self.mg_ops[ind_level]
        mtx = self.mg_mtx[ind_level]
        sol = self.mg_sol[ind_level]
        rhs = self.mg_rhs[ind_level]
        solver = self.mg_solvers[ind_level]
        solver.print_info = "MG iter %i level %i" % (ind_cycle, ind_level)

        solver._solve(rhs, sol, ind_y)

        res = rhs - mtx.dot(sol)
        res_coarse = mg_op.T.dot(res)
        self.mg_rhs[ind_level + 1][:] = res_coarse

    def _coarse_solve(self, ind_cycle, ind_y):
        sol = self.mg_sol[-1]
        rhs = self.mg_rhs[-1]
        solver = self.mg_solvers[-1]
        solver.print_info = "MG iter %i level %i" % (ind_cycle, len(self.mg_ops))
        solver._solve(rhs, sol, ind_y)

    def _smooth_and_interpolate(self, ind_level, ind_cycle, ind_y):
        mg_op = self.mg_ops[ind_level]
        mtx = self.mg_mtx[ind_level]
        sol = self.mg_sol[ind_level]
        rhs = self.mg_rhs[ind_level]
        solver = self.mg_solvers[ind_level]
        solver.print_info = "MG iter %i level %i" % (ind_cycle, ind_level)

        sol_coarse = self.mg_sol[ind_level + 1]
        sol += mg_op.dot(sol_coarse)

        solver._solve(rhs, sol, ind_y)

    def _solve(self, rhs, sol=None, ind_y=0):
        with self._active(self.options["print_solve"]) as printer:
            self.rhs = rhs

            if sol is None:
                sol = np.array(rhs)

            orig_sol = sol

            self.callback.counter = 0
            self.callback.ind_y = ind_y
            self.callback.mtx = self.mtx
            self.callback.rhs = rhs

            self.mg_rhs[0][:] = rhs

            for ind_level in range(len(self.mg_ops)):
                self._restrict(ind_level)

            self._coarse_solve(-1, ind_y)

            for ind_level in range(len(self.mg_ops) - 1, -1, -1):
                self._smooth_and_interpolate(ind_level, -1, ind_y)

            for ind_cycle in range(self.options["mg_cycles"]):

                for ind_level in range(len(self.mg_ops)):
                    self._smooth_and_restrict(ind_level, ind_cycle, ind_y)

                self._coarse_solve(ind_cycle, ind_y)

                for ind_level in range(len(self.mg_ops) - 1, -1, -1):
                    self._smooth_and_interpolate(ind_level, ind_cycle, ind_y)

            orig_sol[:] = self.mg_sol[0]

        return orig_sol
