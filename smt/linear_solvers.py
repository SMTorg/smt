"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
"""

from __future__ import print_function
import numpy as np
import scipy.sparse.linalg
import pyamg.krylov
import six
from six.moves import range

from smt.utils import OptionsDictionary, Printer, Timer


def get_solver(solver):
    if solver == 'lu' or solver == 'ilu':
        return DirectSolver(alg=solver)
    elif solver == 'krylov':
        return KrylovSolver()
    elif solver == 'gs' or solver == 'jacobi':
        return StationarySolver(solver=solver)
    elif solver == 'mg':
        return MultigridSolver()
    elif isinstance(solver, LinearSolver):
        return solver


class LinearSolver(object):

    def __init__(self, **kwargs):
        self.mtx = None
        self.rhs = None

        self.callback = None
        self.counter = 0
        self.ind_y = 0
        self.print_info = ''
        self.norm0 = 0

        self.printer = Printer()
        self.timer = Timer()

        self.options = OptionsDictionary()
        self.options.declare('print_init', True, types=bool)
        self.options.declare('print_solve', False, types=bool)
        self._declare_options()
        self.options.update(kwargs)

    def _declare_options(self):
        pass

    def _initialize(self, mtx, mg_matrices=[], print_status=True):
        pass

    def _solve(self, rhs, sol=None, ind_y=0, print_status=True):
        pass

    def _clone(self):
        clone = self.__class__()
        clone.options.update(clone.options._dict)
        return clone

    def _print_res(self, res):
        norm = res
        if self.counter == 0:
            self.norm0 = norm

        if self.counter % self.options['interval'] == 0:
            self.printer('   %s (%i x %i mtx), output %-3i : %3i  %15.9e  %15.9e' %
                (self.__class__.__name__ + str(self.print_info),
                 self.mtx.shape[0], self.mtx.shape[1],
                 self.ind_y, self.counter, norm, norm / self.norm0))
        self.counter += 1

    def _print_sol(self, sol):
        res = self.mtx.dot(sol) - self.rhs
        norm = np.linalg.norm(res)
        if self.counter == 0:
            self.norm0 = norm

        if self.counter % self.options['interval'] == 0:
            self.printer('   %s (%i x %i mtx), output %-3i : %3i  %15.9e  %15.9e' %
                (self.__class__.__name__ + str(self.print_info),
                 self.mtx.shape[0], self.mtx.shape[1],
                 self.ind_y, self.counter, norm, norm / self.norm0))
        self.counter += 1


class NullSolver(LinearSolver):

    def solve(self, rhs, sol=None, ind_y=0, print_status=True):
        pass


class DirectSolver(LinearSolver):

    def _declare_options(self):
        self.options.declare('alg', 'lu', values=['lu', 'ilu'])

    def _initialize(self, mtx, mg_matrices=[], print_status=True):
        self.printer.active = print_status and self.options['print_init']
        self.mtx = mtx

        self.printer._operation('Performing %s fact. (%i x %i mtx)' % \
            (self.options['alg'], mtx.shape[0], mtx.shape[1]))
        self.timer._start('direct')

        if self.options['alg'] == 'lu':
            self.fact = scipy.sparse.linalg.splu(mtx)
        elif self.options['alg'] == 'ilu':
            self.fact = scipy.sparse.linalg.spilu(
                mtx, drop_rule='interp',
                drop_tol=1e-3, #1e-3,
                fill_factor=2, #1,
            )
        self.timer._stop('direct')
        self.printer._done_time(self.timer['direct'])

    def _solve(self, rhs, sol=None, ind_y=0, print_status=True):
        self.printer.active = print_status and self.options['print_solve']
        self.rhs = rhs

        if sol is None:
            sol = np.array(rhs)

        self.printer._operation('Back solving (%i x %i mtx)' % self.mtx.shape)
        self.timer._start('backsol')

        sol[:] = self.fact.solve(rhs)

        self.timer._stop('backsol')
        self.printer._done_time(self.timer['backsol'])

        return sol


class KrylovSolver(LinearSolver):

    def _declare_options(self):
        self.options.declare('interval', 10, types=int)
        self.options.declare('solver', 'gmres', values=['cg', 'bicgstab', 'gmres', 'fgmres'])
        self.options.declare('pc', 'nopc', values=['ilu', 'lu', 'nopc', 'gs', 'jacobi'],
                             types=LinearSolver)
        self.options.declare('ilimit', 100, types=int)
        self.options.declare('atol', 1e-15, types=(int, float))
        self.options.declare('rtol', 1e-15, types=(int, float))

    def _initialize(self, mtx, mg_matrices=[], print_status=True):
        self.printer.active = print_status and self.options['print_init']
        self.mtx = mtx

        if self.options['pc'] == 'nopc':
            pc_solver = None
        elif self.options['pc'] == 'lu' or self.options['pc'] == 'ilu':
            pc_solver = DirectSolver(alg=self.options['pc'])
        elif self.options['pc'] == 'gs' or self.options['pc'] == 'jacobi':
            pc_solver = StationarySolver(solver=self.options['pc'], damping=1.0, ilimit=1)
        elif self.options['pc'] == 'mg':
            pc_solver = MultigridSolver(mg_ops=mg_ops)
        elif isinstance(self.options['pc'], LinearSolver):
            pc_solver = self.options['pc']

        if pc_solver is not None:
            pc_solver._initialize(mtx, mg_matrices=mg_matrices, print_status=print_status)
            self.pc_solver = pc_solver
            self.pc_op = scipy.sparse.linalg.LinearOperator(mtx.shape, matvec=solver.solve)
        else:
            self.pc_solver = None
            self.pc_op = None

        if self.options['solver'] == 'cg':
            self.solver = scipy.sparse.linalg.cg
            self.callback = self._print_sol
            self.solver_kwargs = {'tol': self.options['atol'],
                                  'maxiter': self.options['ilimit'],
                                  }
        elif self.options['solver'] == 'bicgstab':
            self.solver = scipy.sparse.linalg.bicgstab
            self.callback = self._print_sol
            self.solver_kwargs = {'tol': self.options['atol'],
                                  'maxiter': self.options['ilimit'],
                                  }
        elif self.options['solver'] == 'gmres':
            self.solver = scipy.sparse.linalg.gmres
            self.callback = self._print_res
            self.solver_kwargs = {'tol': self.options['atol'],
                                  'maxiter': self.options['ilimit'],
                                  'restart': min(self.options['ilimit'], mtx.shape[0])}
        elif self.options['solver'] == 'fgmres':
            self.solver = pyamg.krylov.fgmres
            self.callback = self._print_sol
            self.solver_kwargs = {'tol': self.options['rtol'],
                                  'maxiter': self.options['ilimit'],
                                  'restrt': 1,
                                  # 'restrt': min(self.options['ilimit'], mtx.shape[0]),
                                  # 'maxiter': 1, 'restrt': 300,
                                  }

    def _solve(self, rhs, sol=None, ind_y=0, print_status=True):
        self.printer.active = print_status and self.options['print_solve']
        self.rhs = rhs

        if sol is None:
            sol = np.array(rhs)

        self.printer._operation('Running %s Krylov solver (%i x %i mtx)' % \
            (self.options['solver'], self.mtx.shape[0], self.mtx.shape[1]))
        self.timer._start('krylov')

        self.counter = 0
        self.ind_y = ind_y

        if self.pc_solver is not None:
            self.pc_solver.printer.active = print_status

        self._print_sol(sol)
        tmp, info = self.solver(
            self.mtx, rhs, x0=sol, M=self.pc_op,
            callback=self.callback,
            **self.solver_kwargs
        )
        self.printer()

        self.timer._stop('krylov')
        self.printer._done_time(self.timer['krylov'])

        sol[:] = tmp

        return sol


class StationarySolver(LinearSolver):

    def _declare_options(self):
        self.options.declare('interval', 10, types=int)
        self.options.declare('solver', 'gs', values=['gs', 'jacobi'])
        self.options.declare('damping', 1.0, types=(int, float))
        self.options.declare('ilimit', 10, types=int)

    def _initialize(self, mtx, mg_matrices=[], print_status=True):
        self.printer.active = print_status and self.options['print_init']
        self.mtx = mtx

        self.printer._operation('Initializing %s solver (%i x %i mtx)' % \
            (self.options['solver'], self.mtx.shape[0], self.mtx.shape[1]))
        self.timer._start('stationary')

        if self.options['solver'] == 'jacobi':
            # A x = b
            # x_{k+1} = x_k + w D^{-1} (b - A x_k)
            self.d_inv = self.options['damping'] / self._split_mtx_diag()
            self.iterate = self._jacobi

        elif self.options['solver'] == 'gs':
            # A x = b
            # x_{k+1} = x_k + (1/w D + L)^{-1} (b - A x_k)
            mtx_d = self._split_mtx('diag')
            mtx_l = self._split_mtx('lower')
            mtx_ldw = mtx_l + mtx_d / self.options['damping']
            self.inv = scipy.sparse.linalg.splu(mtx_ldw)
            self.iterate = self._gs

        self.timer._stop('stationary')
        self.printer._done_time(self.timer['stationary'])

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

        if part == 'diag':
            mask = rows == cols
        elif part == 'lower':
            mask = rows > cols
        elif part == 'upper':
            mask = rows < cols

        return scipy.sparse.csc_matrix((data[mask], (rows[mask], cols[mask])), shape=shape)

    def _jacobi(self, rhs, sol):
        # A x = b
        # x_{k+1} = x_k + w D^{-1} (b - A x_k)
        sol += self.d_inv * (rhs - self.mtx.dot(sol))

    def _gs(self, rhs, sol):
        # A x = b
        # x_{k+1} = x_k + (1/w D + L)^{-1} (b - A x_k)
        sol += self.inv.solve(rhs - self.mtx.dot(sol))

    def _solve(self, rhs, sol=None, ind_y=0, print_status=True):
        self.printer.active = print_status and self.options['print_solve']
        self.rhs = rhs

        if sol is None:
            sol = np.array(rhs)

        self.counter = 0
        self.ind_y = ind_y

        self.printer._operation('Running %s stationary solver (%i x %i mtx)' % \
            (self.options['solver'], self.mtx.shape[0], self.mtx.shape[1]))
        self.timer._start('stationary')

        for ind in range(self.options['ilimit']):
            self.iterate(rhs, sol)
            self._print_sol(sol)
        self.printer()

        self.timer._stop('stationary')
        self.printer._done_time(self.timer['stationary'])

        return sol


class MultigridSolver(LinearSolver):

    def _declare_options(self):
        self.options.declare('mg_ops')
        self.options.declare('mg_cycles', 0)#11)
        self.options.declare('solver', values=['null', 'gs', 'jacobi', 'krylov'],
                             types=LinearSolver)

    def _initialize(self, mtx, mg_matrices=[], print_status=True):
        self.printer.active = print_status and self.options['print_init']
        self.mtx = mtx

        self.mg_mtx = [mtx]
        self.mg_sol = [np.zeros(self.mtx.shape[0])]
        self.mg_rhs = [np.zeros(self.mtx.shape[0])]

        if self.options['solver'] == 'null':
            solver = NullSolver()
        elif self.options['solver'] == 'gs' or self.options['solver'] == 'jacobi':
            solver = StationarySolver(solver=self.options['solver'], damping=1.0, ilimit=5,
                                      interval=10)
        elif self.options['solver'] == 'krylov':
            solver = KrylovSolver(solver='gmres', pc='nopc', interval=10,
                                  ilimit=101, atol=1e-15)
        elif isinstance(self.options['solver'], LinearSolver):
            solver = self.options['solver']

        mg_solver = solver._clone()
        mg_solver._initialize(mtx, mg_matrices=mg_matrices, print_status=print_status)
        self.mg_solvers = [mg_solver]

        for ind, mg_op in enumerate(self.options['mg_ops']):
            mg_mtx = mg_op.T.dot(self.mg_mtx[-1]).dot(mg_op).tocsc()
            mg_sol = mg_op.T.dot(self.mg_sol[-1])
            mg_rhs = mg_op.T.dot(self.mg_rhs[-1])

            mg_solver = solver._clone()
            mg_solver._initialize(mg_mtx, mg_matrices=mg_matrices, print_status=print_status)

            self.mg_mtx.append(mg_mtx)
            self.mg_sol.append(mg_sol)
            self.mg_rhs.append(mg_rhs)
            self.mg_solvers.append(mg_solver)

        mg_mtx = self.mg_mtx[-1]
        mg_solver = DirectSolver()
        mg_solver._initialize(mg_mtx, mg_matrices=mg_matrices, print_status=print_status)
        self.mg_solvers[-1] = mg_solver

    def _restrict(self, ind_level):
        mg_op = self.options['mg_ops'][ind_level]
        mtx = self.mg_mtx[ind_level]
        sol = self.mg_sol[ind_level]
        rhs = self.mg_rhs[ind_level]

        res = rhs - mtx.dot(sol)
        res_coarse = mg_op.T.dot(res)
        self.mg_rhs[ind_level + 1][:] = res_coarse

    def _smooth_and_restrict(self, ind_level, ind_cycle, ind_y, print_status):
        mg_op = self.options['mg_ops'][ind_level]
        mtx = self.mg_mtx[ind_level]
        sol = self.mg_sol[ind_level]
        rhs = self.mg_rhs[ind_level]
        solver = self.mg_solvers[ind_level]
        solver.print_info = 'MG iter %i level %i' % (ind_cycle, ind_level)

        solver._solve(rhs, sol, ind_y, print_status)

        res = rhs - mtx.dot(sol)
        res_coarse = mg_op.T.dot(res)
        self.mg_rhs[ind_level + 1][:] = res_coarse

    def _coarse_solve(self, ind_cycle, ind_y, print_status):
        sol = self.mg_sol[-1]
        rhs = self.mg_rhs[-1]
        solver = self.mg_solvers[-1]
        solver.print_info = 'MG iter %i level %i' % (ind_cycle, len(self.options['mg_ops']))
        solver._solve(rhs, sol, ind_y, print_status)

    def _smooth_and_interpolate(self, ind_level, ind_cycle, ind_y, print_status):
        mg_op = self.options['mg_ops'][ind_level]
        mtx = self.mg_mtx[ind_level]
        sol = self.mg_sol[ind_level]
        rhs = self.mg_rhs[ind_level]
        solver = self.mg_solvers[ind_level]
        solver.print_info = 'MG iter %i level %i' % (ind_cycle, ind_level)

        sol_coarse = self.mg_sol[ind_level + 1]
        sol += mg_op.dot(sol_coarse)

        solver._solve(rhs, sol, ind_y, print_status)

    def _solve(self, rhs, sol=None, ind_y=0, print_status=True):
        self.printer.active = print_status and self.options['print_solve']
        self.rhs = rhs

        if sol is None:
            sol = np.array(rhs)

        orig_sol = sol

        self.counter = 0
        self.ind_y = ind_y
        self.rhs = rhs

        self.mg_rhs[0][:] = rhs

        for ind_level in range(len(self.options['mg_ops'])):
            self._restrict(ind_level)

        self._coarse_solve(-1, ind_y, print_status)

        for ind_level in range(len(self.options['mg_ops']) - 1, -1, -1):
            self._smooth_and_interpolate(ind_level, -1, ind_y, print_status)

        for ind_cycle in range(self.options['mg_cycles']):

            for ind_level in range(len(self.options['mg_ops'])):
                self._smooth_and_restrict(ind_level, ind_cycle, ind_y, print_status)

            self._coarse_solve(ind_cycle, ind_y, print_status)

            for ind_level in range(len(self.options['mg_ops']) - 1, -1, -1):
                self._smooth_and_interpolate(ind_level, ind_cycle, ind_y, print_status)

        self.printer()

        orig_sol[:] = self.mg_sol[0]
        return orig_sol
