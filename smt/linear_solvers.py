"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
"""

from __future__ import print_function
import numpy as np
import scipy.sparse.linalg
import pyamg.krylov
import six
from six.moves import range

from smt.utils import OptionsDictionary, Timer


class LinearSolver(object):

    def __init__(self, mtx, print_global, print_conv, **kwargs): #print_conv):
        self.mtx = mtx

        self.callback = None
        self.counter = 0
        self.ind_y = 0
        self.rhs = None
        self.norm0 = 0
        self.print_global = print_global
        self.print_conv = print_conv
        self.print_info = ''

        self.timer = Timer(print_global)
        self.options = OptionsDictionary(kwargs)
        self.options.add('interval', 1, type_=int)
        self._initialize()

    def _print(self, string=''):
        if self.print_global:
            print(string)

    def _initialize(self):
        pass

    def _print_res(self, res):
        if not self.print_global or not self.print_conv:
            self.counter += 1
            return

        norm = res
        if self.counter == 0:
            self.norm0 = norm

        if self.counter % self.options['interval'] == 0:
            print('   %s (%i x %i mtx), output %-3i : %3i  %15.9e  %15.9e' %
                (self.__class__.__name__ + str(self.print_info),
                 self.mtx.shape[0], self.mtx.shape[1],
                 self.ind_y, self.counter, norm, norm / self.norm0))
        self.counter += 1

    def _print_sol(self, sol):
        if not self.print_global or not self.print_conv:
            self.counter += 1
            return

        res = self.mtx.dot(sol) - self.rhs
        norm = np.linalg.norm(res)
        if self.counter == 0:
            self.norm0 = norm

        if self.counter % self.options['interval'] == 0:
            print('   %s (%i x %i mtx), output %-3i : %3i  %15.9e  %15.9e' %
                (self.__class__.__name__ + str(self.print_info),
                 self.mtx.shape[0], self.mtx.shape[1],
                 self.ind_y, self.counter, norm, norm / self.norm0))
        self.counter += 1


class NullSolver(LinearSolver):

    def solve(self, rhs, sol=None, ind_y=0):
        pass


class DirectSolver(LinearSolver):

    def _initialize(self):
        self.options.add('alg', 'lu', values=['lu', 'ilu'])

        self.timer._start('direct', 'Performing %s fact. on (%i x %i) mtx' % \
            (self.options['alg'], self.mtx.shape[0], self.mtx.shape[1]))
        if self.options['alg'] == 'lu':
            self.fact = scipy.sparse.linalg.splu(self.mtx)
        elif self.options['alg'] == 'ilu':
            self.fact = scipy.sparse.linalg.spilu(
                self.mtx, drop_rule='interp',
                drop_tol=1e-3, #1e-3,
                fill_factor=2, #1,
            )
        self.timer._stop('direct', print_done=True)

    def solve(self, rhs, sol=None, ind_y=0):
        if sol is None:
            return self.fact.solve(rhs)
        else:
            sol[:] = self.fact.solve(rhs)


class KrylovSolver(LinearSolver):

    def _initialize(self):
        self.options.add('pc', 'lu', values=['ilu', 'lu', 'nopc', 'gs', 'custom'])
        self.options.add('pc_solver', object)
        self.options.add('solver', 'cg', values=['cg', 'bicgstab', 'gmres', 'fgmres'])
        self.options.add('ilimit', 100, type_=int)
        self.options.add('atol', 1e-15, type_=(int, float))
        self.options.add('rtol', 1e-15, type_=(int, float))

        if self.options['pc'] == 'lu' or self.options['pc'] == 'ilu':
            solver = DirectSolver(self.mtx, self.print_global, False, alg=self.options['pc'])
            self.pc_op = scipy.sparse.linalg.LinearOperator(self.mtx.shape, matvec=solver.solve)
        elif self.options['pc'] == 'nopc':
            self.pc_op = None
        elif self.options['pc'] == 'gs':
            solver = StationarySolver(self.mtx, self.print_global, False,
                                      solver='gs', damping=1.0, ilimit=1)
            self.pc_op = scipy.sparse.linalg.LinearOperator(self.mtx.shape, matvec=solver.solve)
        elif self.options['pc'] == 'custom':
            solver = self.options['pc_solver']
            self.pc_op = scipy.sparse.linalg.LinearOperator(self.mtx.shape, matvec=solver.solve)

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
                                  'restart': min(self.options['ilimit'], self.mtx.shape[0])}
        elif self.options['solver'] == 'fgmres':
            self.solver = pyamg.krylov.fgmres
            self.callback = self._print_sol
            self.solver_kwargs = {'tol': self.options['rtol'],
                                  #'maxiter': self.options['ilimit'],
                                  #'restrt': min(self.options['ilimit'], self.mtx.shape[0]),
                                  'maxiter': 1, 'restrt': 300,
                                  }

    def solve(self, rhs, sol=None, ind_y=0):
        if sol is None:
            sol = np.array(rhs)

        orig_rhs = np.array(rhs)

        self.counter = 0
        self.ind_y = ind_y
        self.rhs = rhs

        self._print_sol(sol)
        tmp, info = self.solver(
            self.mtx, rhs, x0=sol, M=self.pc_op,
            callback=self.callback,
            **self.solver_kwargs
        )
        # self._print_sol(tmp)

        if self.print_conv:
            self._print()

        sol[:] = tmp

        return sol


class StationarySolver(LinearSolver):

    def _initialize(self):
        self.options.add('solver', 'gs', values=['gs', 'jacobi'])
        self.options.add('damping', 1.0, type_=(int, float))
        self.options.add('ilimit', 10, type_=int)

        self.timer._start('stationary', 'Initializing %s solver on (%i x %i) mtx' % \
                    (self.options['solver'], self.mtx.shape[0], self.mtx.shape[1]))

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

        self.timer._stop('stationary', print_done=True)

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

    def solve(self, rhs, sol=None, ind_y=0):
        if sol is None:
            sol = np.array(rhs)

        self.counter = 0
        self.ind_y = ind_y
        self.rhs = rhs

        for ind in range(self.options['ilimit']):
            self.iterate(rhs, sol)
            self._print_sol(sol)

        if self.print_conv:
            self._print()

        return sol


class MultigridSolver(LinearSolver):

    def _initialize(self):
        self.options.add('mg_ops')
        self.options.add('mg_cycles', 0)

        self.mg_mtx = [self.mtx]
        self.mg_sol = [np.zeros(self.mtx.shape[0])]
        self.mg_rhs = [np.zeros(self.mtx.shape[0])]
        if 1:
            self.mg_solvers = [NullSolver(self.mtx, self.print_global, self.print_conv)]
        if 0:
            self.mg_solvers = [StationarySolver(self.mtx, self.print_global, self.print_conv,
                                                solver='jacobi', damping=1.0, ilimit=0, #11, #31,
                                                interval=10,
                                                )]
        if 0:
            self.mg_solvers = [KrylovSolver(self.mtx, self.print_global, self.print_conv,
                                            solver='gmres',
                                            #pc=self.mg_solvers[0],
                                            pc='nopc',
                                            interval=10,
                                            ilimit=101, atol=1e-15)]

        for ind, mg_op in enumerate(self.options['mg_ops']):
            mtx = mg_op.T.dot(self.mg_mtx[-1]).dot(mg_op).tocsc()
            sol = mg_op.T.dot(self.mg_sol[-1])
            rhs = mg_op.T.dot(self.mg_rhs[-1])
            if 1:
                solver = NullSolver(mtx, self.print_global, False)
            if 0:
                solver = StationarySolver(mtx, self.print_global, False, #self.print_conv,
                                          solver='jacobi', damping=1.0, ilimit=1, #11, #31,
                                          interval=10,
                                          )
            if 0:
                solver = KrylovSolver(mtx, self.print_global, False, #self.print_conv,
                                      solver='gmres',
                                      #pc=solver,
                                      pc='nopc',
                                      interval=10,
                                      ilimit=101, atol=1e-15)

            self.mg_mtx.append(mtx)
            self.mg_sol.append(sol)
            self.mg_rhs.append(rhs)
            self.mg_solvers.append(solver)

        # self.mg_solvers[-1] = \
        #     KrylovSolver(self.mg_mtx[-1], self.print_global, self.print_conv,
        #                  solver='fgmres',
        #                  pc='nopc',
        #                  ilimit=11,
        #                  atol=1e-15,
        #                  interval=10,
        #                  )
        # self.mg_solvers[-1] = StationarySolver(mtx, True, #self.print_conv,
        #                                        solver='jacobi', damping=0.1, ilimit=31,
        #                                        print_info='mg:coarse')
        self.mg_solvers[-1] = DirectSolver(self.mg_mtx[-1], self.print_global, self.print_conv)

    def _restrict(self, ind_level):
        mg_op = self.options['mg_ops'][ind_level]
        mtx = self.mg_mtx[ind_level]
        sol = self.mg_sol[ind_level]
        rhs = self.mg_rhs[ind_level]

        res = rhs - mtx.dot(sol)
        res_coarse = mg_op.T.dot(res)
        self.mg_rhs[ind_level + 1][:] = res_coarse

    def _smooth_and_restrict(self, ind_level, ind_cycle, ind_y):
        mg_op = self.options['mg_ops'][ind_level]
        mtx = self.mg_mtx[ind_level]
        sol = self.mg_sol[ind_level]
        rhs = self.mg_rhs[ind_level]
        solver = self.mg_solvers[ind_level]
        solver.print_info = 'MG iter %i level %i' % (ind_cycle, ind_level)

        solver.solve(rhs, sol, ind_y)

        res = rhs - mtx.dot(sol)
        res_coarse = mg_op.T.dot(res)
        self.mg_rhs[ind_level + 1][:] = res_coarse

    def _coarse_solve(self, ind_cycle, ind_y):
        sol = self.mg_sol[-1]
        rhs = self.mg_rhs[-1]
        solver = self.mg_solvers[-1]
        solver.print_info = 'MG iter %i level %i' % (ind_cycle, len(self.options['mg_ops']))
        solver.solve(rhs, sol, ind_y)

    def _smooth_and_interpolate(self, ind_level, ind_cycle, ind_y):
        mg_op = self.options['mg_ops'][ind_level]
        mtx = self.mg_mtx[ind_level]
        sol = self.mg_sol[ind_level]
        rhs = self.mg_rhs[ind_level]
        solver = self.mg_solvers[ind_level]
        solver.print_info = 'MG iter %i level %i' % (ind_cycle, ind_level)

        sol_coarse = self.mg_sol[ind_level + 1]
        sol += mg_op.dot(sol_coarse)

        solver.solve(rhs, sol, ind_y)

    def solve(self, rhs, sol=None, ind_y=0):
        if sol is None:
            sol = np.array(rhs)
        orig_sol = sol

        self.counter = 0
        self.ind_y = ind_y
        self.rhs = rhs

        self.mg_rhs[0][:] = rhs

        for ind_level in range(len(self.options['mg_ops'])):
            self._restrict(ind_level)

        self._coarse_solve(-1, ind_y)

        for ind_level in range(len(self.options['mg_ops']) - 1, -1, -1):
            self._smooth_and_interpolate(ind_level, -1, ind_y)

        for ind_cycle in range(self.options['mg_cycles']):

            for ind_level in range(len(self.options['mg_ops'])):
                self._smooth_and_restrict(ind_level, ind_cycle, ind_y)

            self._coarse_solve(ind_cycle, ind_y)

            for ind_level in range(len(self.options['mg_ops']) - 1, -1, -1):
                self._smooth_and_interpolate(ind_level, ind_cycle, ind_y)

        if self.print_conv:
            self._print()

        '''
        sol = self.mg_sol[0]
        rhs = self.mg_rhs[0]
        # solver = StationarySolver(self.mtx, False, #self.print_conv,
        #                           solver='gs', damping=1.0, ilimit=11)
        solver = KrylovSolver(self.mtx, self.print_conv,
                              solver='gmres', pc='nopc', #self.mg_solvers[0],
                              ilimit=101, atol=1e-15)
        solver.interval = 10
        solver.solve(rhs, sol, ind_y)
        '''

        orig_sol[:] = self.mg_sol[0]
        return orig_sol
