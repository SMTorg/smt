"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
"""
from __future__ import division

import numpy as np
import scipy.sparse
from six.moves import range
from numbers import Integral

from smt.utils.linear_solvers import get_solver, LinearSolver, VALID_SOLVERS
from smt.utils.line_search import get_line_search_class, LineSearch, VALID_LINE_SEARCHES
from smt.utils.caching import cached_operation
from smt.methods.sm import SM

from smt.methods import RMTSlib


class RMTS(SM):
    """
    Regularized Minimal-energy Tensor-product Spline interpolant base class for RMTC and RMTB.
    """

    def _declare_options(self):
        super(RMTS, self)._declare_options()
        declare = self.options.declare

        declare('xlimits', types=np.ndarray,
                desc='Lower/upper bounds in each dimension - ndarray [nx, 2]')
        declare('smoothness', 1.0, types=(Integral, float, tuple, list, np.ndarray),
                desc='Smoothness parameter in each dimension - length nx. None implies uniform')
        declare('reg_dv', 1e-10, types=(Integral, float),
                desc='Regularization coeff. for system degrees of freedom. ' +
                     'This ensures there is always a unique solution')
        declare('reg_cons', 1e-4, types=(Integral, float),
                desc='Negative of the regularization coeff. of the Lagrange mult. block ' +
                     'The weight of the energy terms (and reg_dv) relative to the approx terms')
        declare('extrapolate', False, types=bool,
                desc='Whether to perform linear extrapolation for external evaluation points')
        declare('min_energy', True, types=bool,
                desc='Whether to perform energy minimization')
        declare('approx_order', 4, types=Integral,
                desc='Exponent in the approximation term')
        declare('mtx_free', False, types=bool,
                desc='Whether to solve the linear system in a matrix-free way')
        declare('solver', 'krylov', values=VALID_SOLVERS, types=LinearSolver,
                desc='Linear solver')
        declare('grad_weight', 0.5, types=(Integral, float),
                desc='Weight on gradient training data')
        declare('nln_max_iter', 5, types=Integral,
                desc='maximum number of nonlinear iterations')
        declare('line_search', 'backtracking', values=VALID_LINE_SEARCHES, types=LineSearch,
                desc='Line search algorithm')
        declare('save_energy_terms', False, types=bool,
                desc='Whether to cache energy terms in the data_dir directory')
        declare('data_dir', None, values=(None,), types=str,
                desc='Directory for loading / saving cached data; None means do not save or load')
        declare('max_print_depth', 5, types=Integral,
                desc='Maximum depth (level of nesting) to print operation descriptions and times')

    def _initialize_hessian(self):
        diag = self.options['reg_dv'] * np.ones(self.num['dof'])
        arange = np.arange(self.num['dof'])
        full_hess = scipy.sparse.csc_matrix((diag, (arange, arange)))
        return full_hess

    def _compute_jac(self, ix1, ix2, x):
        data, rows, cols = self._compute_jac_raw(ix1, ix2, x)
        n = x.shape[0]
        full_jac = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(n, self.num['coeff']))
        if self.full_dof2coeff is not None:
            full_jac = full_jac * self.full_dof2coeff
        return full_jac

    def _compute_approx_terms(self):
        # This computes the approximation terms for the training points.
        # We loop over kx: 0 is for values and kx>0 represents.
        # the 1-based index of the derivative given by the training point data.
        num = self.num
        xlimits = self.options['xlimits']

        full_jac_dict = {}
        for kx in self.training_points[None]:
            xt, yt = self.training_points[None][kx]

            xmin = np.min(xt, axis=0)
            xmax = np.max(xt, axis=0)
            assert np.all(xlimits[:, 0] <= xmin), 'Training points below min for %s' % kx
            assert np.all(xlimits[:, 1] >= xmax), 'Training points above max for %s' % kx

            if kx == 0:
                c = 1.0
            else:
                self.options['grad_weight'] / xlimits.shape[0]

            full_jac = self._compute_jac(kx, 0, xt)
            full_jac_dict[kx] = (full_jac, full_jac.T.tocsc(), c)

        return full_jac_dict

    def _compute_energy_terms(self):
        # This computes the energy terms that are to be minimized.
        # The quadrature points are the centroids of the multi-dimensional elements.
        num = self.num
        xlimits = self.options['xlimits']

        inputs = {}
        inputs['nx'] = xlimits.shape[0]
        inputs['elem_list'] = num['elem_list']
        if self.__class__.__name__ == 'RMTB':
            inputs['num_ctrl_list'] = num['ctrl_list']
            inputs['order_list'] = num['order_list']

        if self.options['save_energy_terms']:
            cache_dir = self.options['data_dir']
        else:
            cache_dir = None
        with cached_operation(inputs, cache_dir) as outputs:
            if outputs:
                sq_mtx = outputs['sq_mtx']
            else:
                n = np.prod(2 * num['elem_list'])
                x = RMTSlib.compute_quadrature_points(n, num['x'], 2 * num['elem_list'], xlimits)

                sq_mtx = [None] * num['x']
                for kx in range(num['x']):
                    mtx = self._compute_jac(kx+1, kx+1, x)
                    sq_mtx[kx] = mtx.T.tocsc() * mtx * (xlimits[kx, 1] - xlimits[kx, 0]) ** 4

                outputs['sq_mtx'] = sq_mtx

        elem_vol = np.prod((xlimits[:, 1] - xlimits[:, 0]) / (2 * num['elem_list']))
        total_vol = np.prod(xlimits[:, 1] - xlimits[:, 0])

        full_hess = scipy.sparse.csc_matrix((num['dof'], num['dof']))
        for kx in range(num['x']):
            full_hess += sq_mtx[kx] * (elem_vol / total_vol * self.options['smoothness'][kx] \
                / (xlimits[kx, 1] - xlimits[kx, 0]) ** 4)

        return full_hess

    def _opt_func(self, sol, p, full_hess, full_jac_dict, yt_dict):
        func = 0.5 * np.dot(sol, full_hess * sol)
        for kx in self.training_points[None]:
            full_jac, full_jac_T, c = full_jac_dict[kx]
            yt = yt_dict[kx]
            func += 0.5 * c * np.sum((full_jac * sol - yt) ** p)

        return func

    def _opt_grad(self, sol, p, full_hess, full_jac_dict, yt_dict):
        grad = full_hess * sol
        for kx in self.training_points[None]:
            full_jac, full_jac_T, c = full_jac_dict[kx]
            yt = yt_dict[kx]
            grad += 0.5 * c * full_jac_T * p * (full_jac * sol - yt) ** (p - 1)

        return grad

    def _opt_hess(self, sol, p, full_hess, full_jac_dict, yt_dict):
        if self.options['mtx_free']:
            return self._opt_hess_op(sol, p, full_hess, full_jac_dict, yt_dict)
        else:
            return self._opt_hess_mtx(sol, p, full_hess, full_jac_dict, yt_dict)

    def _opt_hess_mtx(self, sol, p, full_hess, full_jac_dict, yt_dict):
        hess = scipy.sparse.csc_matrix(full_hess)
        for kx in self.training_points[None]:
            full_jac, full_jac_T, c = full_jac_dict[kx]
            yt = yt_dict[kx]

            diag_vec = p * (p - 1) * (full_jac * sol - yt) ** (p - 2)
            diag_mtx = scipy.sparse.diags(diag_vec, format='csc')
            hess += 0.5 * c * full_jac_T * diag_mtx * full_jac

        return hess

    def _opt_hess_op(self, sol, p, full_hess, full_jac_dict, yt_dict):
        class SpMatrix(object):
            def __init__(self):
                self.shape = full_hess.shape
                self.diag_vec = {}
                for kx in full_jac_dict:
                    full_jac, full_jac_T, c = full_jac_dict[kx]
                    yt = yt_dict[kx]
                    self.diag_vec[kx] = p * (p - 1) * (full_jac * sol - yt) ** (p - 2)
            def dot(self, other):
                vec = full_hess * other
                for kx in full_jac_dict:
                    full_jac, full_jac_T, c = full_jac_dict[kx]
                    vec += 0.5 * c * full_jac_T * (self.diag_vec[kx] * (full_jac * other))
                return vec

        mtx = SpMatrix()
        op = scipy.sparse.linalg.LinearOperator(mtx.shape, matvec=mtx.dot)
        return op

    def _opt_hess_2(self, full_hess, full_jac_dict):
        if self.options['mtx_free']:
            return self._opt_hess_op_2(full_hess, full_jac_dict)
        else:
            return self._opt_hess_mtx_2(full_hess, full_jac_dict)

    def _opt_hess_mtx_2(self, full_hess, full_jac_dict):
        p = 2

        hess = scipy.sparse.csc_matrix(full_hess)
        for kx in self.training_points[None]:
            full_jac, full_jac_T, c = full_jac_dict[kx]
            hess += 0.5 * c * p * (p - 1) * full_jac_T * full_jac

        return hess

    def _opt_hess_op_2(self, full_hess, full_jac_dict):
        p = 2

        class SpMatrix(object):
            def __init__(self):
                self.shape = full_hess.shape
            def dot(self, other):
                vec = full_hess * other
                for kx in full_jac_dict:
                    full_jac, full_jac_T, c = full_jac_dict[kx]
                    vec += 0.5 * c * p * (p - 1) * full_jac_T * (full_jac * other)
                return vec

        mtx = SpMatrix()
        op = scipy.sparse.linalg.LinearOperator(mtx.shape, matvec=mtx.dot)
        return op

    def _opt_norm(self, sol, p, full_hess, full_jac_dict, yt_dict):
        grad = self._opt_grad(sol, p, full_hess, full_jac_dict, yt_dict)
        return np.linalg.norm(grad)

    def _get_yt_dict(self, ind_y):
        yt_dict = {}
        for kx in self.training_points[None]:
            xt, yt = self.training_points[None][kx]
            yt_dict[kx] = yt[:, ind_y]
        return yt_dict

    def _solve(self, full_hess, full_jac_dict):
        num = self.num
        options = self.options

        solver = get_solver(options['solver'])
        ls_class = get_line_search_class(options['line_search'])

        total_size = int(num['dof'])
        rhs = np.zeros((total_size, num['y']))
        sol = np.zeros((total_size, num['y']))
        d_sol = np.zeros((total_size, num['y']))

        with self.printer._timed_context('Solving initial linear problem (n=%i)' % total_size):

            with self.printer._timed_context('Assembling linear system'):
                mtx = self._opt_hess_2(full_hess, full_jac_dict)
                for ind_y in range(num['y']):
                    yt_dict = self._get_yt_dict(ind_y)
                    rhs[:, ind_y] = -self._opt_grad(sol[:, ind_y], 2, full_hess,
                                                    full_jac_dict, yt_dict)

            with self.printer._timed_context('Initializing linear solver'):
                solver._initialize(mtx, self.printer)

            for ind_y in range(rhs.shape[1]):
                with self.printer._timed_context('Solving linear system (col. %i)' % ind_y):
                    solver._solve(rhs[:, ind_y], sol[:, ind_y], ind_y=ind_y)

        p = self.options['approx_order']
        for ind_y in range(rhs.shape[1]):

            with self.printer._timed_context('Solving nonlinear problem (col. %i)' % ind_y):

                yt_dict = self._get_yt_dict(ind_y)

                if options['nln_max_iter'] > 0:
                    norm = self._opt_norm(sol[:, ind_y], p, full_hess, full_jac_dict, yt_dict)
                    fval = self._opt_func(sol[:, ind_y], p, full_hess, full_jac_dict, yt_dict)
                    self.printer(
                        'Nonlinear (itn, iy, grad. norm, func.) : %3i %3i %15.9e %15.9e'
                        % (0, ind_y, norm, fval))

                for nln_iter in range(options['nln_max_iter']):
                    with self.printer._timed_context():
                        with self.printer._timed_context('Assembling linear system'):
                            mtx = self._opt_hess(sol[:, ind_y], p, full_hess,
                                                    full_jac_dict, yt_dict)
                            rhs[:, ind_y] = -self._opt_grad(sol[:, ind_y], p, full_hess,
                                                            full_jac_dict, yt_dict)

                        with self.printer._timed_context('Initializing linear solver'):
                            solver._initialize(mtx, self.printer)

                        with self.printer._timed_context('Solving linear system'):
                            solver._solve(rhs[:, ind_y], d_sol[:, ind_y], ind_y=ind_y)

                        func = lambda x: self._opt_func(x, p, full_hess,
                                                        full_jac_dict, yt_dict)
                        grad = lambda x: self._opt_grad(x, p, full_hess,
                                                        full_jac_dict, yt_dict)

                        ls = ls_class(sol[:, ind_y], d_sol[:, ind_y], func, grad)
                        with self.printer._timed_context('Performing line search'):
                            sol[:, ind_y] = ls(1.0)

                    norm = self._opt_norm(sol[:, ind_y], p, full_hess,
                                          full_jac_dict, yt_dict)
                    fval = self._opt_func(sol[:, ind_y], p, full_hess,
                                          full_jac_dict, yt_dict)
                    self.printer(
                        'Nonlinear (itn, iy, grad. norm, func.) : %3i %3i %15.9e %15.9e'
                        % (nln_iter + 1, ind_y, norm, fval))

                    if norm < 1e-16:
                        break

        return sol

    def _new_train(self):
        """
        Train the model
        """
        with self.printer._timed_context('Pre-computing matrices', 'assembly'):

            with self.printer._timed_context('Computing dof2coeff', 'dof2coeff'):
                self.full_dof2coeff = self._compute_dof2coeff()

            with self.printer._timed_context('Initializing Hessian', 'init_hess'):
                full_hess = self._initialize_hessian()

            if self.options['min_energy']:
                with self.printer._timed_context('Computing energy terms', 'energy'):
                    full_hess += self._compute_energy_terms()

            with self.printer._timed_context('Computing approximation terms', 'approx'):
                full_jac_dict = self._compute_approx_terms()

            full_hess *= self.options['reg_cons']

        with self.printer._timed_context('Solving for degrees of freedom', 'total_solution'):
            self.sol = self._solve(full_hess, full_jac_dict)

        if self.full_dof2coeff is not None:
            self.sol = self.full_dof2coeff * self.sol

    def _train(self):
        """
        Train the model
        """
        self._initialize()

        inputs = {'self': self}
        with cached_operation(inputs, self.options['data_dir']) as outputs:
            if outputs:
                self.sol = outputs['sol']
            else:
                self._new_train()
                outputs['sol'] = self.sol

    def _predict_value(self, x):
        '''
        Evaluates the model at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        y : np.ndarray
            Evaluation point output variable values
        '''
        y = self._predict(x, 0)
        return y

    def _predict_derivative(self, x, kx):
        '''
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
        '''
        y = self._predict(x, kx + 1)
        return y

    def _predict(self, x, kx):
        """
        Evaluate the surrogate model at x.

        Parameters
        ----------
        x : np.ndarray[n_eval,dim]
            An array giving the point(s) at which the prediction(s) should be made.
        kx : int
            0    for the interpolant.
            > 0  for the derivative with respect to the kx^{th} input variable (kx is 1-based).

        Returns
        -------
        y : np.ndarray[n_eval,1]
            - An array with the output values at x.
        """
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
        if options['extrapolate']:

            # First we evaluate the vector pointing to each evaluation points
            # from the nearest point on the domain, in a matrix called dx.
            # If the ith evaluation point is not external, dx[i, :] = 0.
            ndx = n * num['support']
            dx = RMTSlib.compute_ext_dist(num['x'], n, ndx, options['xlimits'], x)
            isexternal = np.array(np.array(dx, bool), float)

            for ix in range(num['x']):
                # Now we compute the first order term where we have a
                # derivative times (x_k - b_k) or (x_k - a_k).
                data_tmp, rows, cols = self._compute_jac_raw(kx, ix+1, x)
                data_tmp *= dx[:, ix]

                # If we are evaluating a derivative (with index kx),
                # we zero the first order terms for which dx_k = 0.
                if kx != 0:
                    data_tmp *= 1 - isexternal[:, kx-1]

                data += data_tmp

        mtx = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(n, num['coeff']))
        y = mtx.dot(self.sol)

        return y
