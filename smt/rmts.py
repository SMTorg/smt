"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
"""
from __future__ import division

import numpy as np
import scipy.sparse
from six.moves import range

import RMTSlib

from smt.utils.linear_solvers import get_solver
from smt.utils.line_search import get_line_search_class
from smt.utils.caching import _caching_checksum_sm, _caching_load, _caching_save
from smt.utils.sparse import assemble_sparse_mtx
from smt.sm import SM


class RMTS(SM):
    """
    Regularized Minimal-energy Tensor-product Spline (RMTS) interpolant.

    RMTS divides the n-dimensional space using n-dimensional box elements.
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

    def _set_default_options(self):
        sm_options = {
            'name': 'RMTS', # Regularized Minimal-energy Tensor-product Spline
            'num_elem': [],  # int ndarray[nx]: num. of elements in each dimension
            'xlimits': [],    # flt ndarray[nx, 2]: lower/upper bounds in each dimension
            'smoothness': [], # flt ndarray[nx]: smoothness parameter in each dimension
            'reg_dv': 1e-10, # regularization coeff. for dv block
            'reg_cons': 1e-10, # negative of reg. coeff. for Lagrange mult. block
            'mode': 'approx', # 'approx' or 'exact' form of linear system ()
            'extrapolate': False, # perform linear extrapolation for external eval points
            'approx_norm': 4, # order of norm in least-squares approximation term
            'solver': 'krylov',    # Linear solver: 'gmres' or 'cg'
            'max_nln_iter': 0, # number of nonlinear iterations
            'line_search': 'backtracking', # line search algorithm
            'mg_factors': [], # Multigrid level
            'save_solution': False,  # Whether to save linear system solution
            'max_print_depth': 100, # Maximum depth (level of nesting) to print
        }
        printf_options = {
            'global': True,     # Overriding option to print output
            'time_eval': True,  # Print evaluation times
            'time_train': True, # Print assembly and solution time summary
            'problem': True,    # Print problem information
            'solver': False,     # Print convergence progress (i.e., residual norms)
        }

        self.sm_options = sm_options
        self.printf_options = printf_options

    def _compute_uniq2coeff(self, nx, num_elem_list, num_elem, num_term, num_uniq):
        # This computes an num['term'] x num['term'] matrix called coeff2nodal.
        # Multiplying this matrix with the list of coefficients for an element
        # yields the list of function and derivative values at the element nodes.
        # We need the inverse, but the matrix size is small enough to invert since
        # RMTS is normally only used for 1 <= nx <= 4 in most cases.
        elem_coeff2nodal = RMTSlib.compute_coeff2nodal(nx, num_term)
        elem_nodal2coeff = np.linalg.inv(elem_coeff2nodal)

        # This computes a num_coeff_elem x num_coeff_uniq permutation matrix called
        # uniq2elem. This sparse matrix maps the unique list of nodal function and
        # derivative values to the same function and derivative values, but ordered
        # by element, with repetition.
        nnz = num_elem * num_term
        num_coeff_elem = num_term * num_elem
        num_coeff_uniq = num_uniq * 2 ** nx
        data, rows, cols = RMTSlib.compute_uniq2elem(nnz, nx, num_elem_list)
        full_uniq2elem = scipy.sparse.csc_matrix((data, (rows, cols)),
            shape=(num_coeff_elem, num_coeff_uniq))

        # This computes the matrix full_uniq2coeff, which maps the unique
        # degrees of freedom to the list of coefficients ordered by element.
        nnz = num_term ** 2 * num_elem
        num_coeff = num_term * num_elem
        data, rows, cols = RMTSlib.compute_full_from_block(
            nnz, num_term, num_elem, elem_nodal2coeff)
        full_nodal2coeff = scipy.sparse.csc_matrix((data, (rows, cols)),
            shape=(num_coeff, num_coeff))

        full_uniq2coeff = full_nodal2coeff * full_uniq2elem

        return full_uniq2coeff

    def _compute_local_hess(self):
        # This computes the positive-definite, symmetric matrix yields the energy
        # for an element when pre- and post-multiplied by a vector of function and
        # derivative values for the element. This matrix applies to all elements.
        num = self.num
        sm_options = self.sm_options

        elem_hess = np.zeros((num['term'], num['term']))
        for kx in range(num['x']):
            elem_sec_deriv = RMTSlib.compute_sec_deriv(kx+1, num['term'], num['x'],
                num['elem_list'], sm_options['xlimits'])
            elem_hess += elem_sec_deriv.T.dot(elem_sec_deriv) * \
                sm_options['smoothness'][kx]

        return elem_hess

    def _compute_global_hess(self, elem_hess, full_uniq2coeff):
        # This takes the dense elem_hess matrix and stamps out num['elem'] copies
        # of it to form the full sparse matrix with all the elements included.
        num = self.num
        sm_options = self.sm_options

        nnz = num['term'] ** 2 * num['elem']
        num_coeff = num['term'] * num['elem']
        data, rows, cols = RMTSlib.compute_full_from_block(
            nnz, num['term'], num['elem'], elem_hess)
        full_hess = scipy.sparse.csc_matrix((data, (rows, cols)),
            shape=(num_coeff, num_coeff))
        full_hess = full_uniq2coeff.T * full_hess * full_uniq2coeff

        num_coeff_uniq = num['uniq'] * 2 ** num['x']
        diag = sm_options['reg_dv'] * np.ones(num_coeff_uniq)
        arange = np.arange(num_coeff_uniq)
        reg_dv = scipy.sparse.csc_matrix((diag, (arange, arange)))

        full_hess += reg_dv

        return full_hess

    def _compute_approx_terms(self, full_uniq2coeff):
        # This adds the training points, either using a least-squares approach
        # or with exact contraints and Lagrange multipliers.
        # In both approaches, we loop over kx: 0 is for values and kx>0 represents
        # the 1-based index of the derivative given by the training point data.
        num = self.num
        sm_options = self.sm_options

        reg_cons_dict = {}
        full_jac_dict = {}
        for kx in self.training_pts['exact']:
            xt, yt = self.training_pts['exact'][kx]

            nt = xt.shape[0]
            nnz = nt * num['term']
            num_coeff = num['term'] * num['elem']
            data, rows, cols = RMTSlib.compute_jac(kx, 0, nnz, num['x'], nt,
                num['elem_list'], sm_options['xlimits'], xt)
            full_jac = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(nt, num_coeff))
            full_jac = full_jac * full_uniq2coeff

            full_jac_dict[kx] = full_jac

            if sm_options['mode'] == 'exact':
                diag = -sm_options['reg_cons'] * np.ones(nt)
                arange = np.arange(nt)
                reg_cons_dict[kx] = scipy.sparse.csc_matrix((diag, (arange, arange)))

        return full_jac_dict, reg_cons_dict

    def _compute_single_mg_matrix(self, elem_lists_2, elem_lists_1):
        num = self.num
        sm_options = self.sm_options

        mg_full_uniq2coeff = self._compute_uniq2coeff(num['x'], elem_lists_1,
            np.prod(elem_lists_1), num['term'], np.prod(elem_lists_1 + 1))

        ne = np.prod(elem_lists_2 + 1) * 2 ** num['x']
        nnz = ne * num['term']
        num_coeff = num['term'] * np.prod(elem_lists_1)
        data, rows, cols = RMTSlib.compute_jac_interp(
            nnz, num['x'], elem_lists_1, elem_lists_2 + 1, sm_options['xlimits'])
        mg_jac = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(ne, num_coeff))
        mg_matrix = mg_jac * mg_full_uniq2coeff

        if sm_options['mode'] == 'exact':
            num_lagr = np.sum(block_sizes[1:])
            mg_matrix = scipy.sparse.bmat([
                [mg_matrix, None],
                [None, scipy.sparse.identity(num_lagr)]
            ], format='csc')

        return mg_matrix

    def _compute_mg_matrices(self):
        num = self.num
        sm_options = self.sm_options

        power_two = 2 ** len(sm_options['mg_factors'])
        for kx in range(num['x']):
            assert num['elem_list'][kx] % power_two == 0, 'Invalid multigrid level'

        elem_lists = [num['elem_list']]
        for ind_mg, mg_factor in enumerate(sm_options['mg_factors']):
            elem_lists.append(elem_lists[-1] / mg_factor)

            nrows = np.prod(elem_lists[-2] + 1) * 2 ** num['x']
            ncols = np.prod(elem_lists[-1] + 1) * 2 ** num['x']
            string = 'Assembling multigrid op %i (%i x %i mtx)' % (ind_mg, nrows, ncols)
            with self.printer._timed_context('Assembling multigrid op %i (%i x %i mtx)'
                                             % (ind_mg, nrows, ncols)):
                mg_matrix = self._compute_single_mg_matrix(elem_lists[-2], elem_lists[-1])

            mg_matrices.append(mg_matrix)

        return mg_matrices

    def _opt_func(self, sol, p, full_hess, full_jac_dict, yt_dict):
        c = 0.5 / self.sm_options['reg_cons'] / self.num['t']

        func = 0.5 * np.dot(sol, full_hess * sol)
        for kx in self.training_pts['exact']:
            full_jac = full_jac_dict[kx]
            yt = yt_dict[kx]
            func += c * np.sum((full_jac * sol - yt) ** p)

        return func

    def _opt_grad(self, sol, p, full_hess, full_jac_dict, yt_dict):
        c = 0.5 / self.sm_options['reg_cons'] / self.num['t']

        grad = full_hess * sol
        for kx in self.training_pts['exact']:
            full_jac = full_jac_dict[kx]
            yt = yt_dict[kx]
            grad += c * full_jac.T * p * (full_jac * sol - yt) ** (p - 1)

        return grad

    def _opt_hess(self, sol, p, full_hess, full_jac_dict, yt_dict):
        c = 0.5 / self.sm_options['reg_cons'] / self.num['t']

        hess = scipy.sparse.csc_matrix(full_hess)
        for kx in self.training_pts['exact']:
            full_jac = full_jac_dict[kx]
            yt = yt_dict[kx]

            diag_vec = p * (p - 1) * (full_jac * sol - yt) ** (p - 2)
            diag_mtx = scipy.sparse.diags(diag_vec, format='csc')
            hess += c * full_jac.T * diag_mtx * full_jac

        return hess

    def _opt_hess_2(self, full_hess, full_jac_dict):
        c = 0.5 / self.sm_options['reg_cons'] / self.num['t']
        p = 2

        hess = scipy.sparse.csc_matrix(full_hess)
        for kx in self.training_pts['exact']:
            full_jac = full_jac_dict[kx]
            hess += c * p * (p - 1) * full_jac.T * full_jac

        return hess

    def _opt_norm(self, sol, p, full_hess, full_jac_dict, yt_dict):
        grad = self._opt_grad(sol, p, full_hess, full_jac_dict, yt_dict)
        return np.linalg.norm(grad)

    def _get_yt_dict(self, ind_y):
        yt_dict = {}
        for kx in self.training_pts['exact']:
            xt, yt = self.training_pts['exact'][kx]
            yt_dict[kx] = yt[:, ind_y]
        return yt_dict

    def _fit(self):
        """
        Train the model
        """
        sm_options = self.sm_options

        num = {}
        # number of inputs and outputs
        num['x'] = self.training_pts['exact'][0][0].shape[1]
        num['y'] = self.training_pts['exact'][0][1].shape[1]
        # number of elements
        num['elem_list'] = np.array(sm_options['num_elem'], int)
        num['elem'] = np.prod(num['elem_list'])
        # number of terms/coefficients per element
        num['term_list'] = 4 * np.ones(num['x'], int)
        num['term'] = np.prod(num['term_list'])
        # number of nodes
        num['uniq_list'] = num['elem_list'] + 1
        num['uniq'] = np.prod(num['uniq_list'])
        # total number of training points (function values and derivatives)
        num['t'] = 0
        for kx in self.training_pts['exact']:
            num['t'] += self.training_pts['exact'][kx][0].shape[0]

        if len(sm_options['smoothness']) == 0:
            sm_options['smoothness'] = [1.0] * num['x']

        self.num = num

        self.printer.max_print_depth = sm_options['max_print_depth']

        with self.printer._timed_context('Pre-computing matrices'):

            with self.printer._timed_context('Computing uniq2coeff'):
                full_uniq2coeff = self._compute_uniq2coeff(
                    num['x'], num['elem_list'], num['elem'], num['term'], num['uniq'])

            with self.printer._timed_context('Computing local energy terms'):
                elem_hess = self._compute_local_hess()

            with self.printer._timed_context('Computing global energy terms'):
                full_hess = self._compute_global_hess(elem_hess, full_uniq2coeff)

            with self.printer._timed_context('Computing approximation terms'):
                full_jac_dict, reg_cons_dict = self._compute_approx_terms(full_uniq2coeff)

            if sm_options['solver'] == 'mg':
                mg_matrices = self._compute_mg_matrices()
            else:
                mg_matrices = []

        block_names = ['dv']
        block_sizes = [num['uniq'] * 2 ** num['x']]
        if sm_options['mode'] == 'exact':
            block_names += ['con_%s'%kx for kx in self.training_pts['exact']]
            block_sizes += [self.training_pts['exact'][kx][0].shape[0]
                            for kx in self.training_pts['exact']]

        with self.printer._timed_context('Solving for degrees of freedom'):

            solver = get_solver(sm_options['solver'])
            ls_class = get_line_search_class(sm_options['line_search'])

            total_size = int(np.sum(block_sizes))
            rhs = np.zeros((total_size, num['y']))
            sol = np.zeros((total_size, num['y']))
            d_sol = np.zeros((total_size, num['y']))

            with self.printer._timed_context('Solving initial linear problem'):

                with self.printer._timed_context('Assembling linear system'):
                    if sm_options['mode'] == 'approx':
                        mtx = self._opt_hess_2(full_hess, full_jac_dict)
                        for ind_y in range(num['y']):
                            yt_dict = self._get_yt_dict(ind_y)
                            rhs[:, ind_y] = -self._opt_grad(sol[:, ind_y], 2, full_hess,
                                                            full_jac_dict, yt_dict)
                    elif sm_options['mode'] == 'exact':
                        sub_mtx_dict = {}
                        sub_rhs_dict = {}
                        sub_mtx_dict['dv', 'dv'] = scipy.sparse.csc_matrix(full_hess)
                        sub_rhs_dict['dv'] = -full_hess * sol
                        for kx in self.training_pts['exact']:
                            full_jac = full_jac_dict[kx]
                            xt, yt = self.training_pts['exact'][kx]

                            reg_cons = reg_cons_dict[kx]
                            sub_mtx_dict['con_%s'%kx, 'dv'] = full_jac
                            sub_mtx_dict['dv', 'con_%s'%kx] = full_jac.T
                            sub_mtx_dict['con_%s'%kx, 'con_%s'%kx] = reg_cons
                            sub_rhs_dict['con_%s'%kx] = yt

                        mtx, rhs = assemble_sparse_mtx(
                            block_names, block_sizes, sub_mtx_dict, sub_rhs_dict)

                with self.printer._timed_context('Initializing linear solver'):
                    solver._initialize(mtx, self.printer, mg_matrices=mg_matrices)

                for ind_y in range(rhs.shape[1]):
                    with self.printer._timed_context('Solving linear system (col. %i)' % ind_y):
                        solver._solve(rhs[:, ind_y], sol[:, ind_y], ind_y=ind_y)

            p = self.sm_options['approx_norm']
            for ind_y in range(rhs.shape[1]):

                with self.printer._timed_context('Solving nonlinear problem (col. %i)' % ind_y):

                    yt_dict = self._get_yt_dict(ind_y)

                    if sm_options['max_nln_iter'] > 0:
                        norm = self._opt_norm(sol[:, ind_y], p, full_hess, full_jac_dict, yt_dict)
                        fval = self._opt_func(sol[:, ind_y], p, full_hess, full_jac_dict, yt_dict)
                        self.printer(
                            'Nonlinear (itn, iy, grad. norm, func.) : %3i %3i %15.9e %15.9e'
                            % (0, ind_y, norm, fval))

                    for nln_iter in range(sm_options['max_nln_iter']):
                        with self.printer._timed_context():
                            with self.printer._timed_context('Assembling linear system'):
                                mtx = self._opt_hess(sol[:, ind_y], p, full_hess,
                                                     full_jac_dict, yt_dict)
                                rhs[:, ind_y] = -self._opt_grad(sol[:, ind_y], p, full_hess,
                                                                full_jac_dict, yt_dict)

                            with self.printer._timed_context('Initializing linear solver'):
                                solver._initialize(mtx, self.printer, mg_matrices=mg_matrices)

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

                        if norm < 1e-3:
                            break

        self.sol = full_uniq2coeff * sol[:num['uniq'] * 2 ** num['x'], :]

    def fit(self):
        """
        Train the model
        """
        checksum = _caching_checksum_sm(self)

        filename = '%s.sm' % self.sm_options['name']
        success, data = _caching_load(filename, checksum)
        if not success or not self.sm_options['save_solution']:
            self._fit()
            data = {'sol': self.sol, 'num': self.num}
            _caching_save(filename, checksum, data)
        else:
            self.sol = data['sol']
            self.num = data['num']

    def evaluate(self, x):
        """
        This function evaluates the surrogate model at x.

        Parameters
        ----------
        x: np.ndarray[n_eval,dim]
            - An array giving the point(s) at which the prediction(s) should be
              made.

        Returns
        -------
        y : np.ndarray[n_eval,1]
            - An array with the output values at x.
        """
        kx = 0

        ne = x.shape[0]

        num = self.num
        sm_options = self.sm_options

        # Compute sparse Jacobian matrix.
        nnz = ne * num['term']
        num_coeff = num['term'] * num['elem']
        data, rows, cols = RMTSlib.compute_jac(kx, 0, nnz, num['x'], ne,
            num['elem_list'], sm_options['xlimits'], x)

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
        if sm_options['extrapolate']:

            # First we evaluate the vector pointing to each evaluation points
            # from the nearest point on the domain, in a matrix called dx.
            # If the ith evaluation point is not external, dx[i, :] = 0.
            ndx = ne * num['term']
            dx = RMTS.compute_ext_dist(num['x'], ne, ndx, sm_options['xlimits'], x)
            isexternal = np.array(np.array(dx, bool), float)

            for ix in range(num['x']):
                # Now we compute the first order term where we have a
                # derivative times (x_k - b_k) or (x_k - a_k).
                nnz = ne * num['term']
                data_tmp, rows, cols = RMTSlib.compute_jac(kx, ix+1, nnz, num['x'], ne,
                    num['elem_list'], sm_options['xlimits'], x)
                data_tmp *= dx[:, ix]

                # If we are evaluating a derivative (with index kx),
                # we zero the first order terms for which dx_k = 0.
                if kx != 0:
                    data_tmp *= 1 - isexternal[:, kx-1]

                data += data_tmp

        mtx = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(ne, num_coeff))

        return mtx.dot(self.sol)
