"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

TODO:
- Address approx vs exact issue
- Generalize to arbitrary number of outputs
"""

from __future__ import division

import numpy as np
import scipy.sparse
import RMTSlib
import smt.utils
import smt.linalg
from sm import SM
from six.moves import range


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
            'reg_dv': 1e-8, # regularization coeff. for dv block
            'reg_cons': 1e-8, # negative of reg. coeff. for Lagrange mult. block
            'mode': 'exact', # 'approx' or 'exact' form of linear system ()
            'extrapolate': False, # perform linear extrapolation for external eval points
            'solver_type': 'krylov',    # Linear solver: 'gmres' or 'cg'
            'solver_krylov': 'gmres',    # Preconditioner: 'ilu', 'lu', or 'nopc'
            'solver_pc': 'lu',    # Preconditioner: 'ilu', 'lu', or 'nopc'
            'solver_stationary': 'gs',    # Preconditioner: 'ilu', 'lu', or 'nopc'
            'solver_damping': 1.0,    # Damping coeff. for Jacobi/GS
            'solver_mg': [], # Multigrid level
            'solver_atol': 1e-15, # Absolute linear system convergence tolerance
            'solver_rtol': 1e-15, # Relative linear system convergence tolerance
            'solver_ilimit': 150, # Linear system iteration limit
            'solver_save': True,  # Whether to save linear system solution
        }
        printf_options = {
            'global': True,     # Overriding option to print output
            'time_eval': True,  # Print evaluation times
            'time_train': True, # Print assembly and solution time summary
            'problem': True,    # Print problem information
            'solver': True,     # Print convergence progress (i.e., residual norms)
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

    def _fit(self):
        """
        Train the model
        """
        sm_options = self.sm_options

        nx = self.training_pts['exact'][0][0].shape[1]
        ny = self.training_pts['exact'][0][1].shape[1]

        num = {}
        # number of elements
        num['elem_list'] = np.array(sm_options['num_elem'], int)
        num['elem'] = np.prod(num['elem_list'])
        # number of terms/coefficients per element
        num['term_list'] = 4 * np.ones(nx, int)
        num['term'] = np.prod(num['term_list'])
        # number of nodes
        num['uniq_list'] = num['elem_list'] + 1
        num['uniq'] = np.prod(num['uniq_list'])

        self.num = num

        sub_mtx_dict = {}
        sub_rhs_dict = {}

        self.timer._start('total_assembly')

        self.timer._start('uniq2coeff', 'Assembling uniq2coeff')
        full_uniq2coeff = self._compute_uniq2coeff(
            nx, num['elem_list'], num['elem'], num['term'], num['uniq'])
        self.timer._stop('uniq2coeff', print_done=True)

        # This computes the positive-definite, symmetric matrix yields the energy
        # for an element when pre- and post-multiplied by a vector of function and
        # derivative values for the element. This matrix applies to all elements.
        self.timer._start('local hess', 'Assembling local energy terms')
        elem_hess = np.zeros((num['term'], num['term']))
        for kx in range(nx):
            elem_sec_deriv = RMTSlib.compute_sec_deriv(kx+1, num['term'], nx,
                num['elem_list'], sm_options['xlimits'])
            elem_hess += elem_sec_deriv.T.dot(elem_sec_deriv) * \
                sm_options['smoothness'][kx]
        self.timer._stop('local hess', print_done=True)

        # This takes the dense elem_hess matrix and stamps out num['elem'] copies
        # of it to form the full sparse matrix with all the elements included.
        self.timer._start('global hess', 'Assembling global energy terms')
        nnz = num['term'] ** 2 * num['elem']
        num_coeff = num['term'] * num['elem']
        data, rows, cols = RMTSlib.compute_full_from_block(
            nnz, num['term'], num['elem'], elem_hess)
        full_hess = scipy.sparse.csc_matrix((data, (rows, cols)),
            shape=(num_coeff, num_coeff))
        full_hess = full_uniq2coeff.T * full_hess * full_uniq2coeff
        sub_mtx_dict['dv', 'dv'] = full_hess

        num_coeff_uniq = num['uniq'] * 2 ** nx
        diag = sm_options['reg_dv'] * np.ones(num_coeff_uniq)
        arange = np.arange(num_coeff_uniq)
        reg_dv = scipy.sparse.csc_matrix((diag, (arange, arange)))
        sub_mtx_dict['dv', 'dv'] += reg_dv
        self.timer._stop('global hess', print_done=True)

        # This adds the training points, either using a least-squares approach
        # or with exact contraints and Lagrange multipliers.
        # In both approaches, we loop over kx: 0 is for values and kx>0 represents
        # the 1-based index of the derivative given by the training point data.
        if sm_options['mode'] == 'approx':
            self.timer._start('approx', 'Assembling approximation terms')
            for kx in self.training_pts['exact']:
                xt, yt = self.training_pts['exact'][kx]

                nt = xt.shape[0]
                nnz = nt * num['term']
                num_coeff = num['term'] * num['elem']
                data, rows, cols = RMTSlib.compute_jac(kx, 0, nnz, nx, nt,
                    num['elem_list'], sm_options['xlimits'], xt)
                full_jac = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(nt, num_coeff))
                full_jac = full_jac * full_uniq2coeff
                full_jac_sq = full_jac.T * full_jac / sm_options['reg_cons']
                rhs = full_jac.T * yt / sm_options['reg_cons']
                sub_mtx_dict['dv', 'dv'] += full_jac_sq
                sub_rhs_dict['dv'] = rhs
            self.timer._stop('approx', print_done=True)

        elif sm_options['mode'] == 'exact':
            self.timer._start('exact', 'Assembling interpolation terms')
            for kx in self.training_pts['exact']:
                xt, yt = self.training_pts['exact'][kx]

                nt = xt.shape[0]
                nnz = nt * num['term']
                num_coeff = num['term'] * num['elem']
                data, rows, cols = RMTSlib.compute_jac(kx, 0, nnz, nx, nt,
                    num['elem_list'], sm_options['xlimits'], xt)
                full_jac = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(nt, num_coeff))
                full_jac = full_jac * full_uniq2coeff
                sub_mtx_dict['con_%s'%kx, 'dv'] = full_jac
                sub_mtx_dict['dv', 'con_%s'%kx] = full_jac.T
                sub_rhs_dict['con_%s'%kx] = yt

                diag = -sm_options['reg_cons'] * np.ones(nt)
                arange = np.arange(nt)
                reg_cons = scipy.sparse.csc_matrix((diag, (arange, arange)))
                sub_mtx_dict['con_%s'%kx, 'con_%s'%kx] = reg_cons
            self.timer._stop('exact', print_done=True)

        self.timer._start('sparse', 'Assembling global sparse matrix')
        if sm_options['mode'] == 'approx':
            block_names = ['dv']
            block_sizes = [num['uniq'] * 2 ** nx]
        elif sm_options['mode'] == 'exact':
            block_names = ['dv'] + \
                ['con_%s'%kx for kx in self.training_pts['exact']]
            block_sizes = [num['uniq'] * 2 ** nx] + \
                [self.training_pts['exact'][kx][0].shape[0]
                for kx in self.training_pts['exact']]

        mtx, rhs = smt.linalg.assemble_sparse_mtx(
            block_names, block_sizes, sub_mtx_dict, sub_rhs_dict)
        self.timer._stop('sparse', print_done=True)

        mg_matrices = []
        if sm_options['solver_type'] == 'mg' or sm_options['solver_type'] == 'krylov-mg':
            power_two = 2 ** len(sm_options['solver_mg'])
            for kx in range(nx):
                assert num['elem_list'][kx] % power_two == 0, 'Invalid multigrid level'

            elem_lists = [num['elem_list']]
            for ind_mg, mg_factor in enumerate(sm_options['solver_mg']):

                elem_lists.append(elem_lists[-1] / mg_factor)

                nrows = np.prod(elem_lists[-2] + 1) * 2 ** nx
                ncols = np.prod(elem_lists[-1] + 1) * 2 ** nx
                self.timer._start('mg_mat', 'Assembling multigrid op %i (%i x %i mtx)' %
                                  (ind_mg, nrows, ncols))

                if 1:
                    mg_full_uniq2coeff = self._compute_uniq2coeff(nx, elem_lists[-1],
                        np.prod(elem_lists[-1]), num['term'], np.prod(elem_lists[-1] + 1))

                    ne = np.prod(elem_lists[-2] + 1) * 2 ** nx
                    nnz = ne * num['term']
                    num_coeff = num['term'] * np.prod(elem_lists[-1])
                    data, rows, cols = RMTSlib.compute_jac_interp(
                        nnz, nx, elem_lists[-1], elem_lists[-2] + 1, sm_options['xlimits'])
                    mg_jac = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(ne, num_coeff))
                    mg_matrix = mg_jac * mg_full_uniq2coeff
                else:
                    nnz = np.prod(elem_lists[-2] + 1) * 4 ** nx
                    nrows = np.prod(elem_lists[-2] + 1) * 2 ** nx
                    ncols = np.prod(elem_lists[-1] + 1) * 2 ** nx
                    data, rows, cols = RMTSlib.compute_mg_interp(nx, nnz, elem_lists[-1])
                    mg_matrix = scipy.sparse.csc_matrix((data, (rows, cols)),
                                                        shape=(nrows, ncols))

                if sm_options['mode'] == 'exact':
                    num_lagr = np.sum(block_sizes[1:])
                    mg_matrix = scipy.sparse.bmat([
                        [mg_matrix, None],
                        [None, scipy.sparse.identity(num_lagr)]
                    ], format='csc')

                mg_matrices.append(mg_matrix)
                self.timer._stop('mg_mat', print_done=True)

        self.timer._stop('total_assembly')

        self.timer._start('solution')

        sol = np.zeros(rhs.shape)
        smt.linalg.solve_sparse_system(mtx, rhs, sol, sm_options, self.print_status, mg_matrices)

        self.timer._stop('solution')

        self.sol = full_uniq2coeff * sol[:num['uniq'] * 2 ** nx, :]

        self.tmp = mtx

        self.timer._print('total_assembly', 'Total assembly time (sec)')
        self.timer._print('solution', 'Total linear solution time (sec)')

    def fit(self):
        """
        Train the model
        """
        checksum = smt.utils._caching_checksum_sm(self)

        filename = '%s.sm' % self.sm_options['name']
        success, data = smt.utils._caching_load(filename, checksum)
        if not success or not self.sm_options['solver_save']:
            self._fit()
            data = {'sol': self.sol, 'num': self.num}
            smt.utils._caching_save(filename, checksum, data)
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

        nx = self.training_pts['exact'][0][0].shape[1]
        ny = self.training_pts['exact'][0][1].shape[1]
        ne = x.shape[0]

        num = self.num
        sm_options = self.sm_options

        # Compute sparse Jacobian matrix.
        nnz = ne * num['term']
        num_coeff = num['term'] * num['elem']
        data, rows, cols = RMTSlib.compute_jac(kx, 0, nnz, nx, ne,
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
            dx = RMTS.compute_ext_dist(nx, ne, ndx, sm_options['xlimits'], x)
            isexternal = np.array(np.array(dx, bool), float)

            for ix in range(nx):
                # Now we compute the first order term where we have a
                # derivative times (x_k - b_k) or (x_k - a_k).
                nnz = ne * num['term']
                data_tmp, rows, cols = RMTSlib.compute_jac(kx, ix+1, nnz, nx, ne,
                    num['elem_list'], sm_options['xlimits'], x)
                data_tmp *= dx[:, ix]

                # If we are evaluating a derivative (with index kx),
                # we zero the first order terms for which dx_k = 0.
                if kx != 0:
                    data_tmp *= 1 - isexternal[:, kx-1]

                data += data_tmp

        mtx = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(ne, num_coeff))

        return mtx.dot(self.sol)
