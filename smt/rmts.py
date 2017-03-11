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
from smt.rmt import RMT


class RMTS(RMT):
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
            'xlimits': [],    # flt ndarray[nx, 2]: lower/upper bounds in each dimension
            'num_elem': [],  # int ndarray[nx]: num. of elements in each dimension
            'smoothness': [], # flt ndarray[nx]: smoothness parameter in each dimension
            'reg_dv': 1e-10, # regularization coeff. for dv block
            'reg_cons': 1e-10, # negative of reg. coeff. for Lagrange mult. block
            'extrapolate': False, # perform linear extrapolation for external eval points
            'min_energy': True, # whether to include energy minimizaton terms
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

    def _compute_jac(self, ix1, ix2, x):
        n = x.shape[0]
        nnz = n * self.num['term']
        return RMTSlib.compute_jac(ix1, ix2, nnz, self.num['x'], n,
            self.num['elem_list'], self.sm_options['xlimits'], x)

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

    def _compute_energy_terms(self):
        num = self.num
        sm_options = self.sm_options
        xlimits = sm_options['xlimits']

        # Square root of volume of each integration element and of the whole domain
        elem_vol = np.prod((xlimits[:, 1] - xlimits[:, 0]) / num['elem_list'])
        total_vol = np.prod(xlimits[:, 1] - xlimits[:, 0])

        # This computes the positive-definite, symmetric matrix yields the energy
        # for an element when pre- and post-multiplied by a vector of function and
        # derivative values for the element. This matrix applies to all elements.
        elem_hess = np.zeros((num['term'], num['term']))
        for kx in range(num['x']):
            elem_sec_deriv = RMTSlib.compute_sec_deriv(kx+1, num['term'], num['x'],
                num['elem_list'], xlimits)
            elem_hess += elem_sec_deriv.T.dot(elem_sec_deriv) \
                * (elem_vol / total_vol * sm_options['smoothness'][kx])

        # This takes the dense elem_hess matrix and stamps out num['elem'] copies
        # of it to form the full sparse matrix with all the elements included.
        nnz = num['term'] ** 2 * num['elem']
        num_coeff = num['term'] * num['elem']
        data, rows, cols = RMTSlib.compute_full_from_block(
            nnz, num['term'], num['elem'], elem_hess)
        full_hess_coeff = scipy.sparse.csc_matrix((data, (rows, cols)),
            shape=(num_coeff, num_coeff))

        return full_hess_coeff

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

        return mg_matrix

    def _compute_mg_matrices(self):
        num = self.num
        sm_options = self.sm_options

        elem_lists = [num['elem_list']]
        mg_matrices = []
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
        # for RMT
        num['coeff'] = num['term'] * num['elem']
        num['support'] = num['term']
        num['dof'] = num['uniq'] * 2 ** num['x']

        if len(sm_options['smoothness']) == 0:
            sm_options['smoothness'] = [1.0] * num['x']

        self.num = num

        self.printer.max_print_depth = sm_options['max_print_depth']

        with self.printer._timed_context('Pre-computing matrices'):

            with self.printer._timed_context('Computing uniq2coeff'):
                full_uniq2coeff = self._compute_uniq2coeff(
                    num['x'], num['elem_list'], num['elem'], num['term'], num['uniq'])

            with self.printer._timed_context('Initializing Hessian'):
                full_hess = self._initialize_hessian()

            if sm_options['min_energy']:
                with self.printer._timed_context('Computing energy terms'):
                    full_hess_coeff = self._compute_energy_terms()
                    full_hess += full_uniq2coeff.T * full_hess_coeff * full_uniq2coeff

            with self.printer._timed_context('Computing approximation terms'):
                full_jac_dict = self._compute_approx_terms()
                for kx in self.training_pts['exact']:
                    full_jac_dict[kx] = full_jac_dict[kx] * full_uniq2coeff

            full_hess *= sm_options['reg_cons']

            mg_matrices = self._compute_mg_matrices()

        with self.printer._timed_context('Solving for degrees of freedom'):
            sol = self._solve(full_hess, full_jac_dict, mg_matrices)

        self.sol = full_uniq2coeff * sol[:num['uniq'] * 2 ** num['x'], :]
