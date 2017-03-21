"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
"""
from __future__ import division

import numpy as np
import scipy.sparse
from six.moves import range
from numbers import Integral

from smt.utils.linear_solvers import get_solver
from smt.utils.line_search import get_line_search_class
from smt.rmt import RMT

from smt import RMTSlib


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

    def _declare_options(self):
        super(RMTS, self)._declare_options()
        declare = self.options.declare

        declare('name', 'RMTS', types=str,
                desc='Regularized Minimal-energy Tensor-product Spline interpolant')
        declare('num_elements', 4, types=(Integral, list, np.ndarray),
                desc='# elements in each dimension - ndarray [nx]')

    def _compute_jac_raw(self, ix1, ix2, x):
        n = x.shape[0]
        nnz = n * self.num['term']
        return RMTSlib.compute_jac(ix1, ix2, nnz, self.num['x'], n,
            self.num['elem_list'], self.options['xlimits'], x)

    def _compute_jac(self, ix1, ix2, x):
        data, rows, cols = self._compute_jac_raw(ix1, ix2, x)
        n = x.shape[0]
        full_jac = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(n, self.num['coeff']))
        return full_jac * self.full_uniq2coeff

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

    def _get_num_dict(self):
        num = {}
        # number of inputs and outputs
        num['x'] = self.training_pts['exact'][0][0].shape[1]
        num['y'] = self.training_pts['exact'][0][1].shape[1]
        # number of elements
        num['elem_list'] = np.array(self.options['num_elements'], int)
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

        return num

    def _fit(self):
        """
        Train the model
        """
        options = self.options
        nx = self.training_pts['exact'][0][0].shape[1]

        for name in ['smoothness', 'num_elements']:
            if isinstance(options[name], (int, float)):
                options[name] = [options[name]] * nx
            options[name] = np.atleast_1d(options[name])

        self.printer.max_print_depth = options['max_print_depth']

        # -----------------

        self.num = num = self._get_num_dict()

        with self.printer._timed_context('Pre-computing matrices', 'assembly'):

            with self.printer._timed_context('Computing uniq2coeff', 'uniq2coeff'):
                self.full_uniq2coeff = self._compute_uniq2coeff(
                    num['x'], num['elem_list'], num['elem'], num['term'], num['uniq'])

            with self.printer._timed_context('Initializing Hessian', 'init_hess'):
                full_hess = self._initialize_hessian()

            if options['min_energy']:
                with self.printer._timed_context('Computing energy terms', 'energy'):
                    full_hess += self._compute_energy_terms()

            with self.printer._timed_context('Computing approximation terms', 'approx'):
                full_jac_dict = self._compute_approx_terms()

            full_hess *= options['reg_cons']

        with self.printer._timed_context('Solving for degrees of freedom', 'total_solution'):
            sol = self._solve(full_hess, full_jac_dict)

        self.sol = self.full_uniq2coeff * sol[:num['uniq'] * 2 ** num['x'], :]
