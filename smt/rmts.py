"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

TODO:
- Address approx vs exact issue
- Generalize to arbitrary number of outputs
"""

from __future__ import division

import numpy
import scipy.sparse
import RMTSlib
import smt.utils
from six.moves import range


class RMTS(object):
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
    - No risk of convergence failure

    Disadvantages:
    - Training time scales poorly with the number of dimensions
    - Works with 4 dimensions at most
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
        }
        printf_options = {
            'global': True,     # Overriding option to print output
            'time_eval': True,  # Print evaluation times
            'time_train': False, # Print assembly and solution time summary
            'problem': True,    # Print problem information
        }

        self.sm_options = sm_options
        self.printf_options = printf_options

    def fit(self):
        """
        Train the model
        """
        sm_options = self.sm_options

        num = {}
        # number of elements
        num['elem_list'] = numpy.array(sm_options['num_elems'])
        num['elem'] = numpy.prod(num['elem_list'])
        # number of terms/coefficients per element
        num['term_list'] = 4 * numpy.ones(len(sm_options['num_elems']))
        num['term'] = numpy.prod(num['term_list'])
        # number of nodes
        num['uniq_list'] = num['elem_list'] + 1
        num['uniq'] = numpy.prod(num['uniq_list'])

        self.num = num

        sub_mtx_dict = {}
        sub_rhs_dict = {}

        # This computes an num['term'] x num['term'] matrix called coeff2nodal.
        # Multiplying this matrix with the list of coefficients for an element
        # yields the list of function and derivative values at the element nodes.
        # We need the inverse, but the matrix size is small enough to invert since
        # RMTS is normally only used for 1 <= nx <= 4 in most cases.
        elem_coeff2nodal = RMTSlib.compute_coeff2nodal(num['x'], num['term'])
        elem_nodal2coeff = numpy.linalg.inv(elem_coeff2nodal)

        # This computes a num_coeff_elem x num_coeff_uniq permutation matrix called
        # uniq2elem. This sparse matrix maps the unique list of nodal function and
        # derivative values to the same function and derivative values, but ordered
        # by element, with repetition.
        nnz = num['elem'] * num['term']
        num_coeff_elem = num['term'] * num['elem']
        num_coeff_uniq = num['uniq'] * 2 ** num['x']
        data, rows, cols = RMTSlib.compute_uniq2elem(nnz, num['x'], num['elem_list'])
        full_uniq2elem = scipy.sparse.csc_matrix((data, (rows, cols)),
            shape=(num_coeff_elem, num_coeff_uniq))

        # This computes the positive-definite, symmetric matrix yields the energy
        # for an element when pre- and post-multiplied by a vector of function and
        # derivative values for the element. This matrix applies to all elements.
        elem_hess = numpy.zeros((num['term'], num['term']))
        for kx in range(num['x']):
            elem_sec_deriv = RMTSlib.compute_sec_deriv(kx+1, num['term'], num['x'],
                num['elem_list'], sm_options['xlimits'])
            elem_hess += elem_sec_deriv.T.dot(elem_sec_deriv)
        elem_hess = elem_nodal2coeff.T.dot(elem_hess).dot(elem_nodal2coeff)

        # This takes the dense elem_hess matrix and stamps out num['elem'] copies
        # of it to form the full sparse matrix with all the elements included.
        nnz = num['term'] ** 2 * num['elem']
        num_coeff = num['term'] * num['elem']
        data, rows, cols = RMTSlib.compute_full_from_block(
            nnz, num['term'], num['elem'], elem_hess)
        full_hess = scipy.sparse.csc_matrix((data, (rows, cols)),
            shape=(num_coeff, num_coeff))
        full_hess = full_uniq2elem.T * full_hess * full_uniq2elem
        sub_mtx_dict['dv', 'dv'] = full_hess

        num_coeff_uniq = num['uniq'] * 2 ** num['x']
        ones = numpy.ones(num_coeff_uniq)
        arange = numpy.arange(num_coeff_uniq)
        reg_dv = scipy.sparse.csc_matrix((ones, (arange, arange)))
        sub_mtx_dict['dv', 'dv'] = reg_dv

        # This computes the matric full_uniq2coeff, which maps the unique
        # degrees of freedom to the list of coefficients ordered by element.
        nnz = num['term'] ** 2 * num['elem']
        num_coeff = num['term'] * num['elem']
        data, rows, cols = RMTSlib.compute_full_from_block(
            nnz, num['term'], num['elem'], elem_nodal2coeff)
        full_nodal2coeff = scipy.sparse.csc_matrix((data, (rows, cols)),
            shape=(num_coeff, num_coeff))
        full_uniq2coeff = full_nodal2coeff * full_uniq2elem

        # This adds the training points, either using a least-squares approach
        # or with exact contraints and Lagrange multipliers.
        # In both approaches, we loop over kx: 0 is for values and kx>0 represents
        # the 1-based index of the derivative given by the training point data.
        if self.sm_options['mode'] == 'approx':
            for kx in pts['exact']:
                xt, yt = self.training_pts['exact'][kx]

                nx = xt.shape[1]
                ny = yt.shape[1]
                nt = xt.shape[0]
                num_coeff = num['term'] * num['elem']
                nnz = nf * num['term'] * num['term']
                data, rows, cols, rhs = RMTSlib.compute_jac_sq(
                    kx, nnz, nx, ny, num['elem_list'],
                    nt, num_coeff, sm_options['xlimits'], xt, yt)
                full_jac_sq = scipy.sparse.csc_matrix((data, (rows, cols)),
                    shape=(num_coeff, num_coeff))
                full_jac_sq = full_uniq2coeff.T * full_jac_sq * full_uniq2coeff /\
                    sm_options['reg_cons']
                rhs = full_uniq2coeff.T * rhs / sm_options['reg_cons']
                sub_mtx_dict['dv', 'dv'] = full_jac_sq
                sub_mtx_dict['dv', 'dv'] = full_hess
                sub_rhs_dict['dv'] = rhs
        elif self.sm_options['mode'] == 'exact':
            for kx in pts['exact']:
                xt, yt = self.training_pts['exact'][kx]

                nx = xt.shape[1]
                ny = yt.shape[1]
                nt = xt.shape[0]

                nnz = nt * num['term']
                num_coeff = num['term'] * num['elem']
                data, rows, cols = RMTSlib.compute_jac(kx, 0, nnz, nx, nt,
                    num['elem_list'], sm_options['xlimits']), xt)
                full_jac = scipy.sparse.csc_matrix((data, (rows, cols)),
                                  shape=(nt, num_coeff))
                full_jac = full_jac * full_uniq2coeff
                sub_mtx_dict['con_%s'%kx, 'dv'] = full_jac
                sub_mtx_dict['dv', 'con_%s'%kx] = full_jac.T
                sub_rhs_dict['con_%s'%kx] = yt

                ones = numpy.ones(nt)
                arange = numpy.arange(nt)
                reg_cons = scipy.sparse.csc_matrix((-ones, (arange, arange)))
                sub_mtx_dict['con_%s'%kx, 'con_%s'%kx] = reg_cons

        if sm_options['mode'] == 'approx':
            block_names = ['dv']
            block_sizes = [num['uniq'] * 2 ** num['x']]
        elif sm_options['mode'] == 'exact':
            block_names = ['dv'] + ['con_%s'%kx for kx in pts['exact']]
            block_sizes = [num['uniq'] * 2 ** num['x']] + \
                [self.training_pts['exact'][kx][0].shape[0] for kx in pts['exact']]

        mtx, rhs = smt.utils.assemble_sparse_mtx(
            block_names, block_sizes, sub_mtx_dict, sub_rhs_dict)

        solver_options = {
            'pc': 'lu',
            'print': True,
            'atol': 1e-15,
            'ilimit': 100,
        }
        sol = numpy.zeros(rhs.shape)
        smt.utils.solve_sparse_system(mtx, rhs, sol, solver_options)

        self.sol = full_uniq2coeff * sol[:num['uniq'] * 2 ** num['x'], :]

    def evaluate(self, x):
        """
        This function evaluates the Gaussian Process model at x.

        Parameters
        ----------
        x: np.ndarray[n_eval,dim]
            - An array giving the point(s) at which the prediction(s) should be
              made.

        Returns
        -------
        y : np.ndarray[n_eval,1]
            - An array with the Best Linear Unbiased prediction at x.
        """
        kx = 0

        nx = self.training_pts['exact'][0][0].shape[0]
        ny = self.training_pts['exact'][0][0].shape[0]
        ne = x.shape[0]

        num = self.num
        sm_options = self.sm_options

        # Compute sparse Jacobian matrix.
        nnz = ne * num['term']
        num_coeff = num['term'] * num['elem']
        data, rows, cols = RMTSlib.compute_jac(kx, 0, nnz, nx, ne,
            num['elem_list'], sm_options['xlimits']), x)

        # Compute boolean and index vectors for points that are outside xlimits.
        isexternal_any, isexternal = RMTSlib.compute_ext_mask(
            nx, ne, sm_options['xlimits'], x)
        isexternal_any = numpy.array(isexternal_any, bool)
        external_inds = numpy.arange(ne)[isexternal_any]

        if numpy.any(isexternal_any):

            # Blank all entries in the sparse matrix corresponding to points whose
            # (kx)th value is external, where kx is the index of the deriv. we want.
            if kx != 0:
                for ind in range(num['term']):
                    data[ind::num['term']] *= 1 - isexternal[:, kx-1]

            # Get dx and dx2, which contain position vectors from the nearest points
            # in the domain to the external evaluation points.
            # dx2 is simply an outer product of dx with itself.
            x_ext = x[isexternal_any, :]
            n_ext = x_ext.shape[0]
            dx, dx2 = lib.tpsextrapolation(nx, n_ext, n_ext * 4**nx, xlimits, x_ext)

            # Loop over each input x_i.
            for ix in range(nx):
                # For all x_i that are external, compute the derivative in that
                # direction and multiply by dx_i.
                # If we're already evaluating a derivative w.r.t. x_k, the following
                # would compute a second derivative.
                nnz = n_ext * num['term']
                data2, rows2, cols2 = RMTSlib.compute_jac(kx, ix+1, nnz, nx, n_ext,
                    num['elem_list'], sm_options['xlimits'], x_ext)
                data2 *= dx[:, ix]

                # But if we're evaluating a derivative (the kth), blank the term
                # we just computed if the kth coordinate is external.
                # For example, if we want df/dx_1 and x_1 is external, then
                # df/dx_1 at the external point is simply df/dx_1 at the nearest
                # domain point, other than second-order terms (to follow).
                if kx != 0:
                    for ind in xrange(num['term']):
                        data2 *= 1 - numpy.array(dx[:, kx-1], bool)

                # If we're evaluating the kth derivative, get the value of the
                # derivative from the nearest point on the domain.
                if kx == ix+1:
                    data3, rows3, cols3 = RMTSlib.compute_jac(0, ix+1, nnz, nx, n_ext,
                        num['elem_list'], sm_options['xlimits'], x_ext)
                    data3 *= numpy.array(dx[:, ix], bool)
                    data2 += data3

                # Add to our list of non-zeros.
                rows2 = external_inds[rows2]
                data = numpy.concatenate([data, data2])
                rows = numpy.concatenate([rows, rows2])
                cols = numpy.concatenate([cols, cols2])

            # Now deal with second-order terms.
            for ix in xrange(nx):
                for jx in xrange(nx):
                    # Only cross terms.
                    if not ix == jx:
                        # If we're not evaluating a derivative, we just need to compute
                        # 0.5 * d2f/du/dv * du * dv where the factor of a half is there
                        # because each (ix, jx) pair gets covered twice.
                        if kx == 0:
                            data2, rows2, cols2 = RMTSlib.compute_jac(
                                ix+1, jx+1, nnz, nx, n_ext,
                                num['elem_list'], sm_options['xlimits'], x_ext)
                            data2 *= 0.5 * dx2[:, ix, jx]
                        else:
                            # For evaluating derivatives, we need the complex version
                            # because we use complex-step to get 3rd derivatives.
                            # This does the same as the above block of code, except
                            # with an extra derivative.
                            data2, rows2, cols2 = RMTSlib.compute_jac_cplx(
                                kx, ix+1, jx+1, nnz, nx, n_ext,
                                num['elem_list'], sm_options['xlimits'], x_ext)
                            data2 *= 0.5 * dx2[:, ix, jx]

                            if kx == ix+1:
                                data3, rows3, cols3 = RMTSlib.compute_jac(
                                    ix+1, jx+1, nnz, nx, n_ext,
                                    num['elem_list'], sm_options['xlimits'], x_ext)
                                data3 *= 0.5 * numpy.array(dx[:, ix], bool) * dx[:, jx]
                                data2 += data3
                            if kx == jx+1:
                                data3, rows3, cols3 = RMTSlib.compute_jac(
                                    ix+1, jx+1, nnz, nx, n_ext,
                                    num['elem_list'], sm_options['xlimits'], x_ext)
                                data3 *= 0.5 * numpy.array(dx[:, jx], bool) * dx[:, ix]
                                data2 += data3

                        rows2 = extmap[rows2]
                        data = numpy.concatenate([data, data2])
                        rows = numpy.concatenate([rows, rows2])
                        cols = numpy.concatenate([cols, cols2])

        mtx = scipy.sparse.csc_matrix((data, (rows, cols)),
            shape=(ne, num['elem'] * num['term']))

        return mtx.dot(self.sol)
