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


class RMT(SM):
    """
    Regularized Minimal-energy Tensor-product interpolant base class for RMTS and RMTB.
    """

    def _opt_func(self, sol, p, full_hess, full_jac_dict, yt_dict):
        c = 0.5 / self.num['t']

        func = 0.5 * np.dot(sol, full_hess * sol)
        for kx in self.training_pts['exact']:
            full_jac = full_jac_dict[kx]
            yt = yt_dict[kx]
            func += c * np.sum((full_jac * sol - yt) ** p)

        return func

    def _opt_grad(self, sol, p, full_hess, full_jac_dict, yt_dict):
        c = 0.5 / self.num['t']

        grad = full_hess * sol
        for kx in self.training_pts['exact']:
            full_jac = full_jac_dict[kx]
            yt = yt_dict[kx]
            grad += c * full_jac.T * p * (full_jac * sol - yt) ** (p - 1)

        return grad

    def _opt_hess(self, sol, p, full_hess, full_jac_dict, yt_dict):
        c = 0.5 / self.num['t']

        hess = scipy.sparse.csc_matrix(full_hess)
        for kx in self.training_pts['exact']:
            full_jac = full_jac_dict[kx]
            yt = yt_dict[kx]

            diag_vec = p * (p - 1) * (full_jac * sol - yt) ** (p - 2)
            diag_mtx = scipy.sparse.diags(diag_vec, format='csc')
            hess += c * full_jac.T * diag_mtx * full_jac

        return hess

    def _opt_hess_2(self, full_hess, full_jac_dict):
        c = 0.5 / self.num['t']
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

    def evaluate(self, x, kx):
        """
        Evaluate the surrogate model at x.

        Parameters
        ----------
        x : np.ndarray[n_eval,dim]
            An array giving the point(s) at which the prediction(s) should be made.
        kx : int or None
            None if evaluation of the interpolant is desired.
            int  if evaluation of derivatives of the interpolant is desired
                 with respect to the kx^{th} input variable (kx is 0-based).

        Returns
        -------
        y : np.ndarray[n_eval,1]
            - An array with the output values at x.
        """
        n = x.shape[0]

        num = self.num
        sm_options = self.sm_options

        data, rows, cols = self._compute_jac(kx, 0, x)

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
        if sm_options['extrapolate']:

            # First we evaluate the vector pointing to each evaluation points
            # from the nearest point on the domain, in a matrix called dx.
            # If the ith evaluation point is not external, dx[i, :] = 0.
            ndx = n * num['support']
            dx = RMTSlib.compute_ext_dist(num['x'], n, ndx, sm_options['xlimits'], x)
            isexternal = np.array(np.array(dx, bool), float)

            for ix in range(num['x']):
                # Now we compute the first order term where we have a
                # derivative times (x_k - b_k) or (x_k - a_k).
                data_tmp, rows, cols = self._compute_jac(kx, ix+1, x)
                data_tmp *= dx[:, ix]

                # If we are evaluating a derivative (with index kx),
                # we zero the first order terms for which dx_k = 0.
                if kx != 0:
                    data_tmp *= 1 - isexternal[:, kx-1]

                data += data_tmp

        mtx = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(n, num['coeff']))

        return mtx.dot(self.sol)
