"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
"""
from __future__ import division

import numpy as np
import scipy.sparse
from six.moves import range

import MBRlib

from smt.utils.linear_solvers import get_solver
from smt.utils.line_search import get_line_search_class
from smt.utils.caching import _caching_checksum_sm, _caching_load, _caching_save
from smt.sm import SM


class MBR(SM):
    """
    Multi-dimensional B-spline Regression (MBR).

    MBR builds an approximation from a tensor product of B-spline curves.
    In 1-D it is a B-spline curve, in 2-D it is a B-spline surface, in 3-D
    it is a B-spline volume, and so on - it works for any arbitrary number
    of dimensions. However, the training points should preferably be
    arranged in a structured fashion.

    Advantages:
    - Evaluation time is independent of the number of training points
    - The smoothness can be tuned by adjusting the B-spline order and the
    number of B-spline control points (the latter also affects performance)

    Disadvantages:
    - Training time scales poorly with the # dimensions
    - The data should be structured - MBR does not handle track data well
    - MBR approximates, not interpolates - it does not pass through the
    training points
    """

    def _set_default_options(self):
        sm_options = {
            'name': 'MBR', # Multi-dimensional B-spline Regression
            'xlimits': [],    # flt ndarray[nx, 2]: lower/upper bounds in each dimension
            'order': [], # int ndarray[nx]: B-spline order in each dimension
            'num_ctrl_pts': [], # int ndarray[nx]: num. B-spline control pts. in each dim.
            'extrapolate': True, # perform linear extrapolation for external eval points
            'reg_dv': 1e-4, # regularization coeff. for dv block
            'reg_cons': 1e-10, # negative of reg. coeff. for Lagrange mult. block
            'solver': 'krylov-lu',    # Linear solver: 'gmres' or 'cg'
            'mg_factors': [], # Multigrid level
            'save_solution': True,  # Whether to save linear system solution
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

    def _fit(self):
        """
        Train the model
        """
        sm_options = self.sm_options
        xlimits = sm_options['xlimits']

        nx = self.training_pts['exact'][0][0].shape[1]
        ny = self.training_pts['exact'][0][1].shape[1]

        num = {}
        num['order_list'] = np.array(sm_options['order'], int)
        num['order'] = np.prod(num['order_list'])
        num['ctrl_list'] = np.array(sm_options['num_ctrl_pts'], int)
        num['ctrl'] = np.prod(num['ctrl_list'])
        num['knots_list'] = num['order_list'] + num['ctrl_list']
        num['knots'] = np.sum(num['knots_list'])

        self.num = num

        mtx = scipy.sparse.csc_matrix((num['ctrl'], num['ctrl']))
        rhs = np.zeros((num['ctrl'], ny))

        if 1:
            nt_list = num['ctrl_list'] - num['order_list'] + 1
            nt = np.prod(nt_list)
            t = MBRlib.compute_quadrature_points(nt, nx, nt_list)

            # Square root of volume of each integration element
            elem_vol_sqrt = np.prod((xlimits[:, 1] - xlimits[:, 0]) / nt_list)
            for kx in range(nx):
                nnz = nt * num['order']
                data, rows, cols = MBRlib.compute_jac(kx+1, kx+1, nx, nt, nnz,
                    num['order_list'], num['ctrl_list'], t)
                data *= elem_vol_sqrt
                data /= (xlimits[kx, 1] - xlimits[kx, 0]) ** 2
                rect_mtx = scipy.sparse.csc_matrix((data, (rows, cols)),
                    shape=(nt, num['ctrl']))
                mtx = mtx + rect_mtx.T * rect_mtx * sm_options['reg_cons']

        for kx in self.training_pts['exact']:
            xt, yt = self.training_pts['exact'][kx]

            xmin = np.min(xt, axis=0)
            xmax = np.max(xt, axis=0)
            assert np.all(xlimits[:, 0] <= xmin), 'Training pts below min for %s' % kx
            assert np.all(xlimits[:, 1] >= xmax), 'Training pts above max for %s' % kx

            t = np.zeros(xt.shape)
            for ix in range(nx):
                t[:, ix] = (xt[:, ix] - xlimits[ix, 0]) /\
                    (xlimits[ix, 1] - xlimits[ix, 0])

            nt = xt.shape[0]
            nnz = nt * num['order']
            data, rows, cols = MBRlib.compute_jac(kx, 0, nx, nt, nnz,
                num['order_list'], num['ctrl_list'], t)
            if kx > 0:
                data /= xlimits[kx-1, 1] - xlimits[kx-1, 0]

            rect_mtx = scipy.sparse.csc_matrix((data, (rows, cols)),
                shape=(nt, num['ctrl']))

            mtx = mtx + rect_mtx.T * rect_mtx
            rhs += rect_mtx.T * yt

        diag = sm_options['reg_dv'] * sm_options['reg_cons'] * np.ones(num['ctrl'])
        arange = np.arange(num['ctrl'])
        reg = scipy.sparse.csc_matrix((diag, (arange, arange)))
        mtx = mtx + reg

        sol = np.zeros(rhs.shape)

        solver = get_solver(sm_options['solver'])

        with self.printer._timed_context('Initializing linear solver'):
            solver._initialize(mtx, self.printer)

        for ind_y in range(rhs.shape[1]):
            with self.printer._timed_context('Solving linear system (col. %i)' % ind_y):
                solver._solve(rhs[:, ind_y], sol[:, ind_y], ind_y=ind_y)

        self.sol = sol

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
        x: np.ndarray[n_eval,dim]
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

        nx = self.training_pts['exact'][0][0].shape[1]
        ny = self.training_pts['exact'][0][1].shape[1]
        ne = x.shape[0]
        xlimits = self.sm_options['xlimits']
        num = self.num
        sm_options = self.sm_options

        t = np.zeros(x.shape)
        for ix in range(nx):
            t[:, ix] = (x[:, ix] - xlimits[ix, 0]) /\
                (xlimits[ix, 1] - xlimits[ix, 0])
        t = np.maximum(t, 0. + 1e-15)
        t = np.minimum(t, 1. - 1e-15)

        nnz = ne * num['order']
        data, rows, cols = MBRlib.compute_jac(kx, 0, nx, ne, nnz,
            num['order_list'], num['ctrl_list'], t)
        if kx != 0:
            data /= xlimits[kx-1, 1] - xlimits[kx-1, 0]

        if sm_options['extrapolate']:
            ndx = ne * num['order']
            dx = MBRlib.compute_ext_dist(nx, ne, ndx, sm_options['xlimits'], x)
            isexternal = np.array(np.array(dx, bool), float)

            for ix in range(nx):
                nnz = ne * num['order']
                data_tmp, rows, cols = MBRlib.compute_jac(kx, ix+1, nx, ne, nnz,
                    num['order_list'], num['ctrl_list'], t)
                data_tmp /= xlimits[kx-1, 1] - xlimits[kx-1, 0]
                if kx != 0:
                    data_tmp /= xlimits[kx-1, 1] - xlimits[kx-1, 0]
                data_tmp *= dx[:, ix]

                # If we are evaluating a derivative (with index kx),
                # we zero the first order terms for which dx_k = 0.
                if kx != 0:
                    data_tmp *= 1 - isexternal[:, kx-1]

                data += data_tmp

        rect_mtx = scipy.sparse.csc_matrix((data, (rows, cols)),
            shape=(ne, num['ctrl']))

        return rect_mtx.dot(self.sol)
