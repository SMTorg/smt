"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
"""
from __future__ import division

import numpy as np
import scipy.sparse
from six.moves import range

from smt.utils.linear_solvers import get_solver
from smt.utils.line_search import get_line_search_class
from smt.utils.caching import _caching_checksum_sm, _caching_load, _caching_save
from smt.rmt import RMT

from smt import RMTBlib


class RMTB(RMT):
    """
    Regularized Minimal-energy Tensor-product B-Spline (RMTB) interpolant.

    RMTB builds an approximation from a tensor product of B-spline curves.
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
    - The data should be structured - RMTB does not handle track data well
    - RMTB approximates, not interpolates - it does not pass through the
    training points
    """

    def _declare_options(self):
        super(RMTB, self)._declare_options()
        declare = self.options.declare

        declare('name', 'RMTB', types=str,
                desc='Regularized Minimal-energy Tensor-product B-spline interpolant')
        declare('order', 3, types=(int, list, np.ndarray),
                desc='B-spline order in each dimension - length [nx]')
        declare('num_ctrl_pts', 10, types=(int, list, np.ndarray),
                desc='# B-spline control points in each dimension - length [nx]')

    def _compute_jac(self, ix1, ix2, x):
        xlimits = self.options['xlimits']

        t = np.zeros(x.shape)
        for kx in range(self.num['x']):
            t[:, kx] = (x[:, kx] - xlimits[kx, 0]) /\
                (xlimits[kx, 1] - xlimits[kx, 0])
        t = np.maximum(t, 0. + 1e-15)
        t = np.minimum(t, 1. - 1e-15)

        n = x.shape[0]
        nnz = n * self.num['order']
        data, rows, cols = RMTBlib.compute_jac(ix1, ix2, self.num['x'], n, nnz,
            self.num['order_list'], self.num['ctrl_list'], t)
        if ix1 != 0:
            data /= xlimits[ix1-1, 1] - xlimits[ix1-1, 0]
        if ix2 != 0:
            data /= xlimits[ix2-1, 1] - xlimits[ix2-1, 0]

        return data, rows, cols

    def _compute_energy_terms(self):
        num = self.num
        options = self.options
        xlimits = options['xlimits']

        nt_list = num['ctrl_list'] - num['order_list'] + 1
        nt = np.prod(nt_list)
        t = RMTBlib.compute_quadrature_points(nt, num['x'], nt_list)

        # Square root of volume of each integration element and of the whole domain
        elem_vol = np.prod((xlimits[:, 1] - xlimits[:, 0]) / nt_list)
        total_vol = np.prod(xlimits[:, 1] - xlimits[:, 0])

        full_hess = scipy.sparse.csc_matrix((num['dof'], num['dof']))
        for kx in range(num['x']):
            nnz = nt * num['order']
            data, rows, cols = RMTBlib.compute_jac(kx+1, kx+1, num['x'], nt, nnz,
                num['order_list'], num['ctrl_list'], t)
            data /= (xlimits[kx, 1] - xlimits[kx, 0]) ** 2
            rect_mtx = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(nt, num['ctrl']))
            full_hess += rect_mtx.T * rect_mtx \
                * (elem_vol / total_vol * options['smoothness'][kx])

        return full_hess

    def _fit(self):
        """
        Train the model
        """
        options = self.options
        xlimits = options['xlimits']

        nx = self.training_pts['exact'][0][0].shape[1]
        if isinstance(options['order'], int):
            options['order'] = [options['order']] * nx
        if isinstance(options['num_ctrl_pts'], int):
            options['num_ctrl_pts'] = [options['num_ctrl_pts']] * nx
        if options['smoothness'] is None:
            options['smoothness'] = [1.0] * nx

        num = {}
        # number of inputs and outputs
        num['x'] = self.training_pts['exact'][0][0].shape[1]
        num['y'] = self.training_pts['exact'][0][1].shape[1]
        num['order_list'] = np.array(options['order'], int)
        num['order'] = np.prod(num['order_list'])
        num['ctrl_list'] = np.array(options['num_ctrl_pts'], int)
        num['ctrl'] = np.prod(num['ctrl_list'])
        num['knots_list'] = num['order_list'] + num['ctrl_list']
        num['knots'] = np.sum(num['knots_list'])
        # total number of training points (function values and derivatives)
        num['t'] = 0
        for kx in self.training_pts['exact']:
            num['t'] += self.training_pts['exact'][kx][0].shape[0]
        # for RMT
        num['coeff'] = num['ctrl']
        num['support'] = num['order']
        num['dof'] = num['ctrl']

        self.num = num

        self.printer.max_print_depth = options['max_print_depth']

        with self.printer._timed_context('Pre-computing matrices'):

            with self.printer._timed_context('Initializing Hessian'):
                full_hess = self._initialize_hessian()

            if options['min_energy']:
                with self.printer._timed_context('Computing energy terms'):
                    full_hess += self._compute_energy_terms()

            with self.printer._timed_context('Computing approximation terms'):
                full_jac_dict = self._compute_approx_terms()

            full_hess *= options['reg_cons']

            mg_matrices = []

        with self.printer._timed_context('Solving for degrees of freedom'):
            sol = self._solve(full_hess, full_jac_dict, mg_matrices)

        self.sol = sol
