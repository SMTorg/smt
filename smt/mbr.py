"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

TODO:
- Address approx vs exact issue
- Generalize to arbitrary number of outputs
"""

from __future__ import division

import numpy
import scipy.sparse
import MBRlib
import smt.utils
from sm import SM
from six.moves import range


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
            'reg': 1e-10, # regularization coeff. for dv block
        }
        printf_options = {
            'global': True,     # Overriding option to print output
            'time_eval': True,  # Print evaluation times
            'time_train': True, # Print assembly and solution time summary
            'problem': True,    # Print problem information
        }
        solver_options = {
            'pc': 'lu',    # Preconditioner: 'ilu', 'lu', or 'nopc'
            'print': True, # Whether to print convergence progress (i.e., residual norms)
            'atol': 1e-15, # Absolute linear system convergence tolerance
            'ilimit': 100, # Linear system iteration limit
            'save': True,  # Whether to save linear system solution
        }

        self.sm_options = sm_options
        self.printf_options = printf_options
        self.solver_options = solver_options

    def _fit(self):
        """
        Train the model
        """
        sm_options = self.sm_options

        nx = self.training_pts['exact'][0][0].shape[1]
        ny = self.training_pts['exact'][0][1].shape[1]

        num = {}
        num['order_list'] = numpy.array(sm_options['order'], int)
        num['order'] = numpy.prod(num['order_list'])
        num['ctrl_list'] = numpy.array(sm_options['num_ctrl_pts'], int)
        num['ctrl'] = numpy.prod(num['ctrl_list'])
        num['knots_list'] = num['order_list'] + num['ctrl_list']
        num['knots'] = numpy.sum(num['knots_list'])

        self.num = num

        mtx = scipy.sparse.csc_matrix((num['ctrl'], num['ctrl']))
        rhs = numpy.zeros((num['ctrl'], ny))
        xlimits = sm_options['xlimits']
        for kx in self.training_pts['exact']:
            xt, yt = self.training_pts['exact'][kx]

            t = numpy.zeros(xt.shape)
            for ix in range(nx):
                t[:, ix] = (xt[:, ix] - xlimits[ix, 0]) /\
                    (xlimits[ix, 1] - xlimits[ix, 0])

            nt = xt.shape[0]
            nnz = nt * num['order']
            data, rows, cols = MBRlib.compute_jac(kx, 0, nx, nt, nnz,
                num['order_list'], num['ctrl_list'], t)
            if kx != 0:
                data /= xlimits[kx-1, 1] - xlimits[kx-1, 0]

            rect_mtx = scipy.sparse.csc_matrix((data, (rows, cols)),
                shape=(nt, num['ctrl']))

            mtx = mtx + rect_mtx.T * rect_mtx
            rhs = rect_mtx.T * yt

        diag = sm_options['reg'] * numpy.ones(num['ctrl'])
        arange = numpy.arange(num['ctrl'])
        reg = scipy.sparse.csc_matrix((diag, (arange, arange)))
        mtx = mtx + reg

        sol = numpy.zeros(rhs.shape)
        smt.utils.solve_sparse_system(mtx, rhs, sol, self.solver_options)

        self.sol = sol

    def fit(self):
        """
        Train the model
        """
        filename = '%s.sm' % self.sm_options['name']
        checksum = smt.utils._caching_checksum(self)
        success, data = smt.utils._caching_load(filename, checksum)
        if not success or not self.solver_options['save']:
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
        xlimits = self.sm_options['xlimits']
        num = self.num

        t = numpy.zeros(x.shape)
        for ix in range(nx):
            t[:, ix] = (x[:, ix] - xlimits[ix, 0]) /\
                (xlimits[ix, 1] - xlimits[ix, 0])

        nnz = ne * num['order']
        data, rows, cols = MBRlib.compute_jac(kx, 0, nx, ne, nnz,
            num['order_list'], num['ctrl_list'], t)
        if kx != 0:
            data /= xlimits[kx-1, 1] - xlimits[kx-1, 0]
        rect_mtx = scipy.sparse.csc_matrix((data, (rows, cols)),
            shape=(ne, num['ctrl']))

        return rect_mtx.dot(self.sol)
