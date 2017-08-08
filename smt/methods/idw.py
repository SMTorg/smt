"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang         <hwangjt@umich.edu>

TO DO:
- implement the derivative predictions
"""

from __future__ import division

import numpy as np
from scipy.sparse import csc_matrix
from smt.methods.sm import SM
from smt.utils.caching import cached_operation

from smt.methods.idwclib import PyIDW


class IDW(SM):

    """
    Inverse distance weighting interpolant
    This model uses the inverse distance between the unknown and training
    points to predeict the unknown point.
    We do not need to fit this model because the response of an unknown point x
    is computed with respect to the distance between x and the training points.
    """

    def _initialize(self):
        super(IDW, self)._initialize()
        declare = self.options.declare
        supports = self.supports

        declare('p', 2.5, types=(int, float), desc='order of distance norm')
        declare('data_dir', values=None, types=str,
                desc='Directory for loading / saving cached data; None means do not save or load')

        supports['derivatives'] = True
        supports['output_derivatives'] = True

        self.name = 'IDW'

    def _setup(self):
        xt = self.training_points[None][0][0]
        nt = xt.shape[0]
        nx = xt.shape[1]

        self.idwc = PyIDW()
        self.idwc.setup(nx, nt, self.options['p'], xt.flatten())

    ############################################################################
    # Model functions
    ############################################################################

    def _new_train(self):
        """
        Train the model
        """
        pass

    def _train(self):
        """
        Train the model
        """
        self._setup()

        tmp = self.idwc
        self.idwc = None

        inputs = {'self': self}
        with cached_operation(inputs, self.options['data_dir']) as outputs:
            self.idwc = tmp

            if outputs:
                self.sol = outputs['sol']
            else:
                self._new_train()
                #outputs['sol'] = self.sol

    def _predict_values(self,x):
        """
        This function is used by _predict function. See _predict for more details.
        """
        n = x.shape[0]
        nt = self.nt

        yt = self.training_points[None][0][1]

        jac = np.empty(n * nt)
        self.idwc.compute_jac(n, x.flatten(), jac)
        jac = jac.reshape((n, nt))

        y = jac.dot(yt)
        return y

    def _predict_derivatives(self, x, kx):
        """
        Evaluates the derivatives at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values
        kx : int
            The 0-based index of the input variable with respect to which derivatives are desired.

        Returns
        -------
        dy_dx : np.ndarray
            Derivative values.
        """
        n = x.shape[0]
        nt = self.nt

        yt = self.training_points[None][0][1]

        jac = np.empty(n * nt)
        self.idwc.compute_jac_derivs(n, kx, x.flatten(), jac)
        jac = jac.reshape((n, nt))

        dy_dx = jac.dot(yt)
        return dy_dx

    def _predict_output_derivatives(self, x):
        n = x.shape[0]
        nt = self.nt

        jac = np.empty(n * nt)
        self.idwc.compute_jac(n, x.flatten(), jac)
        jac = jac.reshape((n, nt))

        dy_dyt = {None: jac}
        return dy_dyt
