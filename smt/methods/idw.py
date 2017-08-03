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

from smt.methods import IDWlib


class IDW(SM):

    '''
    Inverse distance weighting interpolant
    This model uses the inverse distance between the unknown and training
    points to predeict the unknown point.
    We do not need to fit this model because the response of an unknown point x
    is computed with respect to the distance between x and the training points.
    '''

    def initialize(self):
        super(IDW, self).initialize()
        declare = self.options.declare

        declare('p', 2.5, types=(int, float), desc='order of distance norm')
        declare('data_dir', values=None, types=str,
                desc='Directory for loading / saving cached data; None means do not save or load')

        self.name = 'IDW'

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
        inputs = {'self': self}
        with cached_operation(inputs, self.options['data_dir']) as outputs:
            if outputs:
                self.sol = outputs['sol']
            else:
                self._new_train()
                #outputs['sol'] = self.sol

    def _predict_value(self,x):
        """
        This function is used by _predict function. See _predict for more details.
        """
        n_evals = x.shape[0]
        xt_list = []
        yt_list = []
        if 0 in self.training_points[None]:
            xt_list.append(self.training_points[None][0][0])
            yt_list.append(self.training_points[None][0][1])

        xt = np.vstack(xt_list)
        yt = np.vstack(yt_list)

        mtx = IDWlib.compute_jac(self.dim, n_evals, self.nt, self.options['p'], x, xt)

        y = mtx.dot(yt)
        return y
