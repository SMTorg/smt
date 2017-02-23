"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang         <hwangjt@umich.edu>

"""

from __future__ import division

import numpy as np
from scipy.sparse import csc_matrix
from sm import SM
import IDWlib

class IDW(SM):

    '''
    Inverse distance weighting interpolant
    This model uses the inverse distance between the unknown and training
    points to predeict the unknown point.
    We do not need to fit this model because the response of an unknown point x
    is computed with respect to the distance between x and the training points.
    '''

    def _set_default_options(self):
        sm_options = {
            'name': 'IDW',
            'p': 2.5,           # Parameter p
        }
        printf_options = {
            'global': True,     # Overriding option to print output
            'time_eval': True,  # Print evaluation times
            'time_train': False, # Print assembly and solution time summary
            'problem': True,    # Print problem information
        }

        self.sm_options = sm_options
        self.printf_options = printf_options


    ############################################################################
    # Model functions
    ############################################################################


    def fit(self):

        """
        Train the model
        """
        pass


    def evaluate(self, x):

        """
        Evaluates the model at a set of unknown points

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        y : np.ndarray
            Evaluation point output variable values
        """

        n_evals = x.shape[0]
        xt_list = []
        yt_list = []
        if 0 in self.training_pts['exact']:
            xt_list.append(self.training_pts['exact'][0][0])
            yt_list.append(self.training_pts['exact'][0][1])

        xt = np.vstack(xt_list)
        yt = np.vstack(yt_list)

        mtx = IDWlib.compute_jac(self.dim, n_evals, self.nt, self.sm_options['p'], x, xt)

        return mtx.dot(yt)
