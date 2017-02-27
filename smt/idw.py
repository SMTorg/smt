"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang         <hwangjt@umich.edu>

"""

from __future__ import division

import numpy as np
from scipy.sparse import csc_matrix
from smt.sm import SM
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
