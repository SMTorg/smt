'''
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>
        
This package is distributed under BSD license
'''

from __future__ import division
import warnings
import numpy as np

from smt.methods.krg_based import KRG_BASED
from smt.utils.kriging_utils import componentwise_distance_PLS, compute_pls

"""
The KPLS class.
"""

class KPLS(KRG_BASED):

    """
    - KPLS
    """
    def _initialize(self):
        super(KPLS, self)._initialize()
        declare = self.options.declare
        declare('n_comp', 1, types=int, desc='Number of principal components')
        declare('theta0', [1e-2], types=(list, np.ndarray), desc='Initial hyperparameters')
        self.name = 'KPLS'

    def _compute_pls(self,X,y):
        self.coeff_pls = compute_pls(X.copy(),y.copy(),self.options['n_comp'])
        return X,y

    def _componentwise_distance(self,dx,opt=0):
        d = componentwise_distance_PLS(dx,self.options['corr'].__name__,
                                       self.options['n_comp'],self.coeff_pls)
        return d
