"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. Nathalie.bartoli      <nathalie@onera.fr>

"""

from __future__ import division

import numpy as np
import scipy
from sm import SM


class PA2(SM):
  
    """
    Square polynomial approach
    """

    
    def _set_default_options(self):

        ''' 
        Constructor.
        
        Arguments
        ---------
        sm_options : dict
            Model-related options, listed below

        printf_options : dict
            Output printing options, listed below
        '''
      
        sm_options = {
            'name': 'PA2',
        }
        printf_options = {
            'global': True,     # Overriding option to print output            
            'time_eval': True,  # Print evaluation times
            'time_train': True, # Print assembly and solution time summary
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
        
        if 0 in self.training_pts['exact']:
            x = self.training_pts['exact'][0][0]
            y = self.training_pts['exact'][0][1]
            
        X = self.respoSurf(x)
        self.coef = np.dot(np.linalg.inv(np.dot(X.T,X)),(np.dot(X.T,y)))


    def respoSurf(self,x):
      
        """
        Build the response surface of degree 2 

        argument
        -----------
        x : np.ndarray [nt, dim]
            Training points

        Returns
        -------
        M : np.ndarray
            Matrix of the surface
        """
        
        dim = x.shape[1]
        n = x.shape[0]
        n_app = int(scipy.special.binom(dim+2, dim))
        M = np.zeros((n_app,n))
        x = x.T
        M[0,:] = np.ones((1,n))
        for i in xrange(1,dim+1):
            M[i,:] = x[i-1,:]
        for i in xrange(dim+1,2*dim+1):
            M[i,:]=x[i-(dim+1),:]**2
        for i in xrange(dim-1):
            for j in xrange(i+1,dim):
                k = int(2*dim+2+(i)*dim-((i+1)*(i))/2+(j-(i+2)))
                M[k,:] = x[i,:]*x[j,:]
                    
        return M.T

    
    def evaluate(self,x):

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
        
        X = self.respoSurf(x)
        y = np.dot(X,self.coef).T

        return y
