"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. Nathalie.bartoli      <nathalie@onera.fr>

"""

from __future__ import division

from sklearn import linear_model
from sm import SM


class LS(SM):
  
    """
    Least square model.
    This model uses the linear_model.LinearRegression class from scikit-learn.
    Default-parameters from scikit-learn are used herein.
    """
    
    def _set_default_options(self):
      
        ''' 
        Constructor.
        
        Arguments
        ---------
        sm_options : dict
            Model-related options, listed below
        solver_options : dict

        printf_options : dict
            Output printing options, listed below
        '''
        
        sm_options = {
            'name': 'LS',          
        }
        solver_options = {      # Setting given by default by scikit-learn
        }
        printf_options = {
            'global': True,     # Overriding option to print output            
            'time_eval': True,  # Print evaluation times
            'time_train': True, # Print training time
            'problem': True,    # Print problem information
        }
      
        self.sm_options = sm_options
        self.solver_options = solver_options
        self.printf_options = printf_options
      
        self.mod = linear_model.LinearRegression()


    ############################################################################
    # Model functions
    ############################################################################

        
    def fit(self):
      
        """
        Train the model    
        """
        
        pts = self.training_pts
        
        if 0 in pts['exact']:
            x = pts['exact'][0][0]
            y = pts['exact'][0][1]
        
        self.mod.fit(x,y)

        
    def evaluate(self,x):
      
        """
        Evaluate the model at a set of unknown points

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        y : np.ndarray 
            Evaluation point output variable values
        """
        
        y = self.mod.predict(x)
        
        return y
