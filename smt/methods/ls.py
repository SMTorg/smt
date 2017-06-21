"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. Nathalie.bartoli      <nathalie@onera.fr>

TO DO:
- define outputs['sol'] = self.sol
- implement the derivative predictions
"""

from __future__ import division

from sklearn import linear_model
from smt.methods.sm import SM
from smt.utils.caching import cached_operation

class LS(SM):

    """
    Least square model.
    This model uses the linear_model.LinearRegression class from scikit-learn.
    Default-parameters from scikit-learn are used herein.
    """

    def _declare_options(self):
        super(LS, self)._declare_options()
        declare = self.options.declare

        declare('name', 'LS', types=str,
                desc='Least squares interpolant')
        declare('data_dir', values=None, types=str,
                desc='Directory for loading / saving cached data; None means do not save or load')

    ############################################################################
    # Model functions
    ############################################################################


    def _new_train(self):
        """
        Train the model
        """
        pts = self.training_points

        if 0 in pts['exact']:
            x = pts['exact'][0][0]
            y = pts['exact'][0][1]

        self.mod = linear_model.LinearRegression()
        self.mod.fit(x,y)

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
        return self.mod.predict(x)

    def _predict(self, x, kx):
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
        if kx == 0:
            y = self._predict_value(x)
            return y
        else:
            raise NotImplementedError
