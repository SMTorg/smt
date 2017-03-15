"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. Nathalie.bartoli      <nathalie@onera.fr>

"""

from __future__ import division

from sklearn import linear_model
from smt.sm import SM


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

        self.mod = linear_model.LinearRegression()
        self.mod.fit(x,y)

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

        y = self.mod.predict(x)

        return y
