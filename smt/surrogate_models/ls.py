"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. Nathalie.bartoli      <nathalie@onera.fr>

This package is distributed under New BSD license.

TO DO:
- define outputs['sol'] = self.sol
"""
import numpy as np

from sklearn import linear_model
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.caching import cached_operation


class LS(SurrogateModel):

    """
    Least square model.
    This model uses the linear_model.LinearRegression class from scikit-learn.
    Default-parameters from scikit-learn are used herein.
    """

    name = "LS"

    def _initialize(self):
        super(LS, self)._initialize()
        declare = self.options.declare
        supports = self.supports

        declare(
            "data_dir",
            values=None,
            types=str,
            desc="Directory for loading / saving cached data; None means do not save or load",
        )

        supports["derivatives"] = True

    ############################################################################
    # Model functions
    ############################################################################

    def _new_train(self):
        """
        Train the model
        """
        pts = self.training_points

        if 0 in pts[None]:
            x = pts[None][0][0]
            y = pts[None][0][1]

        self.mod = linear_model.LinearRegression()
        self.mod.fit(x, y)

    def _train(self):
        """
        Train the model
        """
        inputs = {"self": self}
        with cached_operation(inputs, self.options["data_dir"]) as outputs:
            if outputs:
                self.sol = outputs["sol"]
            else:
                self._new_train()
                # outputs['sol'] = self.sol

    def _predict_values(self, x):
        """
        Evaluates the model at a set of points.

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
        y : np.ndarray
            Derivative values.
        """

        # Initialization
        n_eval, n_features_x = x.shape
        y = np.ones((n_eval, self.ny)) * self.mod.coef_[:, kx]

        return y
