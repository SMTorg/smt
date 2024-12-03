"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. Nathalie.bartoli      <nathalie@onera.fr>

This package is distributed under New BSD license.

TO DO:
- define outputs['sol'] = self.sol
"""

import numpy as np
import pickle
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
    filename = "least_square"

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
        X = self.training_points[None][0][0]
        y = self.training_points[None][0][1]
        self.mod = linear_model.LinearRegression()
        self.mod.fit(X, y)

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
    
    def _save(self, filename=None):
        if filename is None:
            filename = self.filename

        try:
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
                print("model saved")
        except:
            print("Couldn't save the model")

    def _load(self, filename):
        if filename is None:
            return ("file is not found")
        else:
            with open(filename, "rb") as file:
                sm2 = pickle.load(file)
                return sm2
