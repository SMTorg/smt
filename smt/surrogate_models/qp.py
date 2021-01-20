"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. Nathalie.bartoli      <nathalie@onera.fr>

This package is distributed under New BSD license.

TO DO:
- define outputs['sol'] = self.sol
"""
import numpy as np
import scipy
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.caching import cached_operation


class QP(SurrogateModel):

    """
    Square polynomial approach
    """

    name = "QP"

    def _initialize(self):
        super(QP, self)._initialize()
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

        if 0 in self.training_points[None]:
            x = self.training_points[None][0][0]
            y = self.training_points[None][0][1]

        if x.shape[0] < (self.nx + 1) * (self.nx + 2) / 2.0:
            raise Exception(
                "Number of training points should be greater or equal to %d."
                % ((self.nx + 1) * (self.nx + 2) / 2.0)
            )

        X = self._response_surface(x)
        self.coef = np.dot(np.linalg.inv(np.dot(X.T, X)), (np.dot(X.T, y)))

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

    def _response_surface(self, x):
        """
        Build the response surface of degree 2
        argument
        -----------
        x : np.ndarray [nt, nx]
            Training points
        Returns
        -------
        M : np.ndarray
            Matrix of the surface
        """
        dim = self.nx
        n = x.shape[0]
        n_app = int(scipy.special.binom(dim + 2, dim))
        M = np.zeros((n_app, n))
        x = x.T
        M[0, :] = np.ones((1, n))
        for i in range(1, dim + 1):
            M[i, :] = x[i - 1, :]
        for i in range(dim + 1, 2 * dim + 1):
            M[i, :] = x[i - (dim + 1), :] ** 2
        for i in range(dim - 1):
            for j in range(i + 1, dim):
                k = int(2 * dim + 2 + (i) * dim - ((i + 1) * (i)) / 2 + (j - (i + 2)))
                M[k, :] = x[i, :] * x[j, :]
        return M.T

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
        dim = self.nx

        linear_coef = self.coef[1 + kx, :]
        quad_coef = 2 * self.coef[1 + dim + kx, :] * x[:, kx]
        neval = np.size(quad_coef, 0)
        cross_coef = np.zeros(neval)

        for i in range(dim):
            if i > kx:
                k = int(
                    2 * dim + 2 + (kx) * dim - ((kx + 1) * (kx)) / 2 + (i - (kx + 2))
                )
                cross_coef += self.coef[k, :] * x[:, i]
            elif i < kx:
                k = int(2 * dim + 2 + (i) * dim - ((i + 1) * (i)) / 2 + (kx - (i + 2)))
                cross_coef += self.coef[k, :] * x[:, i]

        y = (linear_coef + quad_coef + cross_coef).reshape((x.shape[0], self.ny))
        return y

    def _predict_values(self, x):
        """
        Evaluates the model at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, nx]
            Evaluation point input variable values

        Returns
        -------
        y : np.ndarray [n_evals, ny]
            Evaluation point output variable values
        """
        M = self._response_surface(x)
        y = np.dot(M, self.coef)

        return y
