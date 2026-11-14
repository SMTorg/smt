"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

import numpy as np

from smt.surrogate_models.rbf_idw_backend import idw_jac, make_idw_jax_predictor
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils import persistence
from smt.utils.caching import cached_operation


class IDW(SurrogateModel):
    """
    Inverse distance weighting interpolant
    This model uses the inverse distance between the unknown and training
    points to predeict the unknown point.
    We do not need to fit this model because the response of an unknown point x
    is computed with respect to the distance between x and the training points.
    """

    name = "IDW"

    def _initialize(self):
        super(IDW, self)._initialize()
        declare = self.options.declare
        supports = self.supports

        declare("p", 2.5, types=(int, float), desc="order of distance norm")
        declare(
            "data_dir",
            values=None,
            types=str,
            desc="Directory for loading / saving cached data; None means do not save or load",
        )

        supports["derivatives"] = True
        supports["output_derivatives"] = True

    def _setup(self):
        pass

    def __setstate__(self, state):
        # models pickled before the C++ extension was removed carry an
        # ``idwc`` entry (always None); drop it
        state.pop("idwc", None)
        self.__dict__.update(state)

        self._setup()

    def _jac(self, x, kx=None):
        xt = self.training_points[None][0][0]
        return idw_jac(x, xt, self.options["p"], kx)

    ############################################################################
    # Model functions
    ############################################################################

    def _new_train(self):
        """
        Train the model
        """
        pass

    def _train(self):
        """
        Train the model
        """
        self._setup()

        inputs = {"self": self}
        with cached_operation(inputs, self.options["data_dir"]) as outputs:
            if outputs:
                self.sol = outputs["sol"]
            else:
                self._new_train()
                # outputs['sol'] = self.sol

    def _predict_values(self, x):
        """
        This function is used by _predict function. See _predict for more details.
        """
        yt = self.training_points[None][0][1]

        jac = self._jac(x)

        y = jac.dot(yt)
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
        dy_dx : np.ndarray
            Derivative values.
        """
        yt = self.training_points[None][0][1]

        jac = self._jac(x, kx)

        dy_dx = jac.dot(yt)
        return dy_dx

    def _predict_output_derivatives(self, x):
        ny = self.training_points[None][0][1].shape[1]

        jac = self._jac(x)
        jac = np.einsum("ij,k->ijk", jac, np.ones(ny))

        dy_dyt = {None: jac}
        return dy_dyt

    def get_jax_predictor(self):
        """Return a pure JAX differentiable callable ``predict(x)`` for IDW."""
        if (
            not self.training_points
            or None not in self.training_points
            or 0 not in self.training_points[None]
        ):
            raise RuntimeError(
                "IDW model has not been trained yet. Call set_training_values() and train() first."
            )
        xt, yt = self.training_points[None][0]
        return make_idw_jax_predictor(xt, yt, self.options["p"])

    def save(self, filename):
        persistence.save(self, filename)

    @staticmethod
    def load(filename):
        return persistence.load(filename)
