"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""
import numpy as np
from scipy.sparse import csc_matrix
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.caching import cached_operation

from smt.surrogate_models.idwclib import PyIDW


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
        xt = self.training_points[None][0][0]
        nt = xt.shape[0]
        nx = xt.shape[1]

        self.idwc = PyIDW()
        self.idwc.setup(nx, nt, self.options["p"], xt.flatten())

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

        tmp = self.idwc
        self.idwc = None

        inputs = {"self": self}
        with cached_operation(inputs, self.options["data_dir"]) as outputs:
            self.idwc = tmp

            if outputs:
                self.sol = outputs["sol"]
            else:
                self._new_train()
                # outputs['sol'] = self.sol

    def _predict_values(self, x):
        """
        This function is used by _predict function. See _predict for more details.
        """
        n = x.shape[0]
        nt = self.nt

        yt = self.training_points[None][0][1]

        jac = np.empty(n * nt)
        self.idwc.compute_jac(n, x.flatten(), jac)
        jac = jac.reshape((n, nt))

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
        n = x.shape[0]
        nt = self.nt

        yt = self.training_points[None][0][1]

        jac = np.empty(n * nt)
        self.idwc.compute_jac_derivs(n, kx, x.flatten(), jac)
        jac = jac.reshape((n, nt))

        dy_dx = jac.dot(yt)
        return dy_dx

    def _predict_output_derivatives(self, x):
        n = x.shape[0]
        nt = self.nt
        ny = self.training_points[None][0][1].shape[1]

        jac = np.empty(n * nt)
        self.idwc.compute_jac(n, x.flatten(), jac)
        jac = jac.reshape((n, nt))
        jac = np.einsum("ij,k->ijk", jac, np.ones(ny))

        dy_dyt = {None: jac}
        return dy_dyt
