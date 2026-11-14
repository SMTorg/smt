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
from smt.utils import persistence
from smt.utils.caching import cached_operation
from smt.utils.misc import standardization


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
        supports["jax"] = True

    ############################################################################
    # Model functions
    ############################################################################

    def _new_train(self):
        """
        Train the model
        """
        X = self.training_points[None][0][0]
        y = self.training_points[None][0][1]
        (
            self.X_norma,
            self.y_norma,
            self.X_offset,
            self.y_mean,
            self.X_scale,
            self.y_std,
        ) = standardization(X, y)

        if X.shape[0] < (self.nx + 1) * (self.nx + 2) / 2.0:
            raise Exception(
                "Number of training points should be greater or equal to %d."
                % ((self.nx + 1) * (self.nx + 2) / 2.0)
            )

        X = self._response_surface(self.X_norma)
        self.coef = np.linalg.lstsq(X, self.y_norma, rcond=None)[0]

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
        x = (x - self.X_offset) / self.X_scale
        neval = x.shape[0]
        linear_coef = self.coef[1 + kx, :][np.newaxis, :]
        quad_coef = 2 * self.coef[1 + dim + kx, :][np.newaxis, :] * x[:, kx : kx + 1]
        cross_coef = np.zeros((neval, self.ny))

        for i in range(dim):
            if i > kx:
                k = int(
                    2 * dim + 2 + (kx) * dim - ((kx + 1) * (kx)) / 2 + (i - (kx + 2))
                )
                cross_coef += self.coef[k, :][np.newaxis, :] * x[:, i : i + 1]
            elif i < kx:
                k = int(2 * dim + 2 + (i) * dim - ((i + 1) * (i)) / 2 + (kx - (i + 2)))
                cross_coef += self.coef[k, :][np.newaxis, :] * x[:, i : i + 1]

        y = (linear_coef + quad_coef + cross_coef) * self.y_std / self.X_scale[kx]
        return y.reshape((neval, self.ny))

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
        x = (x - self.X_offset) / self.X_scale
        M = self._response_surface(x)
        y_ = np.dot(M, self.coef)
        y = (self.y_mean + self.y_std * y_).ravel()
        y = y.reshape((x.shape[0], self.ny))
        return y

    def save(self, filename):
        persistence.save(self, filename)

    @staticmethod
    def load(filename):
        return persistence.load(filename)

    def get_jax_predictor(self):
        """
        Return a pure JAX function predict(x) that is end-to-end differentiable.

        Returns
        -------
        predict_fn : Callable
            Differentiable JAX prediction function.
        """
        if not hasattr(self, "coef") or self.coef is None:
            raise RuntimeError("QP model has not been trained yet. Call train() first.")
        import jax
        import jax.numpy as jnp

        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        X_offset = jnp.asarray(self.X_offset, dtype=dtype)
        X_scale = jnp.asarray(self.X_scale, dtype=dtype)
        y_mean = jnp.asarray(self.y_mean, dtype=dtype)
        y_std = jnp.asarray(self.y_std, dtype=dtype)
        qp_coef = jnp.asarray(self.coef, dtype=dtype)
        dim = self.nx
        ny = self.ny

        def predict_fn(x):
            x_in = jnp.asarray(x, dtype=dtype)
            is_1d = x_in.ndim == 1
            x_arr = jnp.atleast_2d(x_in)
            x_norm = (x_arr - X_offset) / X_scale
            parts = [
                jnp.ones((x_norm.shape[0], 1), dtype=dtype),
                x_norm,
                x_norm**2,
            ]
            for i in range(dim - 1):
                for j in range(i + 1, dim):
                    parts.append(x_norm[:, i : i + 1] * x_norm[:, j : j + 1])
            M = jnp.concatenate(parts, axis=1)
            y_norma = jnp.matmul(M, qp_coef)
            y = y_mean + y_std * y_norma
            if is_1d:
                return y[0, 0] if ny == 1 else y[0]
            return y

        return predict_fn
