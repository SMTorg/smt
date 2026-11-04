"""
RMTB extended with a closed-form Bayesian posterior variance.

This package is distributed under New BSD license.
"""

import warnings
from numbers import Integral, Real

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from smt.surrogate_models.rmtb import RMTB

# Below this fraction of n left as residual degrees of freedom
# (dof = n - edf), the Wahba sigma2 estimator (RSS / dof) is close enough
# to exact interpolation that it becomes numerically unreliable.
_WAHBA_MIN_DOF_FRACTION = 0.1


class RMTBUQ(RMTB):
    """
    `RMTB`, with a closed-form Bayesian posterior variance added on top of
    its existing mean prediction.

    `RMTB` fits a regularized, minimal-energy tensor-product B-spline by
    minimizing a penalized least-squares objective. The classical
    Kimeldorf-Wahba correspondence (1970) proves that a penalized fit of
    a smoothing spline is exactly the posterior mean of a specific Gaussian process,
    which gives a closed-form posterior covariance over the fitted spline coefficients:
    `Cov(w) = sigma^2 * mtx^-1`, where `mtx` is the same regularized normal
    matrix `RMTB` already factors to get its mean fit. The posterior
    variance of the predicted output at a point `x` is then
    `Var(x) = sigma^2 * f(x)^T mtx^-1 f(x)`, where `f(x)` is `RMTB`'s own
    B-spline basis row at `x`.

    This is a strictly additive subclass: the mean prediction
    (`predict_values`) is untouched, `mtx` is reused as-is (one extra
    Cholesky factorization of a matrix `RMTB` already builds), and the only
    new per-query cost is one triangular solve against that factor.

    `approx_order` must be 2. `RMTB`'s data-fit term is an Lp loss,
    `sum((Phi @ sol - y) ** approx_order)`, with `approx_order` as its own
    option (default 4); only at `approx_order=2` is this ordinary
    least-squares, the one case where `mtx` is a solution-independent
    quadratic form and the Kimeldorf-Wahba correspondence above actually
    applies. Any other value makes `mtx` a local, solution-dependent
    Hessian of a nonlinear loss with no Bayesian interpretation -- and can
    even fail to factor outright. `RMTBUQ` therefore overrides `RMTB`'s own
    default to 2 and raises if a different value is passed.
    """

    name = "RMTBUQ"

    def _initialize(self):
        super(RMTBUQ, self)._initialize()
        declare = self.options.declare

        declare(
            "sigma2",
            "wahba",
            types=(str, Real),
            desc="Observation noise variance used to scale the posterior "
            "variance. 'wahba' (default) estimates it after training via "
            "the GCV-style RSS/(n - edf) estimator; alternatively pass a "
            "known, non-negative float.",
        )
        # Overrides RMTB/RMTS's own default of 4: the posterior variance
        # formula is only valid for ordinary least-squares (see class
        # docstring). Enforced in `_train`.
        declare(
            "approx_order",
            2,
            types=Integral,
            desc="Exponent in RMTB's own data-fit loss. RMTBUQ requires "
            "exactly 2 (ordinary least-squares) for its closed-form "
            "posterior variance to be valid; training raises ValueError "
            "for any other value.",
        )

        self.supports["variances"] = True

        self.sigma2 = None
        self.edf = None
        self._mtx_chol = None

    def _train(self):
        approx_order = self.options["approx_order"]
        if approx_order != 2:
            raise ValueError(
                f"RMTBUQ requires approx_order=2 (ordinary least-squares); "
                f"got approx_order={approx_order!r}. RMTBUQ's closed-form "
                "posterior variance relies on the Kimeldorf-Wahba "
                "correspondence, which only holds for ordinary "
                "least-squares -- any other approx_order makes RMTB solve "
                "a different, non-quadratic Lp loss, where 'mtx' has no "
                "Bayesian interpretation (and can fail to factor outright). "
                "See the class docstring."
            )

        super(RMTBUQ, self)._train()

        x_train, y_train = self.training_points[None][0]
        n, ny = y_train.shape

        mtx = self.mtx.toarray()
        self._mtx_chol = cho_factor(mtx, lower=True)

        phi, _phi_t, _c = self.full_jac_dict[0]
        phi = phi.toarray()
        # Effective degrees of freedom: tr(phi @ mtx^-1 @ phi^T) ==
        # tr(mtx^-1 @ phi^T @ phi), by the cyclic property of trace --
        # cheaper since phi^T @ phi is ndof x ndof, not n x n.
        self.edf = float(np.trace(cho_solve(self._mtx_chol, phi.T @ phi)))

        sigma2_option = self.options["sigma2"]
        if sigma2_option == "wahba":
            dof = n - self.edf
            if dof < _WAHBA_MIN_DOF_FRACTION * n:
                warnings.warn(
                    f"RMTBUQ: edf={self.edf:.1f} leaves only dof=n-edf={dof:.1f} "
                    f"residual degrees of freedom out of n={n} training points -- "
                    "the Wahba sigma2 estimator (RSS / dof) is unreliable this "
                    "close to exact interpolation. Consider increasing "
                    "'energy_weight' and/or 'regularization_weight', reducing "
                    "'num_ctrl_pts', or passing a known 'sigma2' directly "
                    "instead of 'wahba'."
                )
            resid = y_train - self._predict_values(x_train)
            self.sigma2 = np.sum(resid**2, axis=0) / max(dof, 1e-8)
        else:
            self.sigma2 = np.full(ny, float(sigma2_option))

    def _predict_variances(self, x: np.ndarray) -> np.ndarray:
        """
        Posterior variance `sigma2 * f(x)^T mtx^-1 f(x)`, batched over every
        query point and output column at once: `mtx` (and its Cholesky
        factor) doesn't depend on which output column is being predicted,
        only `sigma2` does, so the expensive part (one triangular solve per
        query point) is shared across outputs.

        Parameters
        ----------
        x : np.ndarray[nt, nx]
            Input values for the prediction points.

        Returns
        -------
        s2 : np.ndarray[nt, ny]
            Posterior variances.
        """
        f = self._compute_prediction_mtx(x, 0)
        f = f.toarray() if hasattr(f, "toarray") else np.asarray(f)
        v = cho_solve(self._mtx_chol, f.T)
        base_var = np.sum(f * v.T, axis=1)
        var = np.outer(base_var, self.sigma2)
        return np.clip(var, 0.0, None)
