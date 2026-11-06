"""
RMTB extended with a closed-form Bayesian posterior variance.

This package is distributed under New BSD license.
"""

import warnings
from numbers import Integral, Real

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import minimize_scalar

from smt.surrogate_models.rmtb import RMTB

# Below this fraction of n left as residual degrees of freedom
# (dof = n - edf), the Wahba sigma2 estimator (RSS / dof) is close enough
# to exact interpolation that it becomes numerically unreliable.
_WAHBA_MIN_DOF_FRACTION = 0.1

# How many orders of magnitude of energy_weight to search either side of
# the scale-matched center (see _gcv_select_energy_weight) when
# energy_weight="auto".
_AUTO_ENERGY_WEIGHT_LOG10_HALF_RANGE = 8.0

# Extra slack, as a fraction of n, added on top of safe_dof_fraction when
# _gcv_select_energy_weight enforces its safe-region constraint. The
# constrained GCV optimum typically sits right at that boundary, and the
# search's closed-form edf(energy_weight) (from the one-time generalized
# eigendecomposition) can disagree with RMTBUQ's own real post-fit edf
# (from directly factoring mtx(energy_weight) for the chosen value) by a
# hair -- more on some BLAS/LAPACK backends than others, since the
# eigendecomposition's conditioning is backend-dependent. Without this
# margin, that disagreement can flip a value the search found "safe" into
# one that RMTBUQ's own near-interpolation warning then flags afterward.
_AUTO_ENERGY_WEIGHT_SEARCH_MARGIN = 0.02


def _gcv_select_energy_weight(
    phi,
    h_energy,
    regularization_weight,
    y_train,
    n,
    safe_dof_fraction=_WAHBA_MIN_DOF_FRACTION,
):
    """
    Select `energy_weight` via Generalized Cross-Validation (GCV; Craven &
    Wahba, 1979), restricted to the region where the resulting fit still
    leaves at least `safe_dof_fraction * n` residual degrees of freedom
    (the same line `RMTBUQ`'s own near-interpolation warning already
    enforces -- here as a search constraint rather than a post-hoc check),
    plus a small extra margin (`_AUTO_ENERGY_WEIGHT_SEARCH_MARGIN`) so
    that the choice stays safe even after the small, backend-dependent
    numerical disagreement between this function's closed-form edf and
    `RMTBUQ`'s own real post-fit edf.

    Avoids refitting RMTB at every candidate value: since `mtx(energy_weight)
    = A + energy_weight * B` with `A = phi.T @ phi + regularization_weight * I`
    and `B = h_energy` both independent of `energy_weight`, a single
    generalized eigendecomposition `B v = gamma * A v` (Golub, Heath &
    Wahba, 1979) diagonalizes `mtx` for every `energy_weight` at once:
    `edf`/`RSS`/`GCV` all become closed-form sums over the fixed `gamma`
    and the data projected once onto the eigenbasis.

    Parameters
    ----------
    phi : np.ndarray[n, ndof]
        RMTB's basis matrix at the training points (`full_jac_dict[0][0]`).
    h_energy : np.ndarray[ndof, ndof]
        The energy_weight-independent energy/roughness penalty matrix
        (`_compute_energy_terms()`).
    regularization_weight : float
    y_train : np.ndarray[n, ny]
    n : int
    safe_dof_fraction : float

    Returns
    -------
    float
        The selected `energy_weight`.
    """
    ndof = phi.shape[1]
    a = phi.T @ phi + regularization_weight * np.eye(ndof)
    b = h_energy

    gamma, v = eigh(b, a)  # B v = gamma * A v, normalized so V^T A V = I
    gamma = np.clip(gamma, 0.0, None)  # B is PSD; clip tiny negative numerical noise

    p_diag = np.sum(v * ((phi.T @ phi) @ v), axis=0)  # diag(V^T (Phi^T Phi) V)
    c = v.T @ phi.T @ y_train  # shape (ndof, ny); fixed, independent of energy_weight
    phi_v = phi @ v  # shape (n, ndof); fixed

    def edf_of(log10_ew):
        scale = 1.0 / (1.0 + (10.0**log10_ew) * gamma)
        return float(np.sum(p_diag * scale)), scale

    # A large but finite penalty for the excluded region: scipy's bounded
    # Brent search does arithmetic (differences, ratios) on consecutive
    # function values internally, and a literal `inf` there can produce
    # `inf - inf = nan` and a spurious RuntimeWarning even though the
    # search still converges correctly. Finite keeps that arithmetic clean.
    _unsafe_penalty = 1e300

    def gcv(log10_ew):
        edf, scale = edf_of(log10_ew)
        dof = n - edf
        if dof < (safe_dof_fraction + _AUTO_ENERGY_WEIGHT_SEARCH_MARGIN) * n:
            return _unsafe_penalty  # edf(energy_weight) is non-increasing in
            #                         energy_weight (gamma >= 0), so this
            #                         excludes a single contiguous region,
            #                         not a disconnected one -- safe for a
            #                         bounded 1-D search.
        fitted = phi_v @ (c * scale[:, None])
        rss = float(np.sum((y_train - fitted) ** 2))
        return (rss / (n * y_train.shape[1])) / max(dof / n, 1e-8) ** 2

    # Center the search where the penalty and data-fit terms are comparable
    # in magnitude, then search generously either side -- scale-invariant,
    # unlike a fixed absolute log10 range.
    trace_a = np.trace(a)
    trace_b = np.trace(b)
    log10_center = np.log10(trace_a / trace_b) if trace_b > 0 else 0.0
    bounds = (
        log10_center - _AUTO_ENERGY_WEIGHT_LOG10_HALF_RANGE,
        log10_center + _AUTO_ENERGY_WEIGHT_LOG10_HALF_RANGE,
    )

    # `gcv` is typically but not provably unimodal in log10(energy_weight)
    # over this 16-decade range, and a single bounded Brent search can lock
    # onto the wrong local minimum if numerical noise -- e.g. from a
    # different BLAS/LAPACK backend's `eigh` -- perturbs it near a shallow
    # stretch or the safe-region boundary. A coarse scan first makes the
    # result robust to that: each evaluation here is a cheap closed-form
    # sum, not a refit, so it anchors the Brent polish to the grid cell
    # that actually contains the minimum, rather than trusting Brent's own
    # bracketing alone over the full range.
    n_grid = 321  # ~0.05 dex spacing over the 16-decade range
    grid = np.linspace(bounds[0], bounds[1], n_grid)
    grid_scores = np.array([gcv(g) for g in grid])
    best_idx = int(np.argmin(grid_scores))
    step = (bounds[1] - bounds[0]) / (n_grid - 1)
    polish_bounds = (
        max(bounds[0], grid[best_idx] - step),
        min(bounds[1], grid[best_idx] + step),
    )
    result = minimize_scalar(gcv, bounds=polish_bounds, method="bounded")
    best_log10_ew, best_score = result.x, result.fun
    if grid_scores[best_idx] < best_score:
        best_log10_ew, best_score = grid[best_idx], grid_scores[best_idx]

    if best_score >= _unsafe_penalty:
        raise RuntimeError(
            "RMTBUQ: energy_weight='auto' found no value in the searched "
            f"range (1e{bounds[0]:.0f} to 1e{bounds[1]:.0f}) that leaves at "
            f"least {safe_dof_fraction:.0%} of n as residual degrees of "
            "freedom. Try a smaller 'num_ctrl_pts', or pass a known "
            "'energy_weight' directly."
        )
    return float(10.0**best_log10_ew)


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

    The posterior variance's gradient is also available in closed form,
    via `predict_variance_derivatives`/`predict_variance_gradient`: since
    `mtx` is symmetric and independent of `x`,
    `d(Var)/dx_k = 2 * sigma2 * f(x)^T mtx^-1 (df/dx_k)(x)`, reusing the
    same cached Cholesky factor and `RMTB`'s own basis-derivative
    evaluation (the same one `predict_derivatives` already calls) -- one
    more triangular solve per query, no new factorization.

    `approx_order` must be 2. `RMTB`'s data-fit term is an Lp loss,
    `sum((Phi @ sol - y) ** approx_order)`, with `approx_order` as its own
    option (default 4); only at `approx_order=2` is this ordinary
    least-squares, the one case where `mtx` is a solution-independent
    quadratic form and the Kimeldorf-Wahba correspondence above actually
    applies. Any other value makes `mtx` a local, solution-dependent
    Hessian of a nonlinear loss with no Bayesian interpretation -- and can
    even fail to factor outright. `RMTBUQ` therefore overrides `RMTB`'s own
    default to 2 and raises if a different value is passed.

    `energy_weight="auto"` selects it via Generalized Cross-Validation
    (GCV), restricted to the region where at least
    `_WAHBA_MIN_DOF_FRACTION * n` residual degrees of freedom remain (the
    same line the near-interpolation warning below already enforces, here
    as a search constraint rather than a post-hoc check) -- see
    `_gcv_select_energy_weight` for the derivation. This costs roughly one
    ordinary fit's worth of extra linear algebra (one generalized
    eigendecomposition), not a refit per candidate value.
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
        # Overrides RMTB/RMTS's own declaration to additionally accept the
        # string "auto". Resolved to a concrete float in `_train`, before
        # the real fit -- RMTB itself never sees the string.
        declare(
            "energy_weight",
            1e-4,
            types=(Integral, Real, str),
            desc="Weight of the energy minimization term, or 'auto' to "
            "select it via Generalized Cross-Validation, restricted to "
            "the region the near-interpolation warning below considers "
            "safe (see the class docstring).",
        )

        self.supports["variances"] = True
        self.supports["variance_derivatives"] = True

        self.sigma2 = None
        self.edf = None
        self.energy_weight_ = None
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

        energy_weight_option = self.options["energy_weight"]
        if isinstance(energy_weight_option, str):
            if energy_weight_option != "auto":
                raise ValueError(
                    f"RMTBUQ: energy_weight={energy_weight_option!r} is not "
                    "a recognized string option; pass a non-negative float, "
                    "or 'auto' to select one via GCV."
                )
            x_train, y_train = self.training_points[None][0]
            # Cheap-assembly probe: populates full_dof2coeff/full_jac_dict/
            # the energy matrix without running RMTB's nonlinear solve --
            # verified to have no effect on the real fit that follows.
            self._setup()
            self.full_dof2coeff = self._compute_dof2coeff()
            phi = self._compute_approx_terms()[0][0].toarray()
            h_energy = self._compute_energy_terms().toarray()
            self.energy_weight_ = _gcv_select_energy_weight(
                phi,
                h_energy,
                self.options["regularization_weight"],
                y_train,
                len(x_train),
            )
            self.options["energy_weight"] = self.energy_weight_
        else:
            self.energy_weight_ = float(energy_weight_option)

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

    def _predict_variance_derivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        """
        Derivative of the posterior variance with respect to input
        dimension `kx`: since `mtx` is symmetric and does not depend on
        `x`, differentiating `sigma2 * f(x)^T mtx^-1 f(x)` gives
        `2 * sigma2 * f(x)^T mtx^-1 (df/dx_kx)(x)`. Reuses the same
        Cholesky factor `_predict_variances` already uses, plus `RMTB`'s
        own basis-derivative evaluation (the same one `_predict_derivatives`
        calls) -- one more triangular solve per query point, no new
        `O(m^3)` work.

        Parameters
        ----------
        x : np.ndarray[nt, nx]
            Input values for the prediction points.
        kx : int
            The 0-based index of the input variable with respect to which
            the derivative is desired.

        Returns
        -------
        ds2_dx : np.ndarray[nt, ny]
            Derivatives of the posterior variance.
        """
        f = self._compute_prediction_mtx(x, 0)
        f = f.toarray() if hasattr(f, "toarray") else np.asarray(f)
        df = self._compute_prediction_mtx(x, kx + 1)
        df = df.toarray() if hasattr(df, "toarray") else np.asarray(df)

        v = cho_solve(self._mtx_chol, df.T)
        base_dvar = 2.0 * np.sum(f * v.T, axis=1)
        return np.outer(base_dvar, self.sigma2)
