"""
Functional JAX predictors for Kriging-based surrogate models (KRG, KPLS, KPLSK, GEKPLS, MFK).

Provides pure, JIT-compilable, and end-to-end autodiff-compatible JAX prediction functions.
"""

from functools import lru_cache
from typing import Callable
import numpy as np


@lru_cache(maxsize=1)
def _get_matern_kernels():
    """Return Matérn 3/2 and 5/2 1D kernels with exact C2-smooth custom JVP rules."""
    import jax
    import jax.numpy as jnp

    @jax.custom_jvp
    def matern32_1d(z):
        az = jnp.abs(z)
        s3 = jnp.sqrt(3.0)
        return (1.0 + s3 * az) * jnp.exp(-s3 * az)

    @matern32_1d.defjvp
    def _matern32_1d_jvp(primals, tangents):
        (z,) = primals
        (z_dot,) = tangents
        az = jnp.abs(z)
        s3 = jnp.sqrt(3.0)
        exp_val = jnp.exp(-s3 * az)
        primal_out = (1.0 + s3 * az) * exp_val
        tangent_out = (-3.0 * z * exp_val) * z_dot
        return primal_out, tangent_out

    @jax.custom_jvp
    def matern52_1d(z):
        az = jnp.abs(z)
        s5 = jnp.sqrt(5.0)
        return (1.0 + s5 * az + (5.0 / 3.0) * (z**2)) * jnp.exp(-s5 * az)

    @matern52_1d.defjvp
    def _matern52_1d_jvp(primals, tangents):
        (z,) = primals
        (z_dot,) = tangents
        az = jnp.abs(z)
        s5 = jnp.sqrt(5.0)
        exp_val = jnp.exp(-s5 * az)
        primal_out = (1.0 + s5 * az + (5.0 / 3.0) * (z**2)) * exp_val
        tangent_out = (-(5.0 / 3.0) * z * (1.0 + s5 * az) * exp_val) * z_dot
        return primal_out, tangent_out

    return matern32_1d, matern52_1d


def eval_poly_jax(poly_name: str, x_cont, nx: int):
    """
    Evaluate polynomial trend in pure JAX.

    Parameters
    ----------
    poly_name : str
        'constant', 'linear', or 'quadratic'.
    x_cont : jax.Array of shape (n, nx)
        Normalized continuous evaluation coordinates.
    nx : int
        Number of input dimensions.

    Returns
    -------
    f : jax.Array of shape (n, p)
        Polynomial basis matrix.
    """
    import jax.numpy as jnp

    n = x_cont.shape[0]
    dt = x_cont.dtype
    if poly_name == "constant":
        return jnp.ones((n, 1), dtype=dt)
    elif poly_name == "linear":
        return jnp.concatenate([jnp.ones((n, 1), dtype=dt), x_cont], axis=1)
    elif poly_name == "quadratic":
        parts = [jnp.ones((n, 1), dtype=dt), x_cont]
        for k in range(nx):
            parts.append(x_cont[:, k : k + 1] * x_cont[:, k:])
        return jnp.concatenate(parts, axis=1)
    else:
        raise NotImplementedError(
            f"Polynomial trend {poly_name!r} is not implemented in JAX predictor."
        )


def eval_corr_jax(corr_name: str, dx, theta, power: float = 2.0, coeff_pls=None):
    """
    Evaluate correlation kernel matrix in pure JAX.

    Parameters
    ----------
    corr_name : str
        Correlation function name ('squar_exp', 'abs_exp', 'pow_exp', 'matern32', 'matern52').
    dx : jax.Array of shape (n_eval, nt, nx)
        Pairwise coordinate differences between evaluation points and training points.
    theta : jax.Array of shape (n_comp,) or (nx,)
        Correlation length scale parameters.
    power : float, optional
        Power exponent for pow_exp (default 2.0).
    coeff_pls : jax.Array of shape (nx, n_comp), optional
        PLS projection matrix if dimension reduction is used.

    Returns
    -------
    r : jax.Array of shape (n_eval, nt)
        Correlation matrix.
    """
    import jax.numpy as jnp

    if coeff_pls is not None:
        if corr_name == "squar_exp":
            d = jnp.matmul(dx**2, coeff_pls**2)
        elif corr_name == "pow_exp":
            d = jnp.matmul(jnp.abs(dx) ** power, jnp.abs(coeff_pls) ** power)
        else:
            d = jnp.matmul(jnp.abs(dx), jnp.abs(coeff_pls))
    else:
        if corr_name == "squar_exp":
            d = dx**2
        elif corr_name == "pow_exp":
            d = jnp.abs(dx) ** power
        else:
            d = jnp.abs(dx)

    if corr_name in ["squar_exp", "abs_exp", "pow_exp"]:
        return jnp.exp(-jnp.sum(theta * d, axis=-1))
    elif corr_name == "matern32":
        if coeff_pls is None:
            matern32_1d, _ = _get_matern_kernels()
            return jnp.prod(matern32_1d(theta * dx), axis=-1)
        ll = theta * d
        return jnp.prod(1.0 + jnp.sqrt(3.0) * ll, axis=-1) * jnp.exp(
            -jnp.sqrt(3.0) * jnp.sum(ll, axis=-1)
        )
    elif corr_name == "matern52":
        if coeff_pls is None:
            _, matern52_1d = _get_matern_kernels()
            return jnp.prod(matern52_1d(theta * dx), axis=-1)
        ll = theta * d
        return jnp.prod(
            1.0 + jnp.sqrt(5.0) * ll + (5.0 / 3.0) * (ll**2), axis=-1
        ) * jnp.exp(-jnp.sqrt(5.0) * jnp.sum(ll, axis=-1))
    else:
        raise NotImplementedError(
            f"Correlation kernel {corr_name!r} is not implemented in JAX predictor."
        )


def make_krg_jax_predictor(sm, return_variance: bool = False) -> Callable:
    """
    Build a pure JAX predictor function for a trained Kriging model.

    Supports single-point ``(nx,)`` or batched ``(n, nx)`` inputs and automatic
    differentiation via ``jax.grad`` and ``jax.jacobian``.

    Parameters
    ----------
    sm : KrgBased
        Trained Kriging-based surrogate model instance.
    return_variance : bool, optional
        If True, the predictor returns ``(y, s2)``. Default is False.

    Returns
    -------
    predict_fn : Callable
        Differentiable JAX prediction function.
    """
    if not hasattr(sm, "optimal_par") or sm.optimal_par is None:
        raise RuntimeError(
            f"{sm.__class__.__name__} model has not been trained yet. Call train() first."
        )
    if not getattr(sm, "is_continuous", True):
        raise NotImplementedError(
            "get_jax_predictor is currently supported for continuous design spaces."
        )

    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    X_train_norma = jnp.asarray(sm.X_norma, dtype=dtype)
    X_offset = jnp.asarray(sm.X_offset, dtype=dtype)
    X_scale = jnp.asarray(sm.X_scale, dtype=dtype)
    y_mean = jnp.asarray(sm.y_mean, dtype=dtype)
    y_std = jnp.asarray(sm.y_std, dtype=dtype)
    beta = jnp.asarray(sm.optimal_par["beta"], dtype=dtype)
    gamma = jnp.asarray(sm.optimal_par["gamma"], dtype=dtype)
    theta = jnp.asarray(sm.optimal_theta, dtype=dtype)

    has_pls = getattr(sm, "coeff_pls", None) is not None and not getattr(
        sm, "kplsk_second_loop", False
    )
    coeff_pls = jnp.asarray(sm.coeff_pls, dtype=dtype) if has_pls else None

    corr_name = sm.options["corr"]
    poly_name = sm.options["poly"]
    power = float(getattr(sm, "_pow_exp_power", 2.0))
    nx = sm.nx
    ny = sm.ny

    if return_variance:
        C = jnp.asarray(sm.optimal_par["C"], dtype=dtype)
        G = jnp.asarray(sm.optimal_par["G"], dtype=dtype)
        Ft = jnp.asarray(sm.optimal_par["Ft"], dtype=dtype)
        is_noisy = np.max(sm._noise0) > 0.0 or sm._eval_noise
        if is_noisy and sm.options["is_ri"]:
            A = jnp.asarray(sm.optimal_par["sigma2_ri"], dtype=dtype)
        else:
            A = jnp.asarray(sm.optimal_par["sigma2"], dtype=dtype)

    def predict_fn(x):
        x_in = jnp.asarray(x, dtype=dtype)
        is_1d = x_in.ndim == 1
        x_arr = jnp.atleast_2d(x_in)
        x_cont = (x_arr - X_offset) / X_scale
        dx = jnp.expand_dims(x_cont, 1) - jnp.expand_dims(X_train_norma, 0)

        r = eval_corr_jax(corr_name, dx, theta, power, coeff_pls)
        f = eval_poly_jax(poly_name, x_cont, nx)

        y_norma = jnp.matmul(f, beta) + jnp.matmul(r, gamma)
        y = y_mean + y_std * y_norma

        if return_variance:
            rt = solve_triangular(C, r.T, lower=True)
            u = solve_triangular(G.T, jnp.matmul(Ft.T, rt) - f.T, lower=True)
            B = 1.0 - jnp.sum(rt**2, axis=0) + jnp.sum(u**2, axis=0)
            s2 = jnp.maximum(0.0, jnp.outer(B, A))
            if is_1d:
                y_ret = y[0, 0] if ny == 1 else y[0]
                s2_ret = s2[0, 0] if ny == 1 else s2[0]
                return y_ret, s2_ret
            return y, s2

        if is_1d:
            return y[0, 0] if ny == 1 else y[0]
        return y

    return predict_fn


def make_mfk_jax_predictor(sm) -> Callable:
    """
    Build a pure JAX predictor function for a trained Multi-Fidelity Kriging (MFK) model.

    Parameters
    ----------
    sm : MFK
        Trained MFK surrogate model instance.

    Returns
    -------
    predict_fn : Callable
        Differentiable JAX prediction function.
    """
    if not hasattr(sm, "optimal_par") or sm.optimal_par is None:
        raise RuntimeError("MFK model has not been trained yet. Call train() first.")
    if not getattr(sm, "is_continuous", True):
        raise NotImplementedError(
            "get_jax_predictor is currently supported for continuous design spaces."
        )

    import jax
    import jax.numpy as jnp

    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    nlvl = sm.nlvl
    X_offset = jnp.asarray(sm.X_offset, dtype=dtype)
    X_scale = jnp.asarray(sm.X_scale, dtype=dtype)
    y_mean = jnp.asarray(sm.y_mean, dtype=dtype)
    y_std = jnp.asarray(sm.y_std, dtype=dtype)

    X_norma_all = [jnp.asarray(sm.X_norma_all[i], dtype=dtype) for i in range(nlvl)]
    thetas = [jnp.asarray(sm.optimal_theta[i], dtype=dtype) for i in range(nlvl)]
    betas = [jnp.asarray(sm.optimal_par[i]["beta"], dtype=dtype) for i in range(nlvl)]
    gammas = [jnp.asarray(sm.optimal_par[i]["gamma"], dtype=dtype) for i in range(nlvl)]

    has_pls = getattr(sm, "coeff_pls", None) is not None and not getattr(
        sm, "kplsk_second_loop", False
    )
    coeff_pls = jnp.asarray(sm.coeff_pls, dtype=dtype) if has_pls else None

    corr_name = sm.options["corr"]
    poly_name = sm.options["poly"]
    rho_regr = sm.options["rho_regr"]
    power = float(getattr(sm, "_pow_exp_power", 2.0))
    nx = sm.nx
    ny = betas[-1].shape[1]

    def predict_fn(x):
        x_in = jnp.asarray(x, dtype=dtype)
        is_1d = x_in.ndim == 1
        x_arr = jnp.atleast_2d(x_in)
        x_cont = (x_arr - X_offset) / X_scale

        f0 = eval_poly_jax(poly_name, x_cont, nx)
        dx0 = jnp.expand_dims(x_cont, 1) - jnp.expand_dims(X_norma_all[0], 0)
        r0 = eval_corr_jax(corr_name, dx0, thetas[0], power, coeff_pls)
        mu_prev = jnp.matmul(f0, betas[0]) + jnp.matmul(r0, gammas[0])

        for i in range(1, nlvl):
            g = eval_poly_jax(rho_regr, x_cont, nx)
            dx_i = jnp.expand_dims(x_cont, 1) - jnp.expand_dims(X_norma_all[i], 0)
            r_i = eval_corr_jax(corr_name, dx_i, thetas[i], power, coeff_pls)
            f_i = jnp.concatenate([g * mu_prev, f0], axis=1)
            mu_curr = jnp.matmul(f_i, betas[i]) + jnp.matmul(r_i, gammas[i])
            mu_prev = mu_curr

        y = y_mean + y_std * mu_prev
        if is_1d:
            return y[0, 0] if ny == 1 else y[0]
        return y

    return predict_fn
