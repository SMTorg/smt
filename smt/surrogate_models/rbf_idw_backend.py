"""
Pure-Python kernels for the RBF and IDW surrogate models.

These replace the former C++/Cython extensions (``rbfclib`` / ``idwclib``).

Engines
-------
* NumPy (default): vectorized (pairwise distances via ``scipy.spatial.distance.cdist``),
  no optional dependency.
* Numba (optional): enabled with the environment variable ``USE_NUMBA_JIT=1``
  (same switch as the RMTS backend). The kernels are parallel and cached on
  disk. When Numba is not installed the NumPy engine is used silently.
* JAX (optional): enabled with ``USE_JAX=1`` (for IDW kernels with scoped fp64)
  and via ``IDW.get_jax_predictor()`` for differentiable JAX pipelines.

The public model APIs (options, outputs, shapes) are unchanged.
"""

import contextlib
import os

import numpy as np
from scipy.spatial.distance import cdist


def _parse_bool_env(name, default=0):
    val = os.getenv(name)
    if val is None:
        return bool(default)
    norm = val.strip().lower()
    if norm in ("", "0", "false", "no", "off"):
        return False
    if norm in ("1", "true", "yes", "on"):
        return True
    try:
        return bool(int(norm))
    except ValueError:
        return bool(default)


_NUMBA_KERNELS = None
_JAX_IDW_KERNELS = None


def _get_numba():
    try:
        from numba import njit
        from numba import prange as nb_prange
    except ImportError:
        return None, None
    return njit, nb_prange


def has_numba():
    """Return True if Numba is importable."""
    return _get_numba()[0] is not None


def _get_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return None, None
    return jax, jnp


def has_jax():
    """Return True if JAX is importable."""
    return _get_jax()[0] is not None


def _enable_x64(jax):
    """Return a context manager temporarily enabling JAX 64-bit precision."""
    if bool(jax.config.jax_enable_x64):
        return contextlib.nullcontext()
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    return enable(True)


def _resolve_backend(_backend=None):
    if _backend is None or _backend == "auto":
        if _parse_bool_env("USE_JAX", 0) and has_jax():
            return "jax"
        if _parse_bool_env("USE_NUMBA_JIT", 0) and has_numba():
            return "numba"
        return "numpy"
    if _backend == "jax":
        return "jax" if has_jax() else "numpy"
    if _backend == "numba":
        return "numba" if has_numba() else "numpy"
    if _backend == "numpy":
        return "numpy"
    raise ValueError(
        f"Unsupported backend {_backend!r}; expected one of (None, 'auto', 'numpy', 'numba', 'jax')."
    )


# ======================================================================
# NumPy engine
# ======================================================================


def _rbf_np(x, xt, d0, poly_degree, kx):
    n, nx = x.shape
    nt = xt.shape[0]
    num_poly = {-1: 0, 0: 1}.get(poly_degree, 1 + nx)
    jac = np.empty((n, nt + num_poly))

    r2 = cdist(x / d0, xt / d0, "sqeuclidean")
    rad = np.exp(-r2)
    if kx is None:
        jac[:, :nt] = rad
    else:
        d = x[:, kx, None] - xt[None, :, kx]
        jac[:, :nt] = -rad * (2.0 * d / d0[kx] ** 2)

    if poly_degree >= 0:
        jac[:, nt] = 1.0 if kx is None else 0.0
    if poly_degree == 1:
        if kx is None:
            jac[:, nt + 1 :] = x
        else:
            jac[:, nt + 1 :] = 0.0
            jac[:, nt + 1 + kx] = 1.0
    return jac


def _idw_np(x, xt, p, kx):
    # sum of squared differences: exactly 0 iff x coincides with a training point
    r2 = cdist(x, xt, "sqeuclidean")

    hit = r2 == 0.0  # evaluation point coincides with a training point
    exact = hit.any(axis=1)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        w = r2 ** (-p / 2.0)
        s = w.sum(axis=1, keepdims=True)
        underflow = (~exact) & (s[:, 0] == 0.0)
        if kx is None:
            jac = w / s
        else:
            dr2 = 2.0 * (x[:, kx, None] - xt[None, :, kx])
            dw = -p / 2.0 * r2 ** (-p / 2.0 - 1.0) * dr2
            ds = dw.sum(axis=1, keepdims=True)
            jac = (dw * s - w * ds) / s**2

    if exact.any():
        if kx is None:
            jac[exact] = 0.0
            first = hit[exact].argmax(axis=1)  # first coincident training point
            jac[np.flatnonzero(exact), first] = 1.0
        else:
            jac[exact] = 0.0
    if underflow.any():
        if kx is None:
            jac[underflow] = 0.0
            nearest = r2[underflow].argmin(axis=1)
            jac[np.flatnonzero(underflow), nearest] = 1.0
        else:
            jac[underflow] = 0.0
    return jac


# ======================================================================
# Numba engine
# ======================================================================


def _make_numba_kernels(njit, prange):
    def rbf_kernel(x, xt, d0, poly_degree, kx, deriv, jac):
        n, nx = x.shape
        nt = xt.shape[0]
        for i in prange(n):
            for it in range(nt):
                r2 = 0.0
                for ix in range(nx):
                    d = (x[i, ix] - xt[it, ix]) / d0[ix]
                    r2 += d * d
                if deriv:
                    dd = x[i, kx] - xt[it, kx]
                    jac[i, it] = -np.exp(-r2) * (2.0 * dd / (d0[kx] * d0[kx]))
                else:
                    jac[i, it] = np.exp(-r2)
            if poly_degree >= 0:
                jac[i, nt] = 0.0 if deriv else 1.0
            if poly_degree == 1:
                for ix in range(nx):
                    if deriv:
                        jac[i, nt + 1 + ix] = 1.0 if ix == kx else 0.0
                    else:
                        jac[i, nt + 1 + ix] = x[i, ix]

    def idw_kernel(x, xt, p, kx, deriv, jac):
        n, nx = x.shape
        nt = xt.shape[0]
        for i in prange(n):
            min_loc = -1
            nearest_it = 0
            min_r2 = np.inf
            s = 0.0
            ds = 0.0
            for it in range(nt):
                r2 = 0.0
                for ix in range(nx):
                    d = x[i, ix] - xt[it, ix]
                    r2 += d * d
                if r2 < min_r2:
                    min_r2 = r2
                    nearest_it = it
                if r2 == 0.0:
                    min_loc = it
                    break
                w = r2 ** (-p / 2.0)
                s += w
                jac[i, it] = w
                if deriv:
                    dw = (
                        -p
                        / 2.0
                        * r2 ** (-p / 2.0 - 1.0)
                        * 2.0
                        * (x[i, kx] - xt[it, kx])
                    )
                    ds += dw
            if min_loc >= 0:
                for it in range(nt):
                    jac[i, it] = 0.0
                if not deriv:
                    jac[i, min_loc] = 1.0
            elif s == 0.0:
                for it in range(nt):
                    jac[i, it] = 0.0
                if not deriv:
                    jac[i, nearest_it] = 1.0
            elif deriv:
                for it in range(nt):
                    r2 = 0.0
                    for ix in range(nx):
                        d = x[i, ix] - xt[it, ix]
                        r2 += d * d
                    w = jac[i, it]
                    dw = (
                        -p
                        / 2.0
                        * r2 ** (-p / 2.0 - 1.0)
                        * 2.0
                        * (x[i, kx] - xt[it, kx])
                    )
                    jac[i, it] = (dw * s - w * ds) / (s * s)
            else:
                for it in range(nt):
                    jac[i, it] = jac[i, it] / s

    return (
        njit(cache=True, parallel=True)(rbf_kernel),
        njit(cache=True, parallel=True)(idw_kernel),
    )


def _get_numba_kernels():
    global _NUMBA_KERNELS
    if _NUMBA_KERNELS is None:
        njit, nb_prange = _get_numba()
        _NUMBA_KERNELS = _make_numba_kernels(njit, nb_prange)
    return _NUMBA_KERNELS


# ======================================================================
# JAX engine (IDW)
# ======================================================================


def _idw_jax_terms(jax, jnp, x_arr, xt_arr, p):
    diff = jnp.expand_dims(x_arr, 1) - jnp.expand_dims(xt_arr, 0)
    r2 = jnp.sum(diff**2, axis=-1)
    is_zero = r2 == 0.0
    has_zero = jnp.any(is_zero, axis=1, keepdims=True)

    probe_r2 = jax.lax.stop_gradient(jnp.where(is_zero, 1.0, r2))
    probe_w = jnp.where(is_zero, 0.0, probe_r2 ** (-p / 2.0))
    probe_dw_pow = jnp.where(is_zero, 0.0, probe_r2 ** (-p / 2.0 - 1.0))
    probe_sum_w = jnp.sum(probe_w, axis=1, keepdims=True)
    overflow = (~has_zero) & (
        jnp.any(
            (~jnp.isfinite(probe_w)) | (~jnp.isfinite(probe_dw_pow)),
            axis=1,
            keepdims=True,
        )
        | (~jnp.isfinite(probe_sum_w))
        | (~jnp.isfinite(probe_sum_w**2))
    )

    # Mask exact-hit rows and overflowing rows before differentiating the direct branch.
    skip_direct = has_zero | is_zero | overflow
    safe_r2_direct = jnp.where(skip_direct, 1.0, r2)
    w_direct = jnp.where(skip_direct, 0.0, safe_r2_direct ** (-p / 2.0))
    sum_direct = jnp.sum(w_direct, axis=1, keepdims=True)

    # For near-hits whose unscaled inverse-distance powers overflow, rescale by min(r2).
    min_r2 = jax.lax.stop_gradient(jnp.min(probe_r2, axis=1, keepdims=True))
    safe_min_r2 = jnp.where((min_r2 > 0.0) & (~has_zero), min_r2, 1.0)
    ratio_r2 = jnp.where(is_zero | (~overflow), safe_min_r2, r2) / safe_min_r2
    w_scaled = jnp.where(is_zero | (~overflow), 0.0, ratio_r2 ** (-p / 2.0))
    sum_scaled = jnp.sum(w_scaled, axis=1, keepdims=True)

    w = jnp.where(overflow, w_scaled, w_direct)
    sum_w = jnp.where(overflow, sum_scaled, sum_direct)
    underflow = (~has_zero) & (~overflow) & (sum_direct == 0.0)
    return diff, r2, is_zero, has_zero, w, sum_w, underflow


def _idw_weights_jax(jax, jnp, x_arr, xt_arr, p, dtype):
    nt = xt_arr.shape[0]
    _, r2, is_zero, has_zero, w, sum_w, underflow = _idw_jax_terms(
        jax, jnp, x_arr, xt_arr, p
    )
    safe_sum_w = jnp.where(has_zero | underflow, 1.0, sum_w)
    first_zero_idx = jnp.argmax(is_zero, axis=1, keepdims=True)
    one_hot_zero = (jnp.arange(nt)[None, :] == first_zero_idx).astype(dtype)
    nearest_idx = jnp.argmin(r2, axis=1, keepdims=True)
    one_hot_nearest = (jnp.arange(nt)[None, :] == nearest_idx).astype(dtype)
    return jnp.where(
        has_zero,
        one_hot_zero,
        jnp.where(underflow, one_hot_nearest, w / safe_sum_w),
    )


def _idw_derivs_jax(jax, jnp, x_arr, xt_arr, p, kx):
    diff, r2, is_zero, has_zero, w, sum_w, underflow = _idw_jax_terms(
        jax, jnp, x_arr, xt_arr, p
    )
    skip_deriv = has_zero | underflow | is_zero | (~jnp.isfinite(r2))
    safe_r2 = jnp.where(skip_deriv, 1.0, r2)
    dw = jnp.where(skip_deriv, 0.0, -p * w * diff[:, :, kx] / safe_r2)
    safe_sum_w = jnp.where(has_zero | underflow, 1.0, sum_w)
    ds = jnp.sum(dw, axis=1, keepdims=True)
    deriv = (dw * safe_sum_w - w * ds) / (safe_sum_w**2)
    return jnp.where(has_zero | underflow, 0.0, deriv)


def _get_jax_idw_kernels():
    global _JAX_IDW_KERNELS
    if _JAX_IDW_KERNELS is None:
        jax, jnp = _get_jax()

        def weights_fn(x_j, xt_j, p_val):
            return _idw_weights_jax(jax, jnp, x_j, xt_j, p_val, jnp.float64)

        def derivs_fn(x_j, xt_j, p_val, kx):
            return _idw_derivs_jax(jax, jnp, x_j, xt_j, p_val, kx)

        _JAX_IDW_KERNELS = (
            jax.jit(weights_fn),
            jax.jit(derivs_fn, static_argnums=(3,)),
        )
    return _JAX_IDW_KERNELS


def _idw_jax(x, xt, p, kx):
    jax, jnp = _get_jax()
    if jax is None:
        return _idw_np(x, xt, p, kx)
    weights_jit, derivs_jit = _get_jax_idw_kernels()
    with _enable_x64(jax):
        x_j = jnp.asarray(x, dtype=jnp.float64)
        xt_j = jnp.asarray(xt, dtype=jnp.float64)
        if kx is None:
            res = weights_jit(x_j, xt_j, p)
        else:
            res = derivs_jit(x_j, xt_j, p, int(kx))
        return np.asarray(res, dtype=float)


def make_idw_jax_predictor(xt, yt, p):
    """Return a pure JAX predictor ``predict(x)`` for a trained IDW model."""
    jax, jnp = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for IDW.get_jax_predictor().")
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    xt_j = jnp.asarray(xt, dtype=dtype)
    yt_j = jnp.asarray(yt, dtype=dtype)
    p_val = float(p)
    nx = xt_j.shape[1]
    ny = yt_j.shape[1]

    def predict_fn(x):
        x_in = jnp.asarray(x, dtype=dtype)
        is_scalar = x_in.ndim <= 1
        x_arr = jnp.atleast_2d(x_in)
        if x_arr.shape[-1] != nx:
            raise ValueError(
                f"Expected input with {nx} features, got shape {x_in.shape}."
            )
        weights = _idw_weights_jax(jax, jnp, x_arr, xt_j, p_val, dtype)
        y = jnp.matmul(weights, yt_j)
        if is_scalar:
            return y[0, 0] if ny == 1 else y[0]
        return y

    return predict_fn


# ======================================================================
# Public entry points
# ======================================================================


def _as2d(x):
    return np.ascontiguousarray(np.atleast_2d(x), dtype=float)


def rbf_jac(x, xt, d0, poly_degree, kx=None, _backend=None):
    """
    Jacobian of the RBF states w.r.t. the coefficients, shape (n, nt + n_poly).

    If ``kx`` is given, the derivative of each column w.r.t. ``x[:, kx]``
    is returned instead.
    """
    x = _as2d(x)
    xt = _as2d(xt)
    d0 = np.ascontiguousarray(d0, dtype=float)
    if _resolve_backend(_backend) == "numba":
        n, nx = x.shape
        num_poly = {-1: 0, 0: 1}.get(poly_degree, 1 + nx)
        jac = np.empty((n, xt.shape[0] + num_poly))
        rbf_k, _ = _get_numba_kernels()
        rbf_k(
            x,
            xt,
            d0,
            int(poly_degree),
            0 if kx is None else int(kx),
            kx is not None,
            jac,
        )
        return jac
    return _rbf_np(x, xt, d0, int(poly_degree), kx)


def idw_jac(x, xt, p, kx=None, _backend=None):
    """
    Interpolation weights of IDW, shape (n, nt) (or their derivative w.r.t.
    ``x[:, kx]`` when ``kx`` is given).
    """
    x = _as2d(x)
    xt = _as2d(xt)
    backend = _resolve_backend(_backend)
    if backend == "jax":
        return _idw_jax(x, xt, float(p), kx)
    if backend == "numba":
        jac = np.empty((x.shape[0], xt.shape[0]))
        _, idw_k = _get_numba_kernels()
        idw_k(x, xt, float(p), 0 if kx is None else int(kx), kx is not None, jac)
        return jac
    return _idw_np(x, xt, float(p), kx)
