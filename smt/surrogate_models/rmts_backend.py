"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>, Paul Saves <paul.saves@ird.fr>

This package is distributed under New BSD license.

Pure-Python computational backends (NumPy/SciPy, optional Numba JIT, and
optional JAX with automatic differentiation) for RMTS, RMTB, and RMTC.
"""

import os
from functools import lru_cache

import numpy as np
import scipy.sparse


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


USE_NUMBA_JIT = int(_parse_bool_env("USE_NUMBA_JIT", 0))
USE_JAX = int(_parse_bool_env("USE_JAX", 0))

_NUMBA_KERNELS = None
prange = range


def _is_jax_enabled():
    if os.getenv("USE_JAX") is not None:
        return _parse_bool_env("USE_JAX", 0)
    return bool(USE_JAX)


def _is_numba_enabled():
    if os.getenv("USE_NUMBA_JIT") is not None:
        return _parse_bool_env("USE_NUMBA_JIT", 0)
    return bool(USE_NUMBA_JIT)


def _enable_x64(jax):
    """Return a context manager temporarily enabling JAX 64-bit precision."""
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    return enable(True)


def _get_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return None, None
    return jax, jnp


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


def has_jax():
    """Return True if JAX is importable."""
    return _get_jax()[0] is not None


def _use_jax_opt():
    """Return True when JAX AD should be used for RMTS optimization."""
    return _is_jax_enabled() and has_jax()


def _resolve_backend(_backend=None):
    if _backend is None or _backend == "auto":
        if _is_jax_enabled() and has_jax():
            return "jax"
        if _is_numba_enabled() and has_numba():
            return "numba"
        return "numpy"
    if _backend == "jax":
        if has_jax():
            return "jax"
        if _is_numba_enabled() and has_numba():
            return "numba"
        return "numpy"
    if _backend == "numba":
        if has_numba():
            return "numba"
        return "numpy"
    if _backend == "numpy":
        return "numpy"
    raise ValueError(
        f"Unsupported backend {_backend!r}; expected one of (None, 'auto', 'numpy', 'numba', 'jax')."
    )


# ======================================================================
# Shared geometry / knot utilities
# ======================================================================


def compute_knot_vector_uniform(order, ncp):
    """
    Compute the uniform B-spline knot vector of length ``order + ncp``.

    Parameters
    ----------
    order : int
        B-spline order (>= 1).
    ncp : int
        Number of control points (>= order).

    Returns
    -------
    np.ndarray
        1D float64 knot vector of length ``order + ncp``.
    """
    i = np.arange(order + ncp, dtype=np.float64)
    return (i - (order - 1)) / float(ncp - order + 1)


def compute_ext_dist(x, lower, upper, nterm=1, _backend=None):
    """
    Compute vectors from the nearest point in ``[lower, upper]`` to ``x``,
    repeated ``nterm`` times per point (matching ``RMTS::compute_ext_dist``).

    Parameters
    ----------
    x : array_like
        Points of shape ``(n, nx)`` or flat length ``n * nx``.
    lower : array_like
        Lower bounds of length ``nx``.
    upper : array_like
        Upper bounds of length ``nx``.
    nterm : int, optional
        Number of basis support terms per point.
    _backend : str or None, optional
        Internal backend selector (accepted for API consistency).

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n * nterm, nx)``.
    """
    lower_arr = np.asarray(lower, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)
    nx = lower_arr.shape[0]
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        if x_arr.size == 0:
            return np.empty((0, nx), dtype=np.float64)
        x_arr = x_arr.reshape((-1, nx))
    n = x_arr.shape[0]
    if n == 0:
        return np.empty((0, nx), dtype=np.float64)
    clamped = np.clip(x_arr, lower_arr, upper_arr)
    diff = x_arr - clamped
    if nterm == 1:
        return np.array(diff, dtype=np.float64, copy=True)
    return np.repeat(diff, int(nterm), axis=0)


def compute_quadrature_points(nelem_list, lower, upper):
    """
    Compute element-centroid quadrature points across an n-D box grid
    (matching ``RMTS::compute_quadrature_points``).

    Parameters
    ----------
    nelem_list : array_like
        Number of elements in each dimension, length ``nx``.
    lower : array_like
        Lower bounds, length ``nx``.
    upper : array_like
        Upper bounds, length ``nx``.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(prod(nelem_list), nx)``.
    """
    nelem_arr = np.asarray(nelem_list, dtype=np.int64)
    lower_arr = np.asarray(lower, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)
    nx = nelem_arr.shape[0]
    coords_1d = []
    for ix in range(nx):
        m = int(nelem_arr[ix])
        t = ((1.0 + 2.0 * np.arange(m, dtype=np.float64)) / 2.0) / float(m)
        coords_1d.append(lower_arr[ix] + t * (upper_arr[ix] - lower_arr[ix]))
    grids = np.meshgrid(*coords_1d, indexing="ij")
    return np.column_stack([g.ravel() for g in grids]).astype(np.float64, copy=False)


# ======================================================================
# Numba JIT kernels (top-level functions compiled lazily with cache=True)
# ======================================================================


def _rmtb_jac_numba_raw(
    ix1,
    ix2,
    nx,
    n,
    params,
    order_list,
    ncp_list,
    ncp_strides,
    knots_flat,
    knots_offsets,
    col_offsets,
    nnz_row,
    b0_buf,
    b1_buf,
    b2_buf,
    data,
    rows,
    cols,
):
    for i in prange(n):
        base = i * nnz_row
        for k in range(nnz_row):
            data[base + k] = 1.0
            rows[base + k] = i
            cols[base + k] = col_offsets[k]

    for ix in range(nx):
        order = order_list[ix]
        ncp = ncp_list[ix]
        stride = ncp_strides[ix]
        k_off = knots_offsets[ix]
        prod_order = 1
        for kx in range(ix + 1, nx):
            prod_order *= order_list[kx]
        deriv = 0
        if ix == ix1:
            deriv += 1
        if ix == ix2:
            deriv += 1

        for i in prange(n):
            param = params[i, ix]
            lo = 0
            hi = ncp - order + 1
            k_search_start = k_off + order - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if knots_flat[k_search_start + mid] <= param:
                    lo = mid + 1
                else:
                    hi = mid
            istart = lo - 1
            if istart < 0:
                istart = 0
            elif istart > ncp - order:
                istart = ncp - order

            for j in range(order):
                b0_buf[i, j] = 0.0
                if deriv >= 1:
                    b1_buf[i, j] = 0.0
                if deriv == 2:
                    b2_buf[i, j] = 0.0
            b0_buf[i, order - 1] = 1.0

            for deg in range(1, order):
                j1 = order - deg - 1
                j2 = order - 1
                for j in range(j1, j2 + 1):
                    kn_idx = k_off + istart + j
                    kn = knots_flat[kn_idx]
                    kni = knots_flat[kn_idx + deg]
                    den1 = kni - kn

                    if deriv == 0:
                        b1_val = 0.0
                        if den1 != 0.0:
                            b1_val = (param - kn) / den1 * b0_buf[i, j]
                        b2_val = 0.0
                        if j != j2:
                            kn1 = knots_flat[kn_idx + 1]
                            kni1 = knots_flat[kn_idx + deg + 1]
                            den2 = kni1 - kn1
                            if den2 != 0.0:
                                b2_val = (kni1 - param) / den2 * b0_buf[i, j + 1]
                        b0_buf[i, j] = b1_val + b2_val
                    elif deriv == 1:
                        if den1 != 0.0:
                            b1_val = (param - kn) / den1 * b0_buf[i, j]
                            f1_val = (b0_buf[i, j] + (param - kn) * b1_buf[i, j]) / den1
                        else:
                            b1_val = 0.0
                            f1_val = 0.0
                        if j != j2:
                            kn1 = knots_flat[kn_idx + 1]
                            kni1 = knots_flat[kn_idx + deg + 1]
                            den2 = kni1 - kn1
                            if den2 != 0.0:
                                b2_val = (kni1 - param) / den2 * b0_buf[i, j + 1]
                                f2_val = (
                                    (kni1 - param) * b1_buf[i, j + 1] - b0_buf[i, j + 1]
                                ) / den2
                            else:
                                b2_val = 0.0
                                f2_val = 0.0
                        else:
                            b2_val = 0.0
                            f2_val = 0.0
                        b0_buf[i, j] = b1_val + b2_val
                        b1_buf[i, j] = f1_val + f2_val
                    else:
                        if den1 != 0.0:
                            b1_val = (param - kn) / den1 * b0_buf[i, j]
                            f1_val = (b0_buf[i, j] + (param - kn) * b1_buf[i, j]) / den1
                            s1_val = (
                                2.0 * b1_buf[i, j] + (param - kn) * b2_buf[i, j]
                            ) / den1
                        else:
                            b1_val = 0.0
                            f1_val = 0.0
                            s1_val = 0.0
                        if j != j2:
                            kn1 = knots_flat[kn_idx + 1]
                            kni1 = knots_flat[kn_idx + deg + 1]
                            den2 = kni1 - kn1
                            if den2 != 0.0:
                                b2_val = (kni1 - param) / den2 * b0_buf[i, j + 1]
                                f2_val = (
                                    (kni1 - param) * b1_buf[i, j + 1] - b0_buf[i, j + 1]
                                ) / den2
                                s2_val = (
                                    (kni1 - param) * b2_buf[i, j + 1]
                                    - 2.0 * b1_buf[i, j + 1]
                                ) / den2
                            else:
                                b2_val = 0.0
                                f2_val = 0.0
                                s2_val = 0.0
                        else:
                            b2_val = 0.0
                            f2_val = 0.0
                            s2_val = 0.0
                        b0_buf[i, j] = b1_val + b2_val
                        b1_buf[i, j] = f1_val + f2_val
                        if deg > 1:
                            b2_buf[i, j] = s1_val + s2_val

            base = i * nnz_row
            istart_stride = istart * stride
            for inz_row in range(nnz_row):
                inz_dim = (inz_row // prod_order) % order
                if deriv == 0:
                    data[base + inz_row] *= b0_buf[i, inz_dim]
                elif deriv == 1:
                    data[base + inz_row] *= b1_buf[i, inz_dim]
                else:
                    data[base + inz_row] *= b2_buf[i, inz_dim]
                cols[base + inz_row] += istart_stride


def _rmtc_jac_numba_raw(
    ix1,
    ix2,
    nx,
    n,
    x,
    lower,
    upper,
    dxb_dx,
    nelem_list,
    nterm_list,
    nelem_strides,
    nterm,
    basis_buf,
    data,
    rows,
    cols,
):
    for i in prange(n):
        base = i * nterm
        for iterm in range(nterm):
            data[base + iterm] = 1.0
            rows[base + iterm] = i
            cols[base + iterm] = iterm

    for ix in range(nx):
        a = lower[ix]
        b = upper[ix]
        num = nelem_list[ix]
        nt_ix = nterm_list[ix]
        inv_bma_d2 = dxb_dx[ix]
        stride_elem = nelem_strides[ix] * nterm
        prod_nterm = 1
        for kx in range(ix + 1, nx):
            prod_nterm *= nterm_list[kx]
        deriv = 0
        if ix == ix1:
            deriv += 1
        if ix == ix2:
            deriv += 1

        for i in prange(n):
            xi = x[i, ix]
            idx_1b = int(np.ceil((xi - a) / (b - a) * num))
            if idx_1b < 1:
                idx_1b = 1
            elif idx_1b > num:
                idx_1b = num
            a2 = a + (b - a) * (idx_1b - 1) / num
            b2 = a + (b - a) * idx_1b / num
            bma_d2 = (b2 - a2) / 2.0
            apb_d2 = (a2 + b2) / 2.0
            xbar = (xi - apb_d2) / bma_d2
            if xbar < -1.0:
                xbar = -1.0
            elif xbar > 1.0:
                xbar = 1.0
            ielem_ix = idx_1b - 1

            if deriv == 0:
                val = 1.0
                for p in range(nt_ix):
                    basis_buf[i, p] = val
                    val *= xbar
            elif deriv == 1:
                basis_buf[i, 0] = 0.0
                val = 1.0
                for p in range(1, nt_ix):
                    basis_buf[i, p] = p * val * inv_bma_d2
                    val *= xbar
            else:
                basis_buf[i, 0] = 0.0
                if nt_ix > 1:
                    basis_buf[i, 1] = 0.0
                val = 1.0
                for p in range(2, nt_ix):
                    basis_buf[i, p] = p * (p - 1) * val * inv_bma_d2 * inv_bma_d2
                    val *= xbar

            base = i * nterm
            col_add = ielem_ix * stride_elem
            for iterm in range(nterm):
                iterm_dim = (iterm // prod_nterm) % nt_ix
                data[base + iterm] *= basis_buf[i, iterm_dim]
                cols[base + iterm] += col_add


def _get_numba_kernels():
    global _NUMBA_KERNELS, prange
    njit, nb_prange = _get_numba()
    if njit is None:
        return None
    if _NUMBA_KERNELS is None:
        prange = nb_prange
        rmtb_kernel = njit(cache=True, parallel=True)(_rmtb_jac_numba_raw)
        rmtc_kernel = njit(cache=True, parallel=True)(_rmtc_jac_numba_raw)
        _NUMBA_KERNELS = (rmtb_kernel, rmtc_kernel)
    return _NUMBA_KERNELS


# ======================================================================
# JAX AD basis & evaluation helpers (lazy import + cached JIT)
# ======================================================================


def _rmtb_basis0_scalar_impl(param, knots, order, ncp, istart, jnp, jax):
    dt = param.dtype
    if order == 1:
        return jnp.ones((1,), dtype=dt)
    local_knots = jax.lax.dynamic_slice(knots, (istart,), (2 * order - 1,))
    b0 = jnp.zeros((order,), dtype=dt).at[order - 1].set(1.0)
    for i in range(1, order):
        j1 = order - i - 1
        j2 = order - 1
        b0_next = jnp.zeros((order,), dtype=dt)
        for j in range(j1, j2 + 1):
            kn = local_knots[j]
            kni = local_knots[j + i]
            den1 = kni - kn
            safe_den1 = jnp.where(den1 != 0.0, den1, 1.0)
            t1 = jnp.where(den1 != 0.0, (param - kn) / safe_den1 * b0[j], 0.0)
            if j != j2:
                kn1 = local_knots[j + 1]
                kni1 = local_knots[j + i + 1]
                den2 = kni1 - kn1
                safe_den2 = jnp.where(den2 != 0.0, den2, 1.0)
                t2 = jnp.where(den2 != 0.0, (kni1 - param) / safe_den2 * b0[j + 1], 0.0)
            else:
                t2 = 0.0
            b0_next = b0_next.at[j].set(t1 + t2)
        b0 = b0_next
    return b0


@lru_cache(maxsize=128)
def _get_rmtb_dim_eval_jax(order, ncp, deriv_order):
    jax, jnp = _get_jax()
    order = int(order)
    ncp = int(ncp)
    deriv_order = int(deriv_order)

    def basis0(param, knots):
        istart = jnp.clip(
            jnp.searchsorted(knots[order - 1 : ncp], param, side="right") - 1,
            0,
            ncp - order,
        ).astype(jnp.int64)
        istart = jax.lax.stop_gradient(istart)
        return _rmtb_basis0_scalar_impl(param, knots, order, ncp, istart, jnp, jax)

    if deriv_order == 0:
        basis_scalar = basis0
    elif deriv_order == 1:
        basis_scalar = jax.jacfwd(basis0, argnums=0)
    else:
        basis_scalar = jax.jacfwd(jax.jacfwd(basis0, argnums=0), argnums=0)

    def eval_point(param, knots):
        istart = jnp.clip(
            jnp.searchsorted(knots[order - 1 : ncp], param, side="right") - 1,
            0,
            ncp - order,
        ).astype(jnp.int64)
        istart = jax.lax.stop_gradient(istart)
        return istart, basis_scalar(param, knots)

    return jax.jit(jax.vmap(eval_point, in_axes=(0, None)))


@lru_cache(maxsize=64)
def _get_rmtc_dim_eval_jax(nterm_ix, deriv_order):
    jax, jnp = _get_jax()
    nterm_ix = int(nterm_ix)
    deriv_order = int(deriv_order)

    def poly(u):
        if nterm_ix == 4:
            return jnp.array([1.0, u, u**2, u**3], dtype=jnp.float64)
        return jnp.array(
            [1.0 if p == 0 else u**p for p in range(nterm_ix)], dtype=jnp.float64
        )

    if deriv_order == 0:
        poly_fn = poly
    elif deriv_order == 1:
        poly_fn = jax.jacfwd(poly)
    else:
        poly_fn = jax.jacfwd(jax.jacfwd(poly))

    def eval_point(x_val, a, b, num, dxb_dx_val):
        idx_1b = jnp.clip(
            jnp.ceil((x_val - a) / (b - a) * num).astype(jnp.int64), 1, num
        )
        idx_1b = jax.lax.stop_gradient(idx_1b)
        a2 = a + (b - a) * (idx_1b - 1) / num
        b2 = a + (b - a) * idx_1b / num
        bma_d2 = (b2 - a2) / 2.0
        apb_d2 = (a2 + b2) / 2.0
        u = jnp.clip((x_val - apb_d2) / bma_d2, -1.0, 1.0)
        vals = poly_fn(u)
        if deriv_order == 1:
            vals = vals * dxb_dx_val
        elif deriv_order == 2:
            vals = vals * dxb_dx_val * dxb_dx_val
        return idx_1b - 1, vals

    return jax.jit(
        jax.vmap(eval_point, in_axes=(0, None, None, None, None)), static_argnums=(3,)
    )


@lru_cache(maxsize=64)
def _get_predict_rmtb_jax_fn(order_tuple, ncp_tuple, extrapolate):
    jax, jnp = _get_jax()
    nx = len(order_tuple)

    def eval_interior(x_in, coeff_nd, lower_j, upper_j, knots_tuple):
        t = (x_in - lower_j) / (upper_j - lower_j)
        istarts = []
        bases = []
        for d in range(nx):
            o_d = order_tuple[d]
            c_d = ncp_tuple[d]
            k_d = knots_tuple[d]
            ist = jnp.clip(
                jnp.searchsorted(k_d[o_d - 1 : c_d], t[d], side="right") - 1,
                0,
                c_d - o_d,
            ).astype(jnp.int32)
            ist = jax.lax.stop_gradient(ist)
            istarts.append(ist)
            bases.append(_rmtb_basis0_scalar_impl(t[d], k_d, o_d, c_d, ist, jnp, jax))
        ny = coeff_nd.shape[-1]
        local_c = jax.lax.dynamic_slice(
            coeff_nd, tuple(istarts) + (jnp.int32(0),), order_tuple + (ny,)
        )
        val = local_c
        for b_d in bases:
            val = jnp.tensordot(b_d, val, axes=(0, 0))
        return val

    def eval_single(x_pt, coeff_nd, lower_j, upper_j, knots_tuple):
        x_b = jnp.where(
            x_pt < lower_j, lower_j, jnp.where(x_pt > upper_j, upper_j, x_pt)
        )
        if not extrapolate:
            return eval_interior(x_b, coeff_nd, lower_j, upper_j, knots_tuple)
        val_b = eval_interior(x_b, coeff_nd, lower_j, upper_j, knots_tuple)
        jac_b = jax.jacfwd(eval_interior, argnums=0)(
            x_b, coeff_nd, lower_j, upper_j, knots_tuple
        )
        return val_b + jac_b @ (x_pt - x_b)

    return jax.jit(jax.vmap(eval_single, in_axes=(0, None, None, None, None)))


def predict_rmtb_jax(
    x,
    sol_coeff,
    lower,
    upper,
    order_list,
    ncp_list,
    knots_list=None,
    extrapolate=False,
):
    """
    Pure JAX differentiable RMTB predictor.

    Supports single-point ``(nx,)`` or batched ``(n, nx)`` inputs and
    automatic differentiation with ``jax.grad``, ``jax.jacfwd``, and ``jax.vmap``.
    Uses active JAX floating precision (``float64`` when ``jax_enable_x64``
    is enabled, ``float32`` otherwise).
    """
    jax, jnp = _get_jax()
    if jnp is None:
        raise ImportError("JAX is required for predict_rmtb_jax.")

    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    order_tuple = tuple(int(v) for v in order_list)
    ncp_tuple = tuple(int(v) for v in ncp_list)
    if knots_list is None:
        knots_tuple = tuple(
            jnp.asarray(compute_knot_vector_uniform(o, c), dtype=dtype)
            for o, c in zip(order_tuple, ncp_tuple)
        )
    else:
        knots_tuple = tuple(jnp.asarray(k, dtype=dtype) for k in knots_list)

    lower_j = jnp.asarray(lower, dtype=dtype)
    upper_j = jnp.asarray(upper, dtype=dtype)
    coeff_arr = jnp.asarray(sol_coeff, dtype=dtype)
    squeeze_y = coeff_arr.ndim == 1
    if squeeze_y:
        coeff_arr = coeff_arr[:, None]
    ny = coeff_arr.shape[1]
    coeff_nd = coeff_arr.reshape(ncp_tuple + (ny,))

    x_j = jnp.asarray(x, dtype=dtype)
    squeeze_x = x_j.ndim == 1
    if squeeze_x:
        x_j = x_j[None, :]

    fn = _get_predict_rmtb_jax_fn(order_tuple, ncp_tuple, bool(extrapolate))
    y = fn(x_j, coeff_nd, lower_j, upper_j, knots_tuple)
    if squeeze_x:
        y = y[0]
    if squeeze_y:
        y = y[..., 0]
    return y


@lru_cache(maxsize=64)
def _get_predict_rmtc_jax_fn(nelem_tuple, nterm_tuple, extrapolate):
    jax, jnp = _get_jax()
    nx = len(nelem_tuple)
    nterm = 1
    for nt in nterm_tuple:
        nterm *= nt
    nelem_strides = []
    stride = 1
    for m in reversed(nelem_tuple):
        nelem_strides.insert(0, stride)
        stride *= m
    nelem_strides = tuple(nelem_strides)

    def eval_single(x_pt, coeff_nd, lower_j, upper_j):
        x_b = jnp.where(
            x_pt < lower_j, lower_j, jnp.where(x_pt > upper_j, upper_j, x_pt)
        )
        ielem = jnp.int32(0)
        apb_list = []
        bma_list = []
        for d in range(nx):
            a = lower_j[d]
            b = upper_j[d]
            num = nelem_tuple[d]
            idx_1b = jnp.clip(
                jnp.ceil((x_b[d] - a) / (b - a) * num).astype(jnp.int32), 1, num
            )
            idx_1b = jax.lax.stop_gradient(idx_1b)
            a2 = a + (b - a) * (idx_1b - 1) / num
            b2 = a + (b - a) * idx_1b / num
            bma_list.append((b2 - a2) / 2.0)
            apb_list.append((a2 + b2) / 2.0)
            ielem = ielem + (idx_1b - 1) * jnp.int32(nelem_strides[d])

        ny = coeff_nd.shape[-1]
        elem_c = jax.lax.dynamic_slice(
            coeff_nd, (ielem, jnp.int32(0), jnp.int32(0)), (1, nterm, ny)
        )[0]
        elem_c = elem_c.reshape(nterm_tuple + (ny,))

        def eval_in_elem(x_in):
            val = elem_c
            for d in range(nx):
                u_d = (x_in[d] - apb_list[d]) / bma_list[d]
                nt_d = nterm_tuple[d]
                if nt_d == 4:
                    p_d = jnp.array([1.0, u_d, u_d**2, u_d**3], dtype=u_d.dtype)
                else:
                    p_d = jnp.array(
                        [1.0 if p == 0 else u_d**p for p in range(nt_d)],
                        dtype=u_d.dtype,
                    )
                val = jnp.tensordot(p_d, val, axes=(0, 0))
            return val

        if not extrapolate:
            return eval_in_elem(x_b)
        val_b = eval_in_elem(x_b)
        jac_b = jax.jacfwd(eval_in_elem)(x_b)
        return val_b + jac_b @ (x_pt - x_b)

    return jax.jit(jax.vmap(eval_single, in_axes=(0, None, None, None)))


def predict_rmtc_jax(
    x,
    sol_coeff,
    lower,
    upper,
    nelem_list,
    nterm_list=None,
    extrapolate=False,
):
    """
    Pure JAX differentiable RMTC predictor.

    Supports single-point ``(nx,)`` or batched ``(n, nx)`` inputs and
    automatic differentiation with ``jax.grad``, ``jax.jacfwd``, and ``jax.vmap``.
    Uses active JAX floating precision (``float64`` when ``jax_enable_x64``
    is enabled, ``float32`` otherwise).
    """
    jax, jnp = _get_jax()
    if jnp is None:
        raise ImportError("JAX is required for predict_rmtc_jax.")

    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    nelem_tuple = tuple(int(v) for v in nelem_list)
    if nterm_list is None:
        nterm_tuple = (4,) * len(nelem_tuple)
    else:
        nterm_tuple = tuple(int(v) for v in nterm_list)
    nelem = 1
    for m in nelem_tuple:
        nelem *= m
    nterm = 1
    for nt in nterm_tuple:
        nterm *= nt

    lower_j = jnp.asarray(lower, dtype=dtype)
    upper_j = jnp.asarray(upper, dtype=dtype)
    coeff_arr = jnp.asarray(sol_coeff, dtype=dtype)
    squeeze_y = coeff_arr.ndim == 1
    if squeeze_y:
        coeff_arr = coeff_arr[:, None]
    ny = coeff_arr.shape[1]
    coeff_nd = coeff_arr.reshape((nelem, nterm, ny))

    x_j = jnp.asarray(x, dtype=dtype)
    squeeze_x = x_j.ndim == 1
    if squeeze_x:
        x_j = x_j[None, :]

    fn = _get_predict_rmtc_jax_fn(nelem_tuple, nterm_tuple, bool(extrapolate))
    y = fn(x_j, coeff_nd, lower_j, upper_j)
    if squeeze_x:
        y = y[0]
    if squeeze_y:
        y = y[..., 0]
    return y


# ======================================================================
# JAX AD Newton solver helpers
# ======================================================================


@lru_cache(maxsize=16)
def _get_opt_jit_fns():
    jax, jnp = _get_jax()

    def residual_loss(r, p_s):
        return r**p_s

    loss_sum_jit = jax.jit(
        lambda r, p_s: jnp.sum(r**p_s),
        static_argnums=(1,),
    )
    d1_jit = jax.jit(
        jax.vmap(jax.grad(residual_loss, argnums=0), in_axes=(0, None)),
        static_argnums=(1,),
    )
    d2_jit = jax.jit(
        jax.vmap(
            jax.grad(jax.grad(residual_loss, argnums=0), argnums=0),
            in_axes=(0, None),
        ),
        static_argnums=(1,),
    )
    return loss_sum_jit, d1_jit, d2_jit


def _prepare_jax_opt_cache(full_hess, full_jac_dict):
    """Cache CSR sparse operators once per solve for JAX AD residual loss."""
    _, jnp = _get_jax()
    if jnp is None:
        raise ImportError("JAX is required for JAX optimization cache.")
    keys = tuple(sorted(full_jac_dict.keys()))
    hess_csr = scipy.sparse.csr_matrix(full_hess, dtype=np.float64)
    jacs = []
    for kx in keys:
        full_jac, full_jac_T, c = full_jac_dict[kx]
        jac_csr = scipy.sparse.csr_matrix(full_jac, dtype=np.float64)
        jac_T_csr = scipy.sparse.csr_matrix(full_jac_T, dtype=np.float64)
        jacs.append((kx, jac_csr, jac_T_csr, float(c)))
    return {
        "hess_id": id(full_hess),
        "jac_dict_id": id(full_jac_dict),
        "hess_nnz": full_hess.nnz,
        "keys": keys,
        "hess_csr": hess_csr,
        "jacs": tuple(jacs),
    }


def _normalize_p(p):
    if isinstance(p, (int, np.integer)):
        return int(p)
    return float(p)


def _opt_func_jax(sol, p, yt_dict, cache):
    """Evaluate RMTS objective using SciPy CSR SpMV and JAX residual sum."""
    jax, jnp = _get_jax()
    loss_sum_jit, _, _ = _get_opt_jit_fns()
    sol_arr = np.asarray(sol, dtype=np.float64)
    p_norm = _normalize_p(p)
    val = 0.5 * float(np.dot(sol_arr, cache["hess_csr"].dot(sol_arr)))
    with _enable_x64(jax):
        for kx, jac_csr, _, c in cache["jacs"]:
            r = jac_csr.dot(sol_arr) - yt_dict[kx]
            val += (
                0.5 * c * float(loss_sum_jit(jnp.asarray(r, dtype=jnp.float64), p_norm))
            )
    return val


def _opt_grad_jax(sol, p, yt_dict, cache):
    """Evaluate RMTS gradient via JAX AD on residual losses and CSR SpMV."""
    jax, jnp = _get_jax()
    _, d1_jit, _ = _get_opt_jit_fns()
    sol_arr = np.asarray(sol, dtype=np.float64)
    p_norm = _normalize_p(p)
    grad = cache["hess_csr"].dot(sol_arr)
    with _enable_x64(jax):
        for kx, jac_csr, jac_T_csr, c in cache["jacs"]:
            r = jac_csr.dot(sol_arr) - yt_dict[kx]
            d1 = np.asarray(
                d1_jit(jnp.asarray(r, dtype=jnp.float64), p_norm), dtype=np.float64
            )
            grad = grad + 0.5 * c * jac_T_csr.dot(d1)
    return grad


def _opt_hess_jax(sol, p, yt_dict, cache, full_hess, full_jac_dict):
    """
    Assemble exact sparse CSC Hessian using JAX AD 2nd derivatives of scalar
    residual losses so all 12 SMT linear solvers and downstream tools work cleanly.
    """
    jax, jnp = _get_jax()
    _, _, d2_jit = _get_opt_jit_fns()
    sol_arr = np.asarray(sol, dtype=np.float64)
    p_norm = _normalize_p(p)
    hess = scipy.sparse.csc_matrix(full_hess, dtype=np.float64)
    with _enable_x64(jax):
        for kx, jac_csr, _, c in cache["jacs"]:
            full_jac, full_jac_T, _ = full_jac_dict[kx]
            r = jac_csr.dot(sol_arr) - yt_dict[kx]
            diag_vec = np.asarray(
                d2_jit(jnp.asarray(r, dtype=jnp.float64), p_norm), dtype=np.float64
            )
            diag_mtx = scipy.sparse.diags(diag_vec, format="csc")
            hess += 0.5 * c * full_jac_T * diag_mtx * full_jac
    return hess


# ======================================================================
# PureRMTB
# ======================================================================


class PureRMTB:
    """
    Pure-Python Tensor-product B-spline backend replacing C++ ``PyRMTB``.
    Supports NumPy, optional Numba JIT, and optional JAX AD engines.
    """

    def __init__(
        self,
        nx=None,
        lower=None,
        upper=None,
        order_list=None,
        ncp_list=None,
        _backend=None,
    ):
        self._backend = _backend
        self.nx = 0
        self.lower = None
        self.upper = None
        self.order_list = None
        self.ncp_list = None
        self.nnz_row = 0
        self.max_order = 0
        self.knots_list = []
        self.knots_flat = None
        self.knots_offsets = None
        self.ncp_strides = None
        self.col_offsets = None
        if nx is not None:
            self.setup(nx, lower, upper, order_list, ncp_list)

    def setup(self, nx, lower, upper, order_list, ncp_list):
        self.nx = int(nx)
        self.lower = np.asarray(lower, dtype=np.float64).copy()
        self.upper = np.asarray(upper, dtype=np.float64).copy()
        self.order_list = np.asarray(order_list, dtype=np.int64).copy()
        self.ncp_list = np.asarray(ncp_list, dtype=np.int64).copy()

        if (
            self.lower.shape[0] != self.nx
            or self.upper.shape[0] != self.nx
            or self.order_list.shape[0] != self.nx
            or self.ncp_list.shape[0] != self.nx
        ):
            raise ValueError("All RMTB setup arrays must have length nx.")

        self.nnz_row = int(np.prod(self.order_list))
        self.max_order = int(np.max(self.order_list)) if self.nx > 0 else 0

        self.knots_list = [
            compute_knot_vector_uniform(int(o), int(c))
            for o, c in zip(self.order_list, self.ncp_list)
        ]
        self.knots_flat = (
            np.concatenate(self.knots_list)
            if self.knots_list
            else np.empty(0, dtype=np.float64)
        )
        offsets = []
        pos = 0
        for k in self.knots_list:
            offsets.append(pos)
            pos += k.shape[0]
        self.knots_offsets = np.asarray(offsets, dtype=np.int64)

        strides = np.ones(self.nx, dtype=np.int64)
        for ix in range(self.nx - 2, -1, -1):
            strides[ix] = strides[ix + 1] * self.ncp_list[ix + 1]
        self.ncp_strides = strides

        ranges = [np.arange(int(o), dtype=np.int64) for o in self.order_list]
        grids = np.meshgrid(*ranges, indexing="ij")
        col_offsets = np.zeros(self.nnz_row, dtype=np.int64)
        for g, s in zip(grids, self.ncp_strides):
            col_offsets += g.ravel() * int(s)
        self.col_offsets = col_offsets.astype(np.int32)

    def compute_ext_dist(self, x, nterm=None):
        nt = self.nnz_row if nterm is None else int(nterm)
        return compute_ext_dist(x, self.lower, self.upper, nterm=nt)

    def compute_quadrature_points(self, nelem_list):
        return compute_quadrature_points(nelem_list, self.lower, self.upper)

    def _compute_basis_numpy(self, ix, params_1d, deriv_order):
        order = int(self.order_list[ix])
        ncp = int(self.ncp_list[ix])
        knots = self.knots_list[ix]
        n = params_1d.shape[0]

        istart = np.clip(
            np.searchsorted(knots[order - 1 : ncp], params_1d, side="right") - 1,
            0,
            ncp - order,
        )

        b0 = np.zeros((n, order), dtype=np.float64)
        b0[:, order - 1] = 1.0
        b1 = np.zeros((n, order), dtype=np.float64) if deriv_order >= 1 else None
        b2 = np.zeros((n, order), dtype=np.float64) if deriv_order == 2 else None

        for deg in range(1, order):
            j1 = order - deg - 1
            j2 = order - 1
            for j in range(j1, j2 + 1):
                k_idx = istart + j
                kn = knots[k_idx]
                kni = knots[k_idx + deg]
                den1 = kni - kn
                nz1 = den1 != 0.0
                safe_den1 = np.where(nz1, den1, 1.0)

                if deriv_order == 0:
                    b1_term = np.where(
                        nz1, (params_1d - kn) / safe_den1 * b0[:, j], 0.0
                    )
                    if j != j2:
                        kn1 = knots[k_idx + 1]
                        kni1 = knots[k_idx + deg + 1]
                        den2 = kni1 - kn1
                        nz2 = den2 != 0.0
                        safe_den2 = np.where(nz2, den2, 1.0)
                        b2_term = np.where(
                            nz2, (kni1 - params_1d) / safe_den2 * b0[:, j + 1], 0.0
                        )
                    else:
                        b2_term = 0.0
                    b0[:, j] = b1_term + b2_term
                elif deriv_order == 1:
                    b1_term = np.where(
                        nz1, (params_1d - kn) / safe_den1 * b0[:, j], 0.0
                    )
                    f1_term = np.where(
                        nz1,
                        (b0[:, j] + (params_1d - kn) * b1[:, j]) / safe_den1,
                        0.0,
                    )
                    if j != j2:
                        kn1 = knots[k_idx + 1]
                        kni1 = knots[k_idx + deg + 1]
                        den2 = kni1 - kn1
                        nz2 = den2 != 0.0
                        safe_den2 = np.where(nz2, den2, 1.0)
                        b2_term = np.where(
                            nz2, (kni1 - params_1d) / safe_den2 * b0[:, j + 1], 0.0
                        )
                        f2_term = np.where(
                            nz2,
                            ((kni1 - params_1d) * b1[:, j + 1] - b0[:, j + 1])
                            / safe_den2,
                            0.0,
                        )
                    else:
                        b2_term = 0.0
                        f2_term = 0.0
                    b0[:, j] = b1_term + b2_term
                    b1[:, j] = f1_term + f2_term
                else:
                    b1_term = np.where(
                        nz1, (params_1d - kn) / safe_den1 * b0[:, j], 0.0
                    )
                    f1_term = np.where(
                        nz1,
                        (b0[:, j] + (params_1d - kn) * b1[:, j]) / safe_den1,
                        0.0,
                    )
                    s1_term = np.where(
                        nz1,
                        (2.0 * b1[:, j] + (params_1d - kn) * b2[:, j]) / safe_den1,
                        0.0,
                    )
                    if j != j2:
                        kn1 = knots[k_idx + 1]
                        kni1 = knots[k_idx + deg + 1]
                        den2 = kni1 - kn1
                        nz2 = den2 != 0.0
                        safe_den2 = np.where(nz2, den2, 1.0)
                        b2_term = np.where(
                            nz2, (kni1 - params_1d) / safe_den2 * b0[:, j + 1], 0.0
                        )
                        f2_term = np.where(
                            nz2,
                            ((kni1 - params_1d) * b1[:, j + 1] - b0[:, j + 1])
                            / safe_den2,
                            0.0,
                        )
                        s2_term = np.where(
                            nz2,
                            ((kni1 - params_1d) * b2[:, j + 1] - 2.0 * b1[:, j + 1])
                            / safe_den2,
                            0.0,
                        )
                    else:
                        b2_term = 0.0
                        f2_term = 0.0
                        s2_term = 0.0
                    b0[:, j] = b1_term + b2_term
                    b1[:, j] = f1_term + f2_term
                    if deg > 1:
                        b2[:, j] = s1_term + s2_term

        if deriv_order == 0:
            return istart, b0
        if deriv_order == 1:
            return istart, b1
        return istart, b2

    def _compute_jac_numpy(self, ix1, ix2, params):
        n = params.shape[0]
        basis_list = []
        cols_base = np.zeros(n, dtype=np.int64)
        for ix in range(self.nx):
            deriv = (1 if ix == ix1 else 0) + (1 if ix == ix2 else 0)
            istart, basis_ix = self._compute_basis_numpy(ix, params[:, ix], deriv)
            basis_list.append(basis_ix)
            cols_base += istart * int(self.ncp_strides[ix])

        data_2d = basis_list[0]
        for ix in range(1, self.nx):
            data_2d = (data_2d[:, :, None] * basis_list[ix][:, None, :]).reshape(n, -1)
        data = np.array(data_2d.ravel(), dtype=np.float64, copy=True)
        rows = np.repeat(np.arange(n, dtype=np.int32), self.nnz_row)
        cols = (cols_base[:, None] + self.col_offsets[None, :]).ravel().astype(np.int32)
        return data, rows, cols

    def _compute_jac_numba(self, ix1, ix2, params):
        kernels = _get_numba_kernels()
        if kernels is None:
            return self._compute_jac_numpy(ix1, ix2, params)
        rmtb_kernel, _ = kernels
        n = params.shape[0]
        nnz = n * self.nnz_row
        data = np.empty(nnz, dtype=np.float64)
        rows = np.empty(nnz, dtype=np.int32)
        cols = np.empty(nnz, dtype=np.int32)
        b0_buf = np.empty((n, self.max_order), dtype=np.float64)
        b1_buf = np.empty((n, self.max_order), dtype=np.float64)
        b2_buf = np.empty((n, self.max_order), dtype=np.float64)
        params_c = np.ascontiguousarray(params, dtype=np.float64)
        rmtb_kernel(
            int(ix1),
            int(ix2),
            self.nx,
            n,
            params_c,
            self.order_list,
            self.ncp_list,
            self.ncp_strides,
            self.knots_flat,
            self.knots_offsets,
            self.col_offsets,
            self.nnz_row,
            b0_buf,
            b1_buf,
            b2_buf,
            data,
            rows,
            cols,
        )
        return data, rows, cols

    def _compute_jac_jax(self, ix1, ix2, params):
        jax, jnp = _get_jax()
        if jnp is None:
            return self._compute_jac_numpy(ix1, ix2, params)
        n = params.shape[0]
        basis_list = []
        cols_base = np.zeros(n, dtype=np.int64)
        with _enable_x64(jax):
            for ix in range(self.nx):
                deriv = (1 if ix == ix1 else 0) + (1 if ix == ix2 else 0)
                order = int(self.order_list[ix])
                ncp = int(self.ncp_list[ix])
                fn = _get_rmtb_dim_eval_jax(order, ncp, deriv)
                istart_j, basis_j = fn(
                    jnp.asarray(params[:, ix], dtype=jnp.float64),
                    jnp.asarray(self.knots_list[ix], dtype=jnp.float64),
                )
                istart_np = np.asarray(istart_j, dtype=np.int64)
                basis_np = np.asarray(basis_j, dtype=np.float64)
                basis_list.append(basis_np)
                cols_base += istart_np * int(self.ncp_strides[ix])

        data_2d = basis_list[0]
        for ix in range(1, self.nx):
            data_2d = (data_2d[:, :, None] * basis_list[ix][:, None, :]).reshape(n, -1)
        data = np.array(data_2d.ravel(), dtype=np.float64, copy=True)
        rows = np.repeat(np.arange(n, dtype=np.int32), self.nnz_row)
        cols = (cols_base[:, None] + self.col_offsets[None, :]).ravel().astype(np.int32)
        return data, rows, cols

    def compute_jac(self, ix1, ix2, params, _backend=None):
        """
        Compute sparse B-spline evaluation or derivative Jacobian COO triplets.

        Parameters
        ----------
        ix1 : int
            0-based first derivative dimension index, or ``-1`` for none.
        ix2 : int
            0-based second derivative dimension index, or ``-1`` for none.
        params : array_like
            Normalized coordinates in ``[0, 1]`` of shape ``(n, nx)``.
        _backend : str or None, optional
            Internal engine override (``'auto'``, ``'numpy'``, ``'numba'``, ``'jax'``).

        Returns
        -------
        tuple of (np.ndarray, np.ndarray, np.ndarray)
            ``(data, rows, cols)`` with dtypes ``(float64, int32, int32)``.
        """
        pts = np.asarray(params, dtype=np.float64)
        if pts.ndim == 1:
            if pts.size == 0:
                return (
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.int32),
                )
            pts = pts.reshape((-1, self.nx))
        if pts.shape[0] == 0:
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )

        engine = _resolve_backend(self._backend if _backend is None else _backend)
        if engine == "jax":
            return self._compute_jac_jax(ix1, ix2, pts)
        if engine == "numba":
            return self._compute_jac_numba(ix1, ix2, pts)
        return self._compute_jac_numpy(ix1, ix2, pts)


# ======================================================================
# PureRMTC
# ======================================================================


_RMTC_C2N_1D = np.array(
    [
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, -2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
    ],
    dtype=np.float64,
)


class PureRMTC:
    """
    Pure-Python Tensor-product Cubic Hermite backend replacing C++ ``PyRMTC``.
    Supports NumPy, optional Numba JIT, and optional JAX AD engines.
    """

    def __init__(
        self,
        nx=None,
        lower=None,
        upper=None,
        nelem_list=None,
        nterm_list=None,
        _backend=None,
    ):
        self._backend = _backend
        self.nx = 0
        self.lower = None
        self.upper = None
        self.nelem_list = None
        self.nterm_list = None
        self.nelem = 0
        self.nterm = 0
        self.max_nterm = 0
        self.nelem_strides = None
        self.dxb_dx = None
        if nx is not None:
            self.setup(nx, lower, upper, nelem_list, nterm_list)

    def setup(self, nx, lower, upper, nelem_list, nterm_list):
        self.nx = int(nx)
        self.lower = np.asarray(lower, dtype=np.float64).copy()
        self.upper = np.asarray(upper, dtype=np.float64).copy()
        self.nelem_list = np.asarray(nelem_list, dtype=np.int64).copy()
        self.nterm_list = np.asarray(nterm_list, dtype=np.int64).copy()

        if (
            self.lower.shape[0] != self.nx
            or self.upper.shape[0] != self.nx
            or self.nelem_list.shape[0] != self.nx
            or self.nterm_list.shape[0] != self.nx
        ):
            raise ValueError("All RMTC setup arrays must have length nx.")

        self.nelem = int(np.prod(self.nelem_list))
        self.nterm = int(np.prod(self.nterm_list))
        self.max_nterm = int(np.max(self.nterm_list)) if self.nx > 0 else 0

        strides = np.ones(self.nx, dtype=np.int64)
        for ix in range(self.nx - 2, -1, -1):
            strides[ix] = strides[ix + 1] * self.nelem_list[ix + 1]
        self.nelem_strides = strides

        bma_d2 = (self.upper - self.lower) / self.nelem_list / 2.0
        self.dxb_dx = 1.0 / bma_d2

    def compute_ext_dist(self, x, nterm=None):
        nt = self.nterm if nterm is None else int(nterm)
        return compute_ext_dist(x, self.lower, self.upper, nterm=nt)

    def compute_quadrature_points(self, nelem_list=None):
        elems = self.nelem_list if nelem_list is None else nelem_list
        return compute_quadrature_points(elems, self.lower, self.upper)

    def compute_coeff2nodal(self):
        """
        Compute the ``(nterm, nterm)`` element-level mapping from polynomial
        coefficients to nodal function and derivative values.
        """
        if self.nx == 0:
            return np.ones((1, 1), dtype=np.float64)
        if np.all(self.nterm_list == 4):
            mtx = _RMTC_C2N_1D.copy()
            for _ in range(1, self.nx):
                mtx = np.kron(mtx, _RMTC_C2N_1D)
            return mtx
        raise NotImplementedError(
            "PureRMTC.compute_coeff2nodal only supports 4 terms per dimension (cubic Hermite)."
        )

    def compute_uniq2elem(self):
        """
        Compute COO triplets ``(data, rows, cols)`` mapping unique nodal DOFs
        to per-element nodal DOFs with repetition.
        """
        if not np.all(self.nterm_list == 4):
            raise NotImplementedError(
                "PureRMTC.compute_uniq2elem only supports 4 terms per dimension (cubic Hermite)."
            )
        nx = self.nx
        nelem = self.nelem
        nterm = self.nterm

        derv_map = np.array([0, 0, 1, 1], dtype=np.int64)
        side_map = np.array([0, 1, 0, 1], dtype=np.int64)

        nuniq_list = 1 + self.nelem_list
        nuniq = int(np.prod(nuniq_list))

        nuniq_strides = np.ones(nx, dtype=np.int64)
        for ix in range(nx - 2, -1, -1):
            nuniq_strides[ix] = nuniq_strides[ix + 1] * nuniq_list[ix + 1]
        ndofs_strides = 2 ** np.arange(nx - 1, -1, -1, dtype=np.int64)

        ielem_arr = np.column_stack(
            np.unravel_index(np.arange(nelem, dtype=np.int64), self.nelem_list)
        )
        iterm_arr = np.column_stack(
            np.unravel_index(np.arange(nterm, dtype=np.int64), self.nterm_list)
        )

        iderv = np.sum(derv_map[iterm_arr] * ndofs_strides[None, :], axis=1)
        iside_arr = side_map[iterm_arr]
        iuniq_arr = ielem_arr[:, None, :] + iside_arr[None, :, :]
        iuniq = np.sum(iuniq_arr * nuniq_strides[None, None, :], axis=2)

        cols = (iderv[None, :] * nuniq + iuniq).ravel().astype(np.int32)
        rows = np.arange(nelem * nterm, dtype=np.int32)
        data = np.ones(nelem * nterm, dtype=np.float64)
        return data, rows, cols

    def compute_full_from_block(self, mtx):
        """
        Replicate an ``(nterm, nterm)`` element block across ``nelem`` elements
        into COO triplets ``(data, rows, cols)``.
        """
        mtx_2d = np.asarray(mtx, dtype=np.float64).reshape((self.nterm, self.nterm))
        data = np.tile(mtx_2d.ravel(), self.nelem)
        r_blk, c_blk = np.meshgrid(
            np.arange(self.nterm, dtype=np.int32),
            np.arange(self.nterm, dtype=np.int32),
            indexing="ij",
        )
        offsets = np.arange(self.nelem, dtype=np.int32) * self.nterm
        rows = (offsets[:, None] + r_blk.ravel()[None, :]).ravel()
        cols = (offsets[:, None] + c_blk.ravel()[None, :]).ravel()
        return data, rows, cols

    def _find_interval_numpy(self, ix, x_col):
        a = self.lower[ix]
        b = self.upper[ix]
        num = int(self.nelem_list[ix])

        idx_1b = np.clip(np.ceil((x_col - a) / (b - a) * num).astype(np.int64), 1, num)
        a2 = a + (b - a) * (idx_1b - 1) / float(num)
        b2 = a + (b - a) * idx_1b / float(num)
        bma_d2 = (b2 - a2) / 2.0
        apb_d2 = (a2 + b2) / 2.0
        xbar = np.clip((x_col - apb_d2) / bma_d2, -1.0, 1.0)
        return idx_1b - 1, xbar

    def _compute_jac_numpy(self, ix1, ix2, x):
        n = x.shape[0]
        basis_list = []
        ielem = np.zeros(n, dtype=np.int64)
        for ix in range(self.nx):
            ielem_ix, xbar = self._find_interval_numpy(ix, x[:, ix])
            ielem += ielem_ix * int(self.nelem_strides[ix])

            nt_ix = int(self.nterm_list[ix])
            deriv = (1 if ix == ix1 else 0) + (1 if ix == ix2 else 0)
            b_ix = np.zeros((n, nt_ix), dtype=np.float64)
            if deriv == 0:
                val = np.ones(n, dtype=np.float64)
                for p in range(nt_ix):
                    b_ix[:, p] = val
                    val = val * xbar
            elif deriv == 1:
                inv_bma = self.dxb_dx[ix]
                val = np.ones(n, dtype=np.float64)
                for p in range(1, nt_ix):
                    b_ix[:, p] = float(p) * val * inv_bma
                    val = val * xbar
            else:
                inv_bma = self.dxb_dx[ix]
                val = np.ones(n, dtype=np.float64)
                for p in range(2, nt_ix):
                    b_ix[:, p] = float(p * (p - 1)) * val * inv_bma * inv_bma
                    val = val * xbar
            basis_list.append(b_ix)

        data_2d = basis_list[0]
        for ix in range(1, self.nx):
            data_2d = (data_2d[:, :, None] * basis_list[ix][:, None, :]).reshape(n, -1)
        data = np.array(data_2d.ravel(), dtype=np.float64, copy=True)
        rows = np.repeat(np.arange(n, dtype=np.int32), self.nterm)
        cols = (
            (
                ielem[:, None] * self.nterm
                + np.arange(self.nterm, dtype=np.int64)[None, :]
            )
            .ravel()
            .astype(np.int32)
        )
        return data, rows, cols

    def _compute_jac_numba(self, ix1, ix2, x):
        kernels = _get_numba_kernels()
        if kernels is None:
            return self._compute_jac_numpy(ix1, ix2, x)
        _, rmtc_kernel = kernels
        n = x.shape[0]
        nnz = n * self.nterm
        data = np.empty(nnz, dtype=np.float64)
        rows = np.empty(nnz, dtype=np.int32)
        cols = np.empty(nnz, dtype=np.int32)
        basis_buf = np.empty((n, self.max_nterm), dtype=np.float64)
        x_c = np.ascontiguousarray(x, dtype=np.float64)
        rmtc_kernel(
            int(ix1),
            int(ix2),
            self.nx,
            n,
            x_c,
            self.lower,
            self.upper,
            self.dxb_dx,
            self.nelem_list,
            self.nterm_list,
            self.nelem_strides,
            self.nterm,
            basis_buf,
            data,
            rows,
            cols,
        )
        return data, rows, cols

    def _compute_jac_jax(self, ix1, ix2, x):
        jax, jnp = _get_jax()
        if jnp is None:
            return self._compute_jac_numpy(ix1, ix2, x)
        n = x.shape[0]
        basis_list = []
        ielem = np.zeros(n, dtype=np.int64)
        with _enable_x64(jax):
            for ix in range(self.nx):
                deriv = (1 if ix == ix1 else 0) + (1 if ix == ix2 else 0)
                nt_ix = int(self.nterm_list[ix])
                fn = _get_rmtc_dim_eval_jax(nt_ix, deriv)
                ielem_j, basis_j = fn(
                    jnp.asarray(x[:, ix], dtype=jnp.float64),
                    float(self.lower[ix]),
                    float(self.upper[ix]),
                    int(self.nelem_list[ix]),
                    float(self.dxb_dx[ix]),
                )
                ielem += np.asarray(ielem_j, dtype=np.int64) * int(
                    self.nelem_strides[ix]
                )
                basis_list.append(np.asarray(basis_j, dtype=np.float64))

        data_2d = basis_list[0]
        for ix in range(1, self.nx):
            data_2d = (data_2d[:, :, None] * basis_list[ix][:, None, :]).reshape(n, -1)
        data = np.array(data_2d.ravel(), dtype=np.float64, copy=True)
        rows = np.repeat(np.arange(n, dtype=np.int32), self.nterm)
        cols = (
            (
                ielem[:, None] * self.nterm
                + np.arange(self.nterm, dtype=np.int64)[None, :]
            )
            .ravel()
            .astype(np.int32)
        )
        return data, rows, cols

    def compute_jac(self, ix1, ix2, x, _backend=None):
        """
        Compute sparse cubic Hermite evaluation or derivative Jacobian COO triplets.

        Parameters
        ----------
        ix1 : int
            0-based first derivative dimension index, or ``-1`` for none.
        ix2 : int
            0-based second derivative dimension index, or ``-1`` for none.
        x : array_like
            Physical coordinates of shape ``(n, nx)``.
        _backend : str or None, optional
            Internal engine override (``'auto'``, ``'numpy'``, ``'numba'``, ``'jax'``).

        Returns
        -------
        tuple of (np.ndarray, np.ndarray, np.ndarray)
            ``(data, rows, cols)`` with dtypes ``(float64, int32, int32)``.
        """
        pts = np.asarray(x, dtype=np.float64)
        if pts.ndim == 1:
            if pts.size == 0:
                return (
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.int32),
                )
            pts = pts.reshape((-1, self.nx))
        if pts.shape[0] == 0:
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )

        engine = _resolve_backend(self._backend if _backend is None else _backend)
        if engine == "jax":
            return self._compute_jac_jax(ix1, ix2, pts)
        if engine == "numba":
            return self._compute_jac_numba(ix1, ix2, pts)
        return self._compute_jac_numpy(ix1, ix2, pts)
