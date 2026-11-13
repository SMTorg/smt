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

The public model APIs (options, outputs, shapes) are unchanged.
"""

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


def _resolve_backend(_backend=None):
    if _backend is None or _backend == "auto":
        if _parse_bool_env("USE_NUMBA_JIT", 0) and has_numba():
            return "numba"
        return "numpy"
    if _backend == "numba":
        return "numba" if has_numba() else "numpy"
    if _backend == "numpy":
        return "numpy"
    raise ValueError(
        f"Unsupported backend {_backend!r}; expected one of (None, 'auto', 'numpy', 'numba')."
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
        if kx is None:
            jac = w / w.sum(axis=1, keepdims=True)
        else:
            dr2 = 2.0 * (x[:, kx, None] - xt[None, :, kx])
            dw = -p / 2.0 * r2 ** (-p / 2.0 - 1.0) * dr2
            s = w.sum(axis=1, keepdims=True)
            ds = dw.sum(axis=1, keepdims=True)
            jac = (dw * s - w * ds) / s**2

    if exact.any():
        if kx is None:
            jac[exact] = 0.0
            first = hit[exact].argmax(axis=1)  # first coincident training point
            jac[np.flatnonzero(exact), first] = 1.0
        else:
            jac[exact] = 0.0
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
            s = 0.0
            ds = 0.0
            for it in range(nt):
                r2 = 0.0
                for ix in range(nx):
                    d = x[i, ix] - xt[it, ix]
                    r2 += d * d
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
    if _resolve_backend(_backend) == "numba":
        jac = np.empty((x.shape[0], xt.shape[0]))
        _, idw_k = _get_numba_kernels()
        idw_k(x, xt, float(p), 0 if kx is None else int(kx), kx is not None, jac)
        return jac
    return _idw_np(x, xt, float(p), kx)
