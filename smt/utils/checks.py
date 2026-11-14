"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

import sys
import numpy as np


def _is_jax_tracer(x):
    jax = sys.modules.get("jax")
    return jax is not None and isinstance(x, jax.core.Tracer)


def _enable_jax_x64(jax):
    """Return a context manager temporarily enabling JAX 64-bit precision."""
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    return enable(True)


def is_arraylike(array):
    """
    Return True if ``array`` is a NumPy array or an object exposing the NumPy
    array protocol (``__array__`` or ``__array_interface__``), e.g. JAX arrays,
    PyTorch CPU tensors, xarray / pandas objects. NumPy scalars are rejected.

    SMT only needs read-only access to such inputs, so they are converted to
    NumPy arrays without requiring any optional dependency (see issue #827).
    """
    if _is_jax_tracer(array):
        return False
    if isinstance(array, np.ndarray):
        return True
    if isinstance(array, np.generic) or isinstance(array, (str, bytes)):
        return False
    return hasattr(array, "__array__") or hasattr(array, "__array_interface__")


def ensure_float64_array(array):
    """
    Convert an array or array-like to a writeable float64 NumPy array
    if its dtype is floating-point. For non-ndarray inputs, creates a writeable
    NumPy array.
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if array.dtype.kind == "f" and array.dtype != np.float64:
        array = array.astype(np.float64)
    return array


def ensure_2d_array(array, name, copy=False):
    if _is_jax_tracer(array):
        raise TypeError(
            f"Cannot pass a JAX Tracer to '{name}'. SMT algorithms run on CPU using NumPy "
            "and require concrete numerical values, not symbolic tracers (e.g. inside jax.jit or jax.grad)."
        )

    if isinstance(array, (str, bytes, dict, set)):
        raise ValueError(f"{name} must be a NumPy array or an array-like object")

    if not isinstance(array, np.ndarray):
        try:
            array = np.array(array)
        except Exception:
            raise ValueError(f"{name} must be a NumPy array or an array-like object")
    elif copy:
        array = np.copy(array)

    if array.dtype.kind in ("U", "S"):
        raise ValueError(f"{name} must be a NumPy array or an array-like object")

    if array.dtype.kind == "O":
        try:
            array = array.astype(np.float64)
        except Exception:
            pass

    if array.dtype.kind == "f" and array.dtype != np.float64:
        array = array.astype(np.float64)

    array = np.atleast_2d(array.T).T

    if len(array.shape) != 2:
        raise ValueError(f"{name} must have a rank of 1 or 2")

    return array


def check_support(sm, name, fail=False):
    if not sm.supports[name] or fail:
        class_name = sm.__class__.__name__
        raise NotImplementedError(f"{class_name} does not support {name}")


def check_nx(sm, x, nx=None):
    if nx is None:
        nx = sm.nx
    class_name = sm.__class__.__name__
    if x.shape[1] != nx:
        if nx == 1:
            raise ValueError("x should have shape [:, 1] or [:]")
        else:
            raise ValueError(
                f"X has {x.shape[1]} features, but {class_name} is expecting {nx} features as input"
            )
