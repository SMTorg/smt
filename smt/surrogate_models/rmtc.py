"""
Backward-compatibility shim.

``RMTC`` now lives in :mod:`smt.surrogate_models.rmts`; prefer
``RMTS(spline_type="Cubic_Hermite_Spline")``. This module keeps
``from smt.surrogate_models.rmtc import RMTC`` working for downstream code.
"""

from smt.surrogate_models.rmts import RMTC

__all__ = ["RMTC"]
