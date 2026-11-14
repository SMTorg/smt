"""
Backward-compatibility shim.

``RMTB`` now lives in :mod:`smt.surrogate_models.rmts`; prefer
``RMTS(spline_type="B-Spline")``. This module keeps
``from smt.surrogate_models.rmtb import RMTB`` working for downstream code.
"""

from smt.surrogate_models.rmts import RMTB

__all__ = ["RMTB"]
