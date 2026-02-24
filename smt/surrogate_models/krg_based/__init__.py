"""Kriging-based surrogate model subpackage.

Re-exports the main public symbols so that existing code like::

    from smt.surrogate_models.krg_based import KrgBased, MixIntKernelType

continues to work without modification.
"""

from smt.surrogate_models.krg_based.krg_based import (
    KrgBased,
    MixIntKernelType,
    compute_n_param,
)

__all__ = [
    "KrgBased",
    "MixIntKernelType",
    "compute_n_param",
]
