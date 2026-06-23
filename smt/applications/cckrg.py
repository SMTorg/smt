# -*- coding: utf-8 -*-
"""
Deprecated. CoopCompKRG has moved to smt.surrogate_models.cckrg.
"""

import warnings

from smt.surrogate_models.cckrg import CooperativeCobylaOptimizer, CoopCompKRG

warnings.warn(
    "Importing CoopCompKRG from smt.applications is deprecated. "
    "Use 'from smt.surrogate_models import CoopCompKRG' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["CooperativeCobylaOptimizer", "CoopCompKRG"]
