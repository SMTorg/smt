# -*- coding: utf-8 -*-
"""
Backward-compatibility shim. CoopCompKRG has moved to smt.surrogate_models.cckrg.
"""

from smt.surrogate_models.cckrg import CooperativeCobylaOptimizer, CoopCompKRG

__all__ = ["CooperativeCobylaOptimizer", "CoopCompKRG"]
