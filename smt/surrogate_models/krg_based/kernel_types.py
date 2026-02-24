"""
Kernel type enumerations for Kriging-based surrogate models.

This module is the single source of truth for the categorical and
hierarchical kernel type enums used throughout the ``krg_based``
subpackage.  Keeping them in a standalone leaf module eliminates
circular-import issues that previously required lazy-import hacks.

Classes
-------
MixIntKernelType
    Categorical kernel type for mixed-integer Kriging.
MixHrcKernelType
    Hierarchical (conditionally-acting) kernel type.
"""

from enum import Enum


class MixIntKernelType(Enum):
    """Categorical kernel type for mixed-integer Kriging models."""

    EXP_HOMO_HSPHERE = "EXP_HOMO_HSPHERE"
    HOMO_HSPHERE = "HOMO_HSPHERE"
    CONT_RELAX = "CONT_RELAX"
    GOWER = "GOWER"
    COMPOUND_SYMMETRY = "COMPOUND_SYMMETRY"


class MixHrcKernelType(Enum):
    """Hierarchical kernel type for conditionally-acting variables."""

    ARC_KERNEL = "ARC_KERNEL"
    ALG_KERNEL = "ALG_KERNEL"
