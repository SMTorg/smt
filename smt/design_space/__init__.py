from smt.design_space.design_space import (
    CategoricalVariable,
    BaseDesignSpace,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
)

try:
    from smt_design_space_ext import (
        DesignSpace,
        ensure_design_space,
    )

    HAS_DESIGN_SPACE_EXT = True
except ImportError:
    from .design_space import (
        DesignSpace,
        ensure_design_space,
    )

    HAS_DESIGN_SPACE_EXT = False

__all__ = [
    "BaseDesignSpace",
    "FloatVariable",
    "IntegerVariable",
    "OrdinalVariable",
    "CategoricalVariable",
    "DesignSpace",
    "ensure_design_space",
    "HAS_DESIGN_SPACE_EXT",
]
