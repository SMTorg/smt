import importlib

spec = importlib.util.find_spec("smt_design_space")
if spec:
    HAS_DESIGN_SPACE_EXT = True
    HAS_CONFIG_SPACE = True
    HAS_ADSG = True
else:
    HAS_DESIGN_SPACE_EXT = False
    HAS_CONFIG_SPACE = False
    HAS_ADSG = False


if HAS_DESIGN_SPACE_EXT:
    from smt_design_space.design_space import (
        CategoricalVariable,
        DesignSpace,
        BaseDesignSpace,
        FloatVariable,
        IntegerVariable,
        OrdinalVariable,
        ensure_design_space,
    )

else:
    from smt.design_space.design_space import (
        CategoricalVariable,
        DesignSpace,
        FloatVariable,
        IntegerVariable,
        OrdinalVariable,
        ensure_design_space,
        BaseDesignSpace,
    )

if HAS_DESIGN_SPACE_EXT:
    from smt_design_space.design_space import DesignSpaceGraph
else:

    class DesignSpaceGraph:
        pass


__all__ = [
    "HAS_DESIGN_SPACE_EXT",
    "HAS_CONFIG_SPACE",
    "HAS_ADSG",
    "BaseDesignSpace",
    "DesignSpace",
    "FloatVariable",
    "IntegerVariable",
    "OrdinalVariable",
    "CategoricalVariable",
    "ensure_design_space",
]
