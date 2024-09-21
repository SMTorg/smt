import importlib

spec = importlib.util.find_spec("SMTDesignSpace")
if spec:
    HAS_SMTDesignSpace = True
    HAS_CONFIG_SPACE = True
    HAS_ADSG = True
else:
    HAS_SMTDesignSpace = False
    HAS_CONFIG_SPACE = False
    HAS_ADSG = False


if HAS_SMTDesignSpace:
    from SMTDesignSpace.design_space import (
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

if HAS_SMTDesignSpace:
    from SMTDesignSpace.design_space import ArchDesignSpaceGraph
else:

    class ArchDesignSpaceGraph:
        pass


__all__ = [
    "HAS_SMTDesignSpace",
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
