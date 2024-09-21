try:
    from SMTDesignSpace.design_space import BaseDesignSpace

    HAS_SMTDesignSpace = True
    HAS_CONFIG_SPACE = True
    HAS_ADSG = True
except ImportError:
    HAS_SMTDesignSpace = False
    HAS_CONFIG_SPACE = False
    HAS_ADSG = False


if HAS_SMTDesignSpace:
    from SMTDesignSpace import design_space as ds
    from SMTDesignSpace.design_space import (
        HAS_CONFIG_SPACE,
        HAS_ADSG,
        ArchDesignSpaceGraph,
        BaseDesignSpace,
        CategoricalVariable,
        DesignSpace,
        FloatVariable,
        IntegerVariable,
        OrdinalVariable,
        ensure_design_space,
        _convert_adsg_to_legacy,
    )

else:
    from smt.design_space import design_space as ds
    from smt.design_space.design_space import (
        HAS_CONFIG_SPACE,
        HAS_ADSG,
        BaseDesignSpace,
        CategoricalVariable,
        DesignSpace,
        FloatVariable,
        IntegerVariable,
        OrdinalVariable,
        ensure_design_space,
    )


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
