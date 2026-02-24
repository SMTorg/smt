"""
Backward-compatibility shim.

All symbols that used to live in ``smt.utils.kriging`` have been
reorganised into the ``smt.surrogate_models.krg_based`` sub-package:

* ``smt.surrogate_models.krg_based.kernel_types``  – enums
* ``smt.surrogate_models.krg_based.distances``      – distance / PLS / basis functions
* ``smt.surrogate_models.krg_based.mixed_int_corr`` – mixed-integer helpers

This module re-exports every public name so that existing ``from
smt.utils.kriging import …`` statements keep working.
"""

# --- enums ---------------------------------------------------------------
from smt.surrogate_models.krg_based.kernel_types import (  # noqa: F401
    MixHrcKernelType,
    MixIntKernelType,
)

# --- distances / PLS / basis ---------------------------------------------
from smt.surrogate_models.krg_based.distances import (  # noqa: F401
    njit_use,
    componentwise_distance,
    componentwise_distance_PLS,
    constant,
    cross_distances,
    differences,
    ge_compute_pls,
    linear,
    quadratic,
)

# --- mixed-integer helpers -----------------------------------------------
from smt.surrogate_models.krg_based.mixed_int_corr import (  # noqa: F401
    apply_the_algebraic_distance_to_the_decreed_variable,
    compute_D_cat,
    compute_D_num,
    compute_X_cont,
    compute_X_cross,
    cross_levels,
    cross_levels_homo_space,
    gower_componentwise_distances,
    matrix_data_corr_levels_cat_matrix,
    matrix_data_corr_levels_cat_mod,
    matrix_data_corr_levels_cat_mod_comps,
)
