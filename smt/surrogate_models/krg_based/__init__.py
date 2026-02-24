"""Kriging-based surrogate model subpackage.

Re-exports the main public symbols so that existing code like::

    from smt.surrogate_models.krg_based import KrgBased, MixIntKernelType

continues to work without modification.
"""

from smt.surrogate_models.krg_based.kernel_types import (
    MixHrcKernelType,
    MixIntKernelType,
)
from smt.surrogate_models.krg_based.krg_based import (
    KrgBased,
)
from smt.surrogate_models.krg_based.mixed_int_corr import (
    compute_n_param,
    compute_X_cont,
    compute_X_cross,
    cross_levels,
    cross_levels_homo_space,
    gower_componentwise_distances,
)
from smt.surrogate_models.krg_based.distances import (
    componentwise_distance,
    componentwise_distance_PLS,
    constant,
    cross_distances,
    differences,
    ge_compute_pls,
    linear,
    quadratic,
)
from smt.surrogate_models.krg_based.krg_sampling import (
    covariance_matrix,
    eig_grid,
    evaluate_eigen_function,
    sample_eigen,
    sample_trajectory,
)

__all__ = [
    "KrgBased",
    "MixHrcKernelType",
    "MixIntKernelType",
    "componentwise_distance",
    "componentwise_distance_PLS",
    "compute_n_param",
    "compute_X_cont",
    "compute_X_cross",
    "constant",
    "covariance_matrix",
    "cross_distances",
    "cross_levels",
    "cross_levels_homo_space",
    "differences",
    "eig_grid",
    "evaluate_eigen_function",
    "ge_compute_pls",
    "gower_componentwise_distances",
    "linear",
    "quadratic",
    "sample_eigen",
    "sample_trajectory",
]
