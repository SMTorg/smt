"""
Backward-compatible re-export shim.

All symbols have moved to their canonical locations:

* Kriging sampling:  ``smt.surrogate_models.krg_based.krg_sampling``
* Quadrature grids:  ``smt.utils.quadrature``

This module re-exports every public name so that existing ``from
smt.utils.krg_sampling import …`` statements keep working.
"""

# --- kriging-coupled functions -------------------------------------------
from smt.surrogate_models.krg_based.krg_sampling import (  # noqa: F401
    covariance_matrix,
    eig_grid,
    evaluate_eigen_function,
    sample_eigen,
    sample_trajectory,
)

# --- quadrature functions ------------------------------------------------
from smt.utils.quadrature import (  # noqa: F401
    gauss_legendre_grid,
    rectangular_grid,
    simpson_grid,
    simpson_weigths,
)
