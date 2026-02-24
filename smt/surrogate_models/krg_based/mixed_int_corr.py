"""
Mixed-integer correlation logic for Kriging-based surrogate models.

This module encapsulates the kernel correlation computation for
mixed-integer (categorical + continuous) design spaces. It extracts what
was previously inline in :class:`KrgBased._matrix_data_corr` and related
helpers into a self-contained class, following the Single Responsibility
Principle.

Classes
-------
MixedIntegerCorrelation
    Computes kernel correlations for mixed-integer design spaces.

Functions
---------
compute_n_param
    Pure function: returns the number of hyperparameters needed for a given
    kernel type and design space.
correct_distances_cat_decreed
    Corrects distances for conditionally-acting categorical variables.
"""

import numpy as np

from smt.design_space import CategoricalVariable

from .kernel_types import MixHrcKernelType, MixIntKernelType
from .distances import (
    njit_use,
    prange,
    componentwise_distance,
    componentwise_distance_PLS,
)


# ======================================================================
# Mixed-integer helper functions (absorbed from smt.utils.kriging)
# ======================================================================


def cross_levels(X, ij, design_space, y=None):
    """
    Returns the levels corresponding to the indices i and j of the vectors in X and the number of levels.
    Parameters
    ----------

    X: np.ndarray [n_obs, dim]
            - The input variables.
    y: np.ndarray [n_y, dim]
            - The training data.
    ij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
            - The indices i and j of the vectors in X associated to the cross-
              distances in D.
    design_space: BaseDesignSpace
        - The design space definition
    Returns
    -------

     Lij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
            - The levels corresponding to the indices i and j of the vectors in X.
     n_levels: np.ndarray
            - The number of levels for every categorical variable.
    """

    n_levels = []
    for dv in design_space.design_variables:
        if isinstance(dv, CategoricalVariable):
            n_levels.append(dv.n_values)
    n_levels = np.array(n_levels)
    n_var = n_levels.shape[0]
    n, _ = ij.shape
    X_cont, cat_features = compute_X_cont(X, design_space)
    X_cat = X[:, cat_features]

    if y is None:
        Lij = _cross_levels_mat(n_var, n, X_cat, ij)
    else:
        Lij = _cross_levels_mat_y(n_var, n, X_cat, ij, y, cat_features)

    return Lij, n_levels


@njit_use(parallel=True)
def _cross_levels_mat(n_var, n, X_cat, ij):
    Lij = np.zeros((n_var, n, 2))
    for k in prange(n_var):
        for ll in prange(n):
            i, j = ij[ll]
            Lij[k][ll][0] = X_cat[i, k]
            Lij[k][ll][1] = X_cat[j, k]
    return Lij


@njit_use(parallel=True)
def _cross_levels_mat_y(n_var, n, X_cat, ij, y, cat_features):
    Lij = np.zeros((n_var, n, 2))
    y_cat = y[:, cat_features]
    for k in prange(n_var):
        for ll in prange(n):
            i, j = ij[ll]
            Lij[k][ll][0] = X_cat[i, k]
            Lij[k][ll][1] = y_cat[j, k]
    return Lij


def cross_levels_homo_space(X, ij, y=None):
    """
    Computes the nonzero componentwise (or Hadamard) product between the vectors in X
    Parameters
    ----------

    X: np.ndarray [n_obs, dim]
            - The input variables.
    y: np.ndarray [n_y, dim]
            - The training data.
    ij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
            - The indices i and j of the vectors in X associated to the cross-
              distances in D.

    Returns
    -------
     dx: np.ndarray [n_obs * (n_obs - 1) / 2,dim]
            - The Hadamard product between the vectors in X.
    """
    dim = np.shape(X)[1]
    n, _ = ij.shape
    dx = np.zeros((n, dim))
    for ll in range(n):
        i, j = ij[ll]
        if y is None:
            dx[ll] = X[i] * X[j]
        else:
            dx[ll] = X[i] * y[j]

    return dx


def compute_X_cont(x, design_space):
    """
    Gets the X_cont part of a vector x for mixed integer
    Parameters
    ----------
    x: np.ndarray [n_obs, dim]
            - The input variables.
    design_space : BaseDesignSpace
        - The design space definition
    Returns
    -------
    X_cont: np.ndarray [n_obs, dim_cont]
         - The non categorical values of the input variables.
    cat_features: np.ndarray [dim]
        -  Indices of the categorical input dimensions.

    """
    is_cat_mask = design_space.is_cat_mask
    return x[:, ~is_cat_mask], is_cat_mask


def gower_componentwise_distances(
    X, x_is_acting, design_space, hierarchical_kernel, y=None, y_is_acting=None
):
    """
    Computes the nonzero Gower-distances componentwise between the vectors
    in X.
    Parameters
    ----------
    X: np.ndarray [n_obs, dim]
        - The input variables.
    x_is_acting: np.ndarray [n_obs, dim]
        - is_acting matrix for the inputs
    design_space : BaseDesignSpace
        - The design space definition
    y: np.ndarray [n_y, dim]
        - The training data
    y_is_acting: np.ndarray [n_y, dim]
        - is_acting matrix for the training points
    Returns
    -------
    D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The gower distances between the vectors in X.
    ij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
            - The indices i and j of the vectors in X associated to the cross-
              distances in D.
    X_cont: np.ndarray [n_obs, dim_cont]
         - The non categorical values of the input variables.
    """
    X = X.astype(np.float64)
    Xt = X
    X_cont, cat_features = compute_X_cont(Xt, design_space)
    is_decreed = design_space.is_conditionally_acting

    # function checks
    if y is None:
        Y = X
        y_is_acting = x_is_acting
    else:
        Y = y
        if y_is_acting is None:
            raise ValueError("Expected y_is_acting because y is given")

    if not isinstance(X, np.ndarray):
        if not np.array_equal(X.columns, Y.columns):
            raise TypeError("X and Y must have same columns!")
    else:
        if not X.shape[1] == Y.shape[1]:
            raise TypeError("X and Y must have same y-dim!")

    if x_is_acting.shape != X.shape or y_is_acting.shape != Y.shape:
        raise ValueError("is_acting matrices must have same shape as X!")

    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)

    Z = np.concatenate((X, Y))
    z_is_acting = np.concatenate((x_is_acting, y_is_acting))
    Z_cat = Z[:, cat_features]

    x_index = range(0, x_n_rows)
    y_index = range(x_n_rows, x_n_rows + y_n_rows)
    X_cat = Z_cat[x_index,]
    Y_cat = Z_cat[y_index,]

    # This is to normalize the numeric values between 0 and 1.
    Z_num = Z[:, ~cat_features]
    z_num_is_acting = z_is_acting[:, ~cat_features]
    num_is_decreed = is_decreed[~cat_features]
    num_bounds = design_space.get_num_bounds()[~cat_features, :]
    Z_scale = 1
    if num_bounds.shape[0] > 0:
        Z_offset = num_bounds[:, 0]
        Z_max = num_bounds[:, 1]
        Z_scale = Z_max - Z_offset
        Z_num = (Z_num - Z_offset) / Z_scale
    X_num = Z_num[x_index,]
    Y_num = Z_num[y_index,]
    x_num_is_acting = z_num_is_acting[x_index,]
    y_num_is_acting = z_num_is_acting[y_index,]

    # x_cat_is_acting : activeness vector delta
    # X_cat( not(x_cat_is_acting)) = 0 ###IMPUTED TO FIRST VALUE IN LIST (index 0)
    D_cat = compute_D_cat(X_cat, Y_cat, y)
    D_num, ij = compute_D_num(
        X_num,
        Y_num,
        x_num_is_acting,
        y_num_is_acting,
        num_is_decreed,
        y,
        hierarchical_kernel,
    )
    D_num_out = D_num * Z_scale
    D = np.concatenate((D_cat, D_num), axis=1) * 0
    D[:, np.logical_not(cat_features)] = D_num
    D[:, cat_features] = D_cat
    if y is not None:
        return D, D_num_out
    else:
        return D, ij.astype(np.int32), X_cont, D_num_out


@njit_use(parallel=True)
def compute_D_cat(X_cat, Y_cat, y):
    nx_samples, n_features = X_cat.shape
    ny_samples, n_features = Y_cat.shape
    n_nonzero_cross_dist = nx_samples * ny_samples
    if y is None:
        n_nonzero_cross_dist = nx_samples * (nx_samples - 1) // 2
    D_cat = np.zeros((n_nonzero_cross_dist, n_features))
    indD = 0
    k1max = nx_samples
    if y is None:
        k1max = nx_samples - 1
    for k1 in range(k1max):
        k2max = ny_samples
        if y is None:
            k2max = ny_samples - k1 - 1
        for k2 in prange(k2max):
            l2 = k2
            if y is None:
                l2 = k2 + k1 + 1
            D_cat[indD + k2] = X_cat[k1] != Y_cat[l2]
        indD += k2max
    return D_cat


@njit_use()  # setting parallel=True results in a stack overflow
def compute_D_num(
    X_num,
    Y_num,
    x_num_is_acting,
    y_num_is_acting,
    num_is_decreed,
    y,
    hierarchical_kernel,
):
    nx_samples, n_features = X_num.shape
    ny_samples, n_features = Y_num.shape
    n_nonzero_cross_dist = nx_samples * ny_samples
    if y is None:
        n_nonzero_cross_dist = nx_samples * (nx_samples - 1) // 2
    D_num = np.zeros((n_nonzero_cross_dist, n_features))
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int32)
    ll_1 = 0
    indD = 0
    k1max = nx_samples
    if y is None:
        k1max = nx_samples - 1
    for k1 in range(k1max):
        k2max = ny_samples
        if y is None:
            k2max = ny_samples - k1 - 1
            ll_0 = ll_1
            ll_1 = ll_0 + nx_samples - k1 - 1
            ij[ll_0:ll_1, 0] = k1
            ij[ll_0:ll_1, 1] = np.arange(k1 + 1, nx_samples)
        for k2 in range(k2max):
            l2 = k2
            if y is None:
                l2 = k2 + k1 + 1
            D_num[indD] = np.abs(X_num[k1] - Y_num[l2])
            indD += 1

    if np.any(num_is_decreed):
        D_num = apply_the_algebraic_distance_to_the_decreed_variable(
            X_num,
            Y_num,
            x_num_is_acting,
            y_num_is_acting,
            num_is_decreed,
            y,
            D_num,
            hierarchical_kernel,
        )

    return D_num, ij


@njit_use()  # setting parallel=True results in a stack overflow
def apply_the_algebraic_distance_to_the_decreed_variable(
    X_num,
    Y_num,
    x_num_is_acting,
    y_num_is_acting,
    num_is_decreed,
    y,
    D_num,
    hierarchical_kernel,
):
    nx_samples, n_features = X_num.shape
    ny_samples, n_features = Y_num.shape

    indD = 0
    k1max = nx_samples
    if y is None:
        k1max = nx_samples - 1
    for k1 in range(k1max):
        k2max = ny_samples
        if y is None:
            k2max = ny_samples - k1 - 1
        x_k1_acting = x_num_is_acting[k1]
        for k2 in range(k2max):
            l2 = k2
            if y is None:
                l2 = k2 + k1 + 1
            abs_delta = np.abs(X_num[k1] - Y_num[l2])
            y_l2_acting = y_num_is_acting[l2]

            # Calculate the distances between the decreed (aka conditionally acting) variables
            if hierarchical_kernel == MixHrcKernelType.ALG_KERNEL:
                abs_delta[num_is_decreed] = (
                    2
                    * np.abs(X_num[k1][num_is_decreed] - Y_num[l2][num_is_decreed])
                    / (
                        np.sqrt(1 + X_num[k1][num_is_decreed] ** 2)
                        * np.sqrt(1 + Y_num[l2][num_is_decreed] ** 2)
                    )
                )
            elif hierarchical_kernel == MixHrcKernelType.ARC_KERNEL:
                abs_delta[num_is_decreed] = np.sqrt(2) * np.sqrt(
                    1
                    - np.cos(
                        np.pi
                        * np.abs(X_num[k1][num_is_decreed] - Y_num[l2][num_is_decreed])
                    )
                )

            # Set distances for non-acting variables: 0 if both are non-acting, 1 if only one is non-acting
            both_non_acting = num_is_decreed & ~(x_k1_acting | y_l2_acting)
            abs_delta[both_non_acting] = 0.0

            either_acting = num_is_decreed & (x_k1_acting != y_l2_acting)
            abs_delta[either_acting] = 1.0

            D_num[indD] = abs_delta
            indD += 1
    return D_num


def compute_X_cross(X, n_levels):
    """
    Computes the full space cross-relaxation of the input X for
    the homoscedastic hypersphere kernel.
    Parameters
    ----------
    X: np.ndarray [n_obs, 1]
            - The input variables.
    n_levels: np.ndarray
            - The number of levels for the categorical variable.
    Returns
    -------
    Zeta: np.ndarray [n_obs, n_levels * (n_levels - 1) / 2]
         - The non categorical values of the input variables.
    """

    dim = int(n_levels * (n_levels - 1) / 2)
    nt = len(X)
    Zeta = np.zeros((nt, dim))
    k = 0
    for i in range(n_levels):
        for j in range(n_levels):
            if j > i:
                s = 0
                for x in X:
                    if int(x) == i or int(x) == j:
                        Zeta[s, k] = 1
                    s += 1
                k += 1

    return Zeta


@njit_use(parallel=True)
def matrix_data_corr_levels_cat_matrix(
    i, n_levels, theta_cat, theta_bounds, is_ehh: bool
):
    Theta_mat = np.zeros((n_levels[i], n_levels[i]), dtype=np.float64)
    L = np.zeros((n_levels[i], n_levels[i]))
    v = 0
    for j in range(n_levels[i]):
        for k in range(n_levels[i] - j):
            if j == k + j:
                Theta_mat[j, k + j] = 1.0
            else:
                Theta_mat[j, k + j] = theta_cat[v].item()
                Theta_mat[k + j, j] = theta_cat[v].item()
                v = v + 1

    for j in range(n_levels[i]):
        for k in range(n_levels[i] - j):
            if j == k + j:
                if j == 0:
                    L[j, k + j] = 1

                else:
                    L[j, k + j] = 1
                    for ll in range(j):
                        L[j, k + j] = L[j, k + j] * np.sin(Theta_mat[j, ll])

            else:
                if j == 0:
                    L[k + j, j] = np.cos(Theta_mat[k, 0])
                else:
                    L[k + j, j] = np.cos(Theta_mat[k + j, j])
                    for ll in range(j):
                        L[k + j, j] = L[k + j, j] * np.sin(Theta_mat[k + j, ll])

    T = np.dot(L, L.T)

    if is_ehh:
        T = (T - 1) * theta_bounds[1] / 2
        T = np.exp(2 * T)
    k = (1 + np.exp(-theta_bounds[1])) / np.exp(-theta_bounds[0])
    T = (T + np.exp(-theta_bounds[1])) / (k)
    return T


@njit_use()
def matrix_data_corr_levels_cat_mod(i, Lij, r_cat, T, has_cat_kernel):
    for k in range(np.shape(Lij[i])[0]):
        indi = int(Lij[i][k][0])
        indj = int(Lij[i][k][1])

        if indi == indj:
            r_cat[k] = 1.0
        else:
            if has_cat_kernel:
                r_cat[k] = T[indi, indj]


@njit_use()
def matrix_data_corr_levels_cat_mod_comps(
    i, Lij, r_cat, n_levels, T, d_cat_i, has_cat_kernel
):
    for k in range(np.shape(Lij[i])[0]):
        indi = int(Lij[i][k][0])
        indj = int(Lij[i][k][1])

        if indi == indj:
            r_cat[k] = 1.0
        else:
            if has_cat_kernel:
                Theta_i_red = np.zeros(int((n_levels[i] - 1) * n_levels[i] / 2))
                indmatvec = 0
                for j in range(n_levels[i]):
                    for ll in range(n_levels[i]):
                        if ll > j:
                            Theta_i_red[indmatvec] = T[j, ll]
                            indmatvec += 1
                kval_cat = 0
                for indijk in range(len(Theta_i_red)):
                    kval_cat += np.multiply(
                        Theta_i_red[indijk], d_cat_i[k : k + 1][0][indijk]
                    )
                r_cat[k] = kval_cat


# ======================================================================
# Original mixed_int_corr.py content
# ======================================================================


def compute_n_param(design_space, cat_kernel, d, n_comp, mat_dim):
    """Return the number of hyperparameters needed for the given kernel/design space.

    Parameters
    ----------
    design_space : BaseDesignSpace
        Design space definition.
    cat_kernel : MixIntKernelType
        Categorical kernel type.
    d : int
        ``n_comp`` or ``nx``.
    n_comp : int or None
        Number of PLS components, or ``None`` if PLS is not used.
    mat_dim : list or None
        Per-variable PLS component counts for matrix kernels, or ``None``.

    Returns
    -------
    int
        Number of hyperparameters.
    """
    n_param = design_space.n_dv
    if n_comp is not None:
        n_param = d
        if cat_kernel == MixIntKernelType.CONT_RELAX:
            return n_param
        if mat_dim is not None:
            return int(np.sum([i * (i - 1) / 2 for i in mat_dim]) + n_param)
    if cat_kernel in [MixIntKernelType.GOWER, MixIntKernelType.COMPOUND_SYMMETRY]:
        return n_param
    for i, dv in enumerate(design_space.design_variables):
        if isinstance(dv, CategoricalVariable):
            n_values = dv.n_values
            if design_space.n_dv == d:
                n_param -= 1
            if cat_kernel in [
                MixIntKernelType.EXP_HOMO_HSPHERE,
                MixIntKernelType.HOMO_HSPHERE,
            ]:
                n_param += int(n_values * (n_values - 1) / 2)
            if cat_kernel == MixIntKernelType.CONT_RELAX:
                n_param += int(n_values)
    return n_param


def correct_distances_cat_decreed(
    D,
    is_acting,
    listcatdecreed,
    ij,
    design_space,
    cat_features,
    n_levels,
    X2_offset=None,
    X2_scale=None,
    is_acting_y=None,
    mixint_type=None,
):
    """Correct distances for conditionally-acting (decreed) categorical variables.

    For pairs where one variable is acting and the other is not, distances are
    set to fixed sentinel values. For both-acting pairs, distances are scaled
    by sqrt(2).

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (modified in-place).
    is_acting : np.ndarray
        Acting status for training (or prediction) points.
    listcatdecreed : np.ndarray
        Boolean mask: which categorical features are conditionally acting.
    ij : np.ndarray
        Index pairs.
    design_space : BaseDesignSpace
        Design space definition.
    cat_features : np.ndarray
        Boolean mask of categorical feature indices.
    n_levels : np.ndarray
        Number of levels per categorical variable.
    X2_offset : np.ndarray, optional
        Offset for CONT_RELAX standardisation.
    X2_scale : np.ndarray, optional
        Scale for CONT_RELAX standardisation.
    is_acting_y : np.ndarray, optional
        Acting status for Y points (prediction case).
    mixint_type : MixIntKernelType, optional
        Kernel type (CONT_RELAX or GOWER).

    Returns
    -------
    np.ndarray
        The corrected distance matrix ``D``.
    """
    if mixint_type is None:
        mixint_type = MixIntKernelType.CONT_RELAX

    indjcat = -1
    for j in listcatdecreed:
        indjcat = indjcat + 1
        if j:
            indicat = -1
            indices = 0
            for v in range(len(design_space.design_variables)):
                if isinstance(design_space.design_variables[v], CategoricalVariable):
                    indicat = indicat + 1
                    if indicat == indjcat:
                        ia2 = np.zeros((len(ij), 2), dtype=bool)
                        if is_acting_y is None:
                            ia2 = (is_acting[:, cat_features][:, indjcat])[ij]
                        else:
                            ia2[:, 0] = (is_acting[:, cat_features][:, indjcat])[
                                ij[:, 0]
                            ]
                            ia2[:, 1] = (is_acting_y[:, cat_features][:, indjcat])[
                                ij[:, 1]
                            ]

                        act_inact = ia2[:, 0] ^ ia2[:, 1]
                        act_act = ia2[:, 0] & ia2[:, 1]

                        if mixint_type == MixIntKernelType.CONT_RELAX:
                            val_act = (
                                np.array([1] * n_levels[indjcat])
                                - X2_offset[indices : indices + n_levels[indjcat]]
                            ) / X2_scale[indices : indices + n_levels[indjcat]] - (
                                np.array([0] * n_levels[indjcat])
                                - X2_offset[indices : indices + n_levels[indjcat]]
                            ) / X2_scale[indices : indices + n_levels[indjcat]]
                            D[:, indices : indices + n_levels[indjcat]][act_inact] = (
                                val_act
                            )
                            D[:, indices : indices + n_levels[indjcat]][act_act] = (
                                np.sqrt(2)
                                * D[:, indices : indices + n_levels[indjcat]][act_act]
                            )
                        elif mixint_type == MixIntKernelType.GOWER:
                            D[:, indices : indices + 1][act_inact] = (
                                n_levels[indjcat] * 0.5
                            )
                            D[:, indices : indices + 1][act_act] = (
                                np.sqrt(2) * D[:, indices : indices + 1][act_act]
                            )

                        else:
                            raise ValueError(
                                "Continuous decreed kernel not implemented"
                            )
                    else:
                        if mixint_type == MixIntKernelType.CONT_RELAX:
                            indices = indices + n_levels[indicat]
                        elif mixint_type == MixIntKernelType.GOWER:
                            indices = indices + 1
                else:
                    indices = indices + 1
    return D


class MixedIntegerCorrelation:
    """Encapsulates kernel correlation computation for mixed-integer design spaces.

    This class holds the cached parameter layout (``_corr_params``) and
    PLS coefficient state (``pls_coeff_cont``, ``coeff_pls_cat``) that was
    previously scattered across ``KrgBased`` instance attributes.

    Parameters
    ----------
    model : KrgBased
        The parent model. Used to access ``options``, ``corr``, ``ij``,
        ``training_points``, and the ``_compute_pls`` callback (if available).
    """

    def __init__(self, model):
        self._model = model
        self._corr_params = None
        self.pls_coeff_cont = []
        self.coeff_pls_cat = []
        self.n_levels_origin = None

    def reset(self):
        """Clear cached state. Call when retraining."""
        self._corr_params = None
        self.pls_coeff_cont = []
        self.coeff_pls_cat = []
        self.n_levels_origin = None

    # ------------------------------------------------------------------
    # Parameter layout
    # ------------------------------------------------------------------

    def _initialize_theta(self, theta, n_levels, cat_features, cat_kernel):
        """Compute the theta parameter layout for the mixed-integer kernel.

        Results are cached in ``self._corr_params`` so that repeated calls
        with the same structure return immediately.

        Returns
        -------
        tuple
            ``(cat_kernel_comps, ncomp, theta_cat_features,
            theta_cont_features, nx, n_levels)``
        """
        self.n_levels_origin = n_levels
        if self._corr_params is not None:
            return self._corr_params

        model = self._model
        nx = model.nx

        try:
            cat_kernel_comps = model.options["cat_kernel_comps"]
            if cat_kernel_comps is not None:
                n_levels = np.array(cat_kernel_comps)
        except KeyError:
            cat_kernel_comps = None
        try:
            ncomp = model.options["n_comp"]
            if not self.pls_coeff_cont:
                pass  # will be populated in compute()
        except KeyError:
            cat_kernel_comps = None
            ncomp = 1e5

        theta_cont_features = np.zeros((len(theta), 1), dtype=bool)
        theta_cat_features = np.ones((len(theta), len(n_levels)), dtype=bool)
        if cat_kernel in [
            MixIntKernelType.EXP_HOMO_HSPHERE,
            MixIntKernelType.HOMO_HSPHERE,
        ]:
            theta_cat_features = np.zeros((len(theta), len(n_levels)), dtype=bool)
        i = 0
        j = 0
        n_theta_cont = 0
        for feat in cat_features:
            if feat:
                if cat_kernel in [
                    MixIntKernelType.EXP_HOMO_HSPHERE,
                    MixIntKernelType.HOMO_HSPHERE,
                ]:
                    theta_cat_features[
                        j : j + int(n_levels[i] * (n_levels[i] - 1) / 2), i
                    ] = [True] * int(n_levels[i] * (n_levels[i] - 1) / 2)
                    j += int(n_levels[i] * (n_levels[i] - 1) / 2)
                i += 1
            else:
                if n_theta_cont < ncomp:
                    theta_cont_features[j] = True
                    theta_cat_features[j] = False
                    j += 1
                    n_theta_cont += 1

        theta_cat_features = (
            [
                np.where(theta_cat_features[:, i_lvl])[0]
                for i_lvl in range(len(n_levels))
            ],
            np.any(theta_cat_features, axis=1) if len(n_levels) > 0 else None,
        )

        self._corr_params = params = (
            cat_kernel_comps,
            ncomp,
            theta_cat_features,
            theta_cont_features,
            nx,
            n_levels,
        )
        return params

    # ------------------------------------------------------------------
    # Main correlation computation
    # ------------------------------------------------------------------

    def compute(
        self,
        corr,
        design_space,
        power,
        theta,
        theta_bounds,
        dx,
        Lij,
        n_levels,
        cat_features,
        cat_kernel,
        x=None,
        kplsk_second_loop=False,
    ):
        """Compute the mixed-integer kernel correlation.

        This method replaces ``KrgBased._matrix_data_corr``. It delegates
        continuous-kernel evaluation to ``model.corr`` and categorical
        correlation to the matrix / homo-spherical / compound-symmetry
        helpers in ``smt.utils.kriging``.

        Parameters
        ----------
        corr : str or Kernel
            Correlation function identifier.
        design_space : BaseDesignSpace
            Design space definition.
        power : float
            Power for the power-exponential kernel.
        theta : np.ndarray
            Hyperparameter vector.
        theta_bounds : array-like
            ``[lower, upper]`` bounds for theta.
        dx : np.ndarray
            Componentwise distances.
        Lij : np.ndarray
            Level-pair indices for categorical variables.
        n_levels : np.ndarray
            Number of levels per categorical variable.
        cat_features : np.ndarray
            Boolean mask of categorical feature indices.
        cat_kernel : MixIntKernelType
            Categorical kernel type.
        x : np.ndarray, optional
            Raw input points (needed for homo-spherical prediction).
        kplsk_second_loop : bool
            Whether this is the KPLSK second (full Kriging) loop.

        Returns
        -------
        np.ndarray
            Correlation vector/matrix.
        """
        model = self._model

        # Initialize static parameters
        (
            cat_kernel_comps,
            ncomp,
            theta_cat_features,
            theta_cont_features,
            nx,
            n_levels,
        ) = self._initialize_theta(theta, n_levels, cat_features, cat_kernel)

        # Sampling points X and y
        X, y = model._get_fidelity_training_data()

        if cat_kernel == MixIntKernelType.CONT_RELAX:
            X_pls_space, _, _ = design_space.unfold_x(X)
            nx = len(theta)
        elif cat_kernel == MixIntKernelType.GOWER:
            X_pls_space = np.copy(X)
        else:
            X_pls_space, _ = compute_X_cont(X, design_space)

        if not kplsk_second_loop and (cat_kernel_comps is not None or ncomp < 1e5):
            if np.size(self.pls_coeff_cont) == 0:
                X, y = model._compute_pls(X_pls_space.copy(), y.copy())
                self.pls_coeff_cont = model.coeff_pls
            if cat_kernel in [MixIntKernelType.GOWER, MixIntKernelType.CONT_RELAX]:
                d = componentwise_distance_PLS(
                    dx,
                    corr,
                    model.options["n_comp"],
                    self.pls_coeff_cont,
                    power,
                    theta=None,
                    return_derivative=False,
                )
                model.corr.theta = theta
                r = model.corr(d)
                return r
            else:
                d_cont = componentwise_distance_PLS(
                    dx[:, np.logical_not(cat_features)],
                    corr,
                    model.options["n_comp"],
                    self.pls_coeff_cont,
                    power,
                    theta=None,
                    return_derivative=False,
                )
        else:
            d = componentwise_distance(
                dx,
                corr,
                nx,
                power,
                theta=None,
                return_derivative=False,
            )
            if cat_kernel in [MixIntKernelType.GOWER, MixIntKernelType.CONT_RELAX]:
                model.corr.theta = theta
                r = model.corr(d)
                return r
            else:
                d_cont = d[:, np.logical_not(cat_features)]

        if model.options["corr"] == "squar_sin_exp":
            if model.options["categorical_kernel"] != MixIntKernelType.GOWER:
                theta_cont_features[-len([design_space.is_cat_mask]) :] = np.atleast_2d(
                    np.array([True] * len([design_space.is_cat_mask]))
                ).T
                theta_cat_features[1][-len([design_space.is_cat_mask]) :] = (
                    np.atleast_2d(np.array([False] * len([design_space.is_cat_mask]))).T
                )

        theta_cont = theta[theta_cont_features[:, 0]]
        model.corr.theta = theta_cont
        r_cont = model.corr(d_cont)
        r_cat = np.copy(r_cont) * 0
        r = np.copy(r_cont)

        # Theta_cat_i loop
        theta_cat_kernel = theta
        if len(n_levels) > 0:
            theta_cat_kernel = theta.copy()
            if cat_kernel == MixIntKernelType.EXP_HOMO_HSPHERE:
                theta_cat_kernel[theta_cat_features[1]] *= 0.5 * np.pi / theta_bounds[1]
            elif cat_kernel == MixIntKernelType.HOMO_HSPHERE:
                theta_cat_kernel[theta_cat_features[1]] *= 2.0 * np.pi / theta_bounds[1]
            elif cat_kernel == MixIntKernelType.COMPOUND_SYMMETRY:
                theta_cat_kernel[theta_cat_features[1]] *= 2.0
                theta_cat_kernel[theta_cat_features[1]] -= (
                    theta_bounds[1] + theta_bounds[0]
                )
                theta_cat_kernel[theta_cat_features[1]] *= 1 / (
                    1.000000000001 * theta_bounds[1]
                )

        for i in range(len(n_levels)):
            theta_cat = theta_cat_kernel[theta_cat_features[0][i]]
            if cat_kernel == MixIntKernelType.COMPOUND_SYMMETRY:
                T = np.zeros((n_levels[i], n_levels[i]))
                for tij in range(n_levels[i]):
                    for tji in range(n_levels[i]):
                        if tij == tji:
                            T[tij, tji] = 1
                        else:
                            T[tij, tji] = max(
                                theta_cat[0], 1e-10 - 1 / (n_levels[i] - 1)
                            )
            else:
                T = matrix_data_corr_levels_cat_matrix(
                    i,
                    n_levels,
                    theta_cat,
                    theta_bounds,
                    is_ehh=cat_kernel == MixIntKernelType.EXP_HOMO_HSPHERE,
                )

            if cat_kernel_comps is not None:
                # Sampling points
                X = model.training_points[None][0][0]
                y = model.training_points[None][0][1]
                X_icat = X[:, cat_features]
                X_icat = X_icat[:, i]
                old_n_comp = (
                    model.options["n_comp"] if "n_comp" in model.options else None
                )
                model.options["n_comp"] = int(n_levels[i] / 2 * (n_levels[i] - 1))
                X_full_space = compute_X_cross(X_icat, self.n_levels_origin[i])
                try:
                    model.coeff_pls = self.coeff_pls_cat[i]
                except IndexError:
                    _, _ = model._compute_pls(X_full_space.copy(), y.copy())
                    self.coeff_pls_cat.append(model.coeff_pls)

                if x is not None:
                    x_icat = x[:, cat_features]
                    x_icat = x_icat[:, i]
                    x_full_space = compute_X_cross(x_icat, self.n_levels_origin[i])
                    dx_cat_i = cross_levels_homo_space(
                        x_full_space, model.ij, y=X_full_space
                    )
                else:
                    dx_cat_i = cross_levels_homo_space(X_full_space, model.ij)

                d_cat_i = componentwise_distance_PLS(
                    dx_cat_i,
                    "squar_exp",
                    model.options["n_comp"],
                    model.coeff_pls,
                    power=model.options["pow_exp_power"],
                    theta=None,
                    return_derivative=False,
                )

                matrix_data_corr_levels_cat_mod_comps(
                    i,
                    Lij,
                    r_cat,
                    n_levels,
                    T,
                    d_cat_i,
                    has_cat_kernel=cat_kernel
                    in [
                        MixIntKernelType.EXP_HOMO_HSPHERE,
                        MixIntKernelType.HOMO_HSPHERE,
                    ],
                )
            else:
                matrix_data_corr_levels_cat_mod(
                    i,
                    Lij,
                    r_cat,
                    T,
                    has_cat_kernel=cat_kernel
                    in [
                        MixIntKernelType.EXP_HOMO_HSPHERE,
                        MixIntKernelType.HOMO_HSPHERE,
                        MixIntKernelType.COMPOUND_SYMMETRY,
                    ],
                )

            r = np.multiply(r, r_cat)
            if cat_kernel_comps is not None:
                if old_n_comp is None:
                    model.options._dict.pop("n_comp", None)
                else:
                    model.options["n_comp"] = old_n_comp
        return r
