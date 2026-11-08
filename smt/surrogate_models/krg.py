"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

from typing import Tuple

import numpy as np
from scipy.linalg import cholesky, qr, solve_triangular

from smt.kernels import Kernel
from smt.surrogate_models.krg_based import KrgBased

from smt.surrogate_models.krg_based.distances import cross_distances
from smt.surrogate_models.krg_based.kernel_types import MixIntKernelType
from smt.surrogate_models.krg_based.mixed_int_corr import (
    compute_X_cont,
    cross_levels,
    gower_componentwise_distances,
)

from smt.surrogate_models.krg_based.distances import (
    componentwise_distance,
    differences,
)
from smt.utils.checks import check_nx, ensure_2d_array


class KRG(KrgBased):
    name = "Kriging"

    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        declare(
            "corr",
            "squar_exp",
            values=(
                "pow_exp",
                "abs_exp",
                "squar_exp",
                "squar_sin_exp",
                "matern52",
                "matern32",
            ),
            desc="Correlation function type",
            types=(str, Kernel),
        )

    def _componentwise_distance(self, dx, theta=None, return_derivative=False):
        d = componentwise_distance(
            dx,
            self.options["corr"],
            self.nx,
            self._pow_exp_power,
            theta=theta,
            return_derivative=return_derivative,
        )
        return d

    def update_model(self, X, Y):
        """
        Add training points to a trained model without re-training it.

        The hyperparameters (theta, noise) and the scaling factors of X and Y
        are kept fixed. The Cholesky factor of the correlation matrix is
        extended by a block update instead of being recomputed from scratch,
        then the regression coefficients (beta), the GP weights (gamma) and
        the process variance (sigma2) are re-estimated on the whole training
        set.

        Contrary to ``train()``, the scaling factors are not recomputed, so
        results differ slightly from a re-training with ``hyper_opt="NoOp"``.

        Supported for continuous and mixed-integer hierarchical design spaces.
        However, heteroscedastic noise, reinterpolation (is_ri), and non-distance
        based categorical kernels (e.g., PLS-based hypersphere) are not supported.

        Parameters
        ----------
        X : np.ndarray[n_new, nx] or np.ndarray[n_new]
            New input points.
        Y : np.ndarray[n_new, ny] or np.ndarray[n_new]
            Output values at the new input points.

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        NotImplementedError
            If the model or its options are not supported by the block update.
        """
        self._update_smt_model(X_new=X, Y_new=Y)

    @staticmethod
    def _update_cholesky_block(
        L_old: np.ndarray, R12: np.ndarray, R22: np.ndarray, nugget: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Cholesky block update.

        Returns the new Cholesky matrix along with the off-diagonal (L21)
        and new diagonal (L22) blocks needed to extend the triangular solves.

        Parameters
        ----------
        L_old : np.ndarray
            Previous Cholesky lower triangular matrix.
        R12 : np.ndarray
            Cross-correlation matrix between old and new points.
        R22 : np.ndarray
            Correlation matrix of new points.
        nugget : float
            Value added to the diagonal of the new block (nugget plus
            homoscedastic noise variance).

        Returns
        -------
        L_new : np.ndarray
            Updated lower triangular Cholesky matrix.
        L21 : np.ndarray
            Off-diagonal block of the new Cholesky matrix.
        L22 : np.ndarray
            New diagonal block of the Cholesky matrix.
        """
        n_old = L_old.shape[0]
        n_new = R22.shape[0]

        L21_T = solve_triangular(L_old, R12, lower=True, check_finite=False)
        L21 = L21_T.T

        is_single_point_added = n_new == 1

        if is_single_point_added:
            R22_eff = R22[0, 0] - np.sum(L21_T**2) + nugget
            is_positive_definite = R22_eff > 0.0

            if not is_positive_definite:
                raise ValueError("Updated correlation matrix is not positive definite.")
            L22 = np.array([[np.sqrt(R22_eff)]])

        else:
            R22_eff = R22 - (L21 @ L21_T)
            np.fill_diagonal(R22_eff, R22_eff.diagonal() + nugget)
            L22 = cholesky(R22_eff, lower=True, check_finite=False)

        L_new = np.empty((n_old + n_new, n_old + n_new))
        L_new[:n_old, :n_old] = L_old
        L_new[:n_old, n_old:] = 0.0
        L_new[n_old:, :n_old] = L21
        L_new[n_old:, n_old:] = L22

        return L_new, L21, L22

    def _check_model_compatibility(self) -> None:
        """
        Check if the surrogate model is compatible with block updates.

        Raises
        ------
        NotImplementedError
            If the model is a subclass of KRG, if heteroscedastic noise or
            reinterpolation is used, or if the categorical kernel is not distance-based.
        """
        # Subclasses (e.g. SGP) store a different model state in optimal_par
        if type(self) is not KRG:
            raise NotImplementedError(
                f"Block update is not supported for {type(self).__name__} models."
            )


        if self.options["use_het_noise"]:
            raise NotImplementedError(
                "Block update is not supported with heteroscedastic noise."
            )

        if self.options["is_ri"]:
            raise NotImplementedError(
                "Block update is not supported with reinterpolation (is_ri)."
            )

        try:
            cat_kernel = self.options["categorical_kernel"]
            if cat_kernel in [
                MixIntKernelType.EXP_HOMO_HSPHERE,
                MixIntKernelType.HOMO_HSPHERE,
            ]:
                raise NotImplementedError(
                    "Block update is not supported for non distance-based categorical kernels."
                )
        except KeyError:
            pass


    def _correct_categorical_acting_distances(self, D: np.ndarray, is_acting1: np.ndarray, is_acting2: np.ndarray, ij: np.ndarray, mixint_type) -> np.ndarray:
        listcatdecreed = self.design_space.is_conditionally_acting[self.cat_features]
        if np.any(listcatdecreed):
            D = self._correct_distances_cat_decreed(
                D, is_acting1, listcatdecreed, ij,
                is_acting_y=is_acting2, mixint_type=mixint_type
            )
        return D

    def _apply_continuous_relaxation(self, D: np.ndarray, D_num: np.ndarray, X1: np.ndarray, is_acting1: np.ndarray, X2: np.ndarray, is_acting2: np.ndarray, ij: np.ndarray) -> np.ndarray:
        X1_unfolded, _, unfolded_cat = self.design_space.unfold_x(X1)
        X2_unfolded, _, _ = self.design_space.unfold_x(X2)
        X1_norma = (X1_unfolded - self.X2_offset) / self.X2_scale
        X2_norma = (X2_unfolded - self.X2_offset) / self.X2_scale
        D = np.abs(differences(X1_norma, Y=X2_norma.copy()))

        if (
            "n_comp" not in self.options._dict.keys()
            and "cat_kernel_comp" not in self.options._dict.keys()
        ):
            D = self._correct_categorical_acting_distances(D, is_acting1, is_acting2, ij, MixIntKernelType.GOWER)
        if np.any(self.design_space.is_conditionally_acting):
            D[:, np.logical_not(unfolded_cat)] = (
                D_num / self.X2_scale[np.logical_not(unfolded_cat)]
            )
        return D

    def _apply_distributional_encoding(self, D: np.ndarray, D_num: np.ndarray, X1: np.ndarray, is_acting1: np.ndarray, X2: np.ndarray, is_acting2: np.ndarray, ij: np.ndarray) -> np.ndarray:
        self._mix_int_corr.replace_with_de_distances(
            D, X2, ij, self.options["categorical_kernel_beta"], x_pred=X1
        )
        return D

    def _compute_mixed_distances_between(
        self, X1, is_acting1, X2, is_acting2
    ):
        _, ij = cross_distances(X1, y=X2)
        D, D_num = gower_componentwise_distances(
            X1, x_is_acting=is_acting1, design_space=self.design_space,
            hierarchical_kernel=self.options["hierarchical_kernel"],
            y=np.copy(X2), y_is_acting=is_acting2,
        )

        D = self._correct_categorical_acting_distances(D, is_acting1, is_acting2, ij, MixIntKernelType.GOWER)

        kernel_handlers = {
            MixIntKernelType.CONT_RELAX: self._apply_continuous_relaxation,
            MixIntKernelType.DIST_ENCODING: self._apply_distributional_encoding,
        }

        handler = kernel_handlers.get(self.options["categorical_kernel"])
        if handler:
            D = handler(D, D_num, X1, is_acting1, X2, is_acting2, ij)

        Lij, _ = cross_levels(
            X=X1, ij=ij, design_space=self.design_space, y=X2
        )
        return D, Lij

    def _compute_R12_R22_mixed(self, X_new: np.ndarray, is_acting_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_new = X_new.shape[0]

        _, _, _, _, Lij12, dx12 = self._predict_init(X_new, is_acting_new)
        R12 = self._matrix_data_corr(
            corr=self.options["corr"],
            design_space=self.design_space,
            power=self._pow_exp_power,
            theta=self.optimal_theta,
            theta_bounds=self.options["theta_bounds"],
            dx=dx12,
            Lij=Lij12,
            n_levels=self.n_levels,
            cat_features=self.cat_features,
            cat_kernel=self.options["categorical_kernel"],
            x=X_new,
        ).reshape(n_new, self.nt).T

        dx22, Lij22 = self._compute_mixed_distances_between(
            X_new, is_acting_new, X_new, is_acting_new
        )
        R22 = self._matrix_data_corr(
            corr=self.options["corr"],
            design_space=self.design_space,
            power=self._pow_exp_power,
            theta=self.optimal_theta,
            theta_bounds=self.options["theta_bounds"],
            dx=dx22,
            Lij=Lij22,
            n_levels=self.n_levels,
            cat_features=self.cat_features,
            cat_kernel=self.options["categorical_kernel"],
            x=X_new,
        ).reshape(n_new, n_new)
        return R12, R22

    def _compute_R12_R22_cont(self, X_new: np.ndarray, X_new_norma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_new = X_new.shape[0]

        _, _, _, _, _, dx12 = self._predict_init(X_new, None)
        d12 = self._componentwise_distance(dx12)
        R12 = self.corr(d12).reshape(n_new, self.nt).T

        dx22 = differences(X_new_norma, Y=X_new_norma)
        d22 = self._componentwise_distance(dx22)
        R22 = self.corr(d22).reshape(n_new, n_new)
        return R12, R22

    def _extend_cholesky(
        self, X_new: np.ndarray, X_new_norma: np.ndarray, is_acting_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extend the Cholesky factor of the training correlation matrix with
        new points.

        Parameters
        ----------
        X_new : np.ndarray
            New input points (original).
        X_new_norma : np.ndarray
            New input points, normalized with the model scaling factors.
        is_acting_new : np.ndarray
            Is acting array for the new points.

        Returns
        -------
        L_new : np.ndarray
            Updated lower triangular Cholesky matrix.
        L21 : np.ndarray
            Off-diagonal block of the new Cholesky matrix.
        L22 : np.ndarray
            New diagonal block of the Cholesky matrix.
        """
        if not self.is_continuous:
            R12, R22 = self._compute_R12_R22_mixed(X_new, is_acting_new)
        else:
            R12, R22 = self._compute_R12_R22_cont(X_new, X_new_norma)

        L_old = self.optimal_par["C"]

        # Same diagonal term as in the likelihood evaluation: 1 + nugget + noise
        nugget = 0.0 if self._eval_noise else self.options["nugget"]
        noise = float(np.atleast_1d(self.optimal_noise)[0])

        return self._update_cholesky_block(L_old, R12, R22, nugget=nugget + noise)

    def _get_is_acting(self, X_new: np.ndarray) -> np.ndarray:
        if self.is_acting_points.get(None) is not None:
            _, is_acting_new = self.design_space.correct_get_acting(X_new)
            return is_acting_new
        return None

    def _sync_is_acting(self, is_acting_new: np.ndarray) -> None:
        is_acting = self.is_acting_points.get(None)
        if is_acting is not None:
            self.is_acting_points[None] = np.vstack((is_acting, is_acting_new))
            self.is_acting_train = self.is_acting_points[None]

    def _get_normalized_inputs(self, X_new: np.ndarray) -> np.ndarray:
        if not self.is_continuous:
            X_cont_new, _ = compute_X_cont(X_new, self.design_space)
        else:
            X_cont_new = np.copy(X_new)
        return (X_cont_new - self.X_offset) / self.X_scale

    def _get_normalized_outputs(self, Y_new: np.ndarray) -> np.ndarray:
        return (Y_new - self.y_mean) / self.y_std

    def _recompute_categorical_metadata(self, X_new: np.ndarray) -> None:
        if not self.is_continuous:
            _, self.ij = cross_distances(self.X_train)
            self.Lij, self.n_levels = cross_levels(
                X=self.X_train, ij=self.ij, design_space=self.design_space
            )
            if hasattr(self, "X2_norma"):
                X_new_unfolded, _, _ = self.design_space.unfold_x(X_new)
                X_new_unfolded_norma = (X_new_unfolded - self.X2_offset) / self.X2_scale
                self.X2_norma = np.vstack((self.X2_norma, X_new_unfolded_norma))

    def _update_predictive_parameters(self, Ft: np.ndarray, Yt: np.ndarray, L_new: np.ndarray) -> None:
        Q, G = qr(Ft, mode="economic")
        beta = solve_triangular(G, Q.T @ Yt, check_finite=False)
        rho = Yt - Ft @ beta
        p, q = self._get_pq()
        sigma2 = (rho**2.0).sum(axis=0) / (self.nt - p - q)

        self.optimal_par["sigma2"] = sigma2 * self.y_std**2.0
        self.optimal_par["beta"] = beta
        self.optimal_par["gamma"] = solve_triangular(L_new.T, rho, check_finite=False)
        self.optimal_par["C"] = L_new
        self.optimal_par["Ft"] = Ft
        self.optimal_par["G"] = G
        self.optimal_par["Q"] = Q

    def _check_update_dimensions(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        if not hasattr(self, "optimal_par"):
            raise RuntimeError("update_model requires a trained model.")

        self._check_model_compatibility()

        check_nx(self, X_new)
        if Y_new.shape[0] != X_new.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of rows "
                f"(got {X_new.shape[0]} and {Y_new.shape[0]})."
            )
        if Y_new.shape[1] != self.ny:
            raise ValueError(
                f"Y has {Y_new.shape[1]} outputs, but the model was trained "
                f"with {self.ny}."
            )

    def _update_smt_model(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        """
        Perform the Cholesky block update and store the new state in the model
        (training points, normalized data and optimal_par), in place.

        Only the rows of the new points are solved against the block factor, so
        no full factorization or inverse of the correlation matrix is needed.
        New points are normalized with the existing scaling factors.

        Parameters
        ----------
        X_new : np.ndarray
            New input points to add to the model.
        Y_new : np.ndarray
            Output values at the new input points.
        """
        X_new = ensure_2d_array(X_new, "X")
        Y_new = ensure_2d_array(Y_new, "Y")

        self._check_update_dimensions(X_new, Y_new)

        is_acting_new = self._get_is_acting(X_new)
        X_new_norma = self._get_normalized_inputs(X_new)
        Y_new_norma = self._get_normalized_outputs(Y_new)

        L_old = self.optimal_par["C"]
        L_new, L21, L22 = self._extend_cholesky(X_new, X_new_norma, is_acting_new)

        # Whitened (L^-1 applied) regression matrix and outputs of the old points.
        # Ft_old comes from training; Yt_old costs one O(nt^2) triangular solve.
        Ft_old = self.optimal_par["Ft"]
        Yt_old = solve_triangular(L_old, self.y_norma, lower=True, check_finite=False)

        self.X_train = np.vstack((self.X_train, X_new))
        self.training_points[None][0][0] = self.X_train

        Y_train_appended = np.vstack((self.training_points[None][0][1], Y_new))
        self.training_points[None][0][1] = Y_train_appended

        self._sync_is_acting(is_acting_new)
        self._recompute_categorical_metadata(X_new)

        self.X_norma = np.vstack((self.X_norma, X_new_norma))
        self.y_norma = np.vstack((self.y_norma, Y_new_norma))
        self.nt = self.X_norma.shape[0]

        regression_type = self._regression_types[self.options["poly"]]
        F_new = regression_type(X_new_norma)
        self.F = np.vstack((self.F, F_new))

        # Forward substitution with the block factor [[L_old, 0], [L21, L22]]:
        # the old rows are unchanged, only the new ones need to be solved.
        Ft_new = solve_triangular(
            L22, F_new - L21 @ Ft_old, lower=True, check_finite=False
        )
        Yt_new = solve_triangular(
            L22, Y_new_norma - L21 @ Yt_old, lower=True, check_finite=False
        )
        Ft = np.vstack((Ft_old, Ft_new))
        Yt = np.vstack((Yt_old, Yt_new))

        self._update_predictive_parameters(Ft, Yt, L_new)
