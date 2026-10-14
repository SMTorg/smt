"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from smt.kernels import Kernel
from smt.surrogate_models.krg_based import KrgBased
from smt.surrogate_models.krg_based.distances import (
    componentwise_distance,
    differences,
)


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
        Update the KRG model with n training points (X, Y) without full re-train:
        Theta hyperparameters and scaling factors on X and Y are kept fixed while the correlation matrix is updated
        using a block matrix approach instead of being recomputed from scratch.
        """
        self._update_smt_model(X_new=X, Y_new=Y)

    @staticmethod
    def _update_cholesky_block(
        L_old: np.ndarray, R12: np.ndarray, R22: np.ndarray, nugget: float = 1e-10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Cholesky block update.

        Returns the new Cholesky matrix along with the off-diagonal (L21)
        and new diagonal (L22) blocks needed for the inverse update.

        Parameters
        ----------
        L_old : np.ndarray
            Previous Cholesky lower triangular matrix.
        R12 : np.ndarray
            Cross-correlation matrix between old and new points.
        R22 : np.ndarray
            Correlation matrix of new points.
        nugget : float, optional
            Nugget to add to the diagonal of the new block for stability.

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

    @staticmethod
    def _update_inverse_cholesky_block(
        L_old_inv: np.ndarray, L21: np.ndarray, L22: np.ndarray
    ) -> np.ndarray:
        """
        Compute the inverse of the Cholesky Gram matrix L^{-1} block update.

        Parameters
        ----------
        L_old_inv : np.ndarray
            Inverse of the previous Cholesky lower triangular matrix.
        L21 : np.ndarray
            Off-diagonal block of the new Cholesky matrix.
        L22 : np.ndarray
            New diagonal block of the Cholesky matrix.

        Returns
        -------
        L_new_inv : np.ndarray
            Updated inverse of the Cholesky matrix.
        """
        n_old = L_old_inv.shape[0]
        n_new = L22.shape[0]

        is_single_point_added = n_new == 1
        if is_single_point_added:
            L22_inv = np.array([[1.0 / L22[0, 0]]])
        else:
            L22_inv = solve_triangular(
                L22, np.eye(n_new), lower=True, check_finite=False
            )

        M21 = -L22_inv @ (L21 @ L_old_inv)

        L_new_inv = np.empty((n_old + n_new, n_old + n_new))
        L_new_inv[:n_old, :n_old] = L_old_inv
        L_new_inv[:n_old, n_old:] = 0.0
        L_new_inv[n_old:, :n_old] = M21
        L_new_inv[n_old:, n_old:] = L22_inv

        return L_new_inv

    def _check_model_compatibility(self) -> None:
        """
        Check if the surrogate model is compatible with block updates.

        Parameters
        ----------
        self : KRG
            The surrogate model being updated.

        Raises
        ------
        TypeError
            If the model is not an instance of KRG or if the design space
            has mixed variables.
        """
        is_continuous_design_space = self.design_space.is_all_cont
        if not is_continuous_design_space:
            raise NotImplementedError(
                "Block update is only supported for continuous design variables."
            )

        is_hierarchical = self.design_space.is_conditionally_acting.any()
        if is_hierarchical:
            raise NotImplementedError(
                "Block update is only supported for non-hierarchical design spaces."
            )

    def _predict_covariance(
        self,
        X_new: np.ndarray,
        is_normalized: bool = False,
        compute_inverse: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute joint Cholesky factorization of training data and new candidates.

        Parameters
        ----------
        self : KRG
            The KRG surrogate model to update.
        X_new : np.ndarray
            New input points to add to the covariance structure.
        is_normalized : bool, optional
            If True, bypasses the internal SMT normalization step.
        compute_inverse : bool, optional
            If True, computes the inverse of the Cholesky Gram matrix using
            the block update approach.

        Returns
        -------
        L_new : np.ndarray
            Updated lower triangular Cholesky matrix.
        L_new_inv : np.ndarray or None
            Updated inverse of the Cholesky matrix, or None if compute_inverse
            is False.
        """
        self._check_model_compatibility()

        if is_normalized:
            X_new_norma = X_new
        else:
            X_new_norma = (X_new - self.X_offset) / self.X_scale

        dx12 = differences(self.X_norma, Y=X_new_norma)
        d12 = self._componentwise_distance(dx12)
        R12 = self.corr(d12).reshape(self.nt, X_new.shape[0])

        dx22 = differences(X_new_norma, Y=X_new_norma)
        d22 = self._componentwise_distance(dx22)
        R22 = self.corr(d22).reshape(X_new.shape[0], X_new.shape[0])

        L_old = self.optimal_par["C"]

        try:
            nugget = self.options["nugget"]
        except KeyError:
            nugget = 1e-8

        L_new, L21, L22 = self._update_cholesky_block(L_old, R12, R22, nugget=nugget)

        should_skip_inverse = not compute_inverse
        if should_skip_inverse:
            return L_new, None

        has_cached_inverse = "C_inv" in self.optimal_par
        if has_cached_inverse:
            L_old_inv = self.optimal_par["C_inv"]
        else:
            L_old_inv = solve_triangular(
                L_old, np.eye(L_old.shape[0]), lower=True, check_finite=False
            )

        L_new_inv = self._update_inverse_cholesky_block(L_old_inv, L21, L22)

        return L_new, L_new_inv

    def _update_smt_model(self, X_new: np.ndarray, Y_new: np.ndarray) -> "KRG":
        """
        Perform Cholesky block update and inject the state back into the SMT model.

        Utilizes L^{-1} for heavily accelerated coefficient updates.

        Parameters
        ----------
        self : KRG
            The KRG surrogate model to update.
        X_new : np.ndarray
            New input points to add to the model.
        Y_new : np.ndarray
            Target values corresponding to the new input points.

        Returns
        -------
        sm_model : KRG
            The updated KRG model, whose training values have been normalized (X)
            or standardized (Y) based on already existing scaling factors.
        """
        self._check_model_compatibility()

        X_new_norma = (X_new - self.X_offset) / self.X_scale
        Y_new_norma = (Y_new - self.y_mean) / self.y_std

        L_new, L_new_inv = self._predict_covariance(
            X_new_norma, is_normalized=True, compute_inverse=True
        )

        self.X_train = np.vstack((self.X_train, X_new))
        self.training_points[None][0][0] = self.X_train

        Y_train_appended = np.vstack(
            (self.training_points[None][0][1], Y_new.reshape(-1, 1))
        )
        self.training_points[None][0][1] = Y_train_appended

        self.X_norma = np.vstack((self.X_norma, X_new_norma))
        self.y_norma = np.vstack((self.y_norma, Y_new_norma.reshape(-1, 1)))
        self.nt = self.X_norma.shape[0]

        regression_type = self._regression_types[self.options["poly"]]
        F_combined = regression_type(self.X_norma)
        self.F = F_combined

        v_F = L_new_inv @ F_combined
        v_Y = L_new_inv @ self.y_norma

        R_F = v_F.T @ v_F
        beta_new = np.linalg.solve(R_F, v_F.T @ v_Y)

        residuals = self.y_norma - (F_combined @ beta_new)

        rho = L_new_inv @ residuals
        gamma_new = L_new_inv.T @ rho

        # GLS estimate of the process variance, as in the likelihood evaluation
        p, q = self._get_pq()
        sigma2 = (rho**2.0).sum(axis=0) / (self.nt - p - q)
        self.optimal_par["sigma2"] = sigma2 * self.y_std**2.0

        self.optimal_par["C"] = L_new
        self.optimal_par["C_inv"] = L_new_inv
        self.optimal_par["beta"] = beta_new
        self.optimal_par["gamma"] = gamma_new

        self.optimal_par["Ft"] = v_F
        Q, G = np.linalg.qr(v_F)
        self.optimal_par["Q"] = Q
        self.optimal_par["G"] = G
