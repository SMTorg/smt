"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression as pls

from smt.surrogate_models.krg_based import KrgBased, MixIntKernelType
from smt.surrogate_models.krg_based.distances import componentwise_distance_PLS


class KPLS(KrgBased):
    name = "KPLS"

    @property
    def _use_pls(self) -> bool:
        return True

    def _initialize(self):
        super(KPLS, self)._initialize()
        declare = self.options.declare

        declare(
            "n_comp",
            1,
            types=int,
            desc=(
                "Number of principal components. Only used when eval_n_comp=False. "
                "Ignored when eval_n_comp=True (the optimal value is estimated instead)."
            ),
        )
        # KPLS used only with "abs_exp", "squar_exp" and "pow_exp" correlations
        declare(
            "corr",
            "squar_exp",
            values=("abs_exp", "squar_exp", "pow_exp"),
            desc="Correlation function type",
            types=str,
        )
        declare(
            "eval_n_comp",
            False,
            types=bool,
            values=(True, False),
            desc=(
                "If True, the optimal number of components is estimated automatically "
                "using the strategy defined by eval_n_comp_strategy. "
                "If False, the value of n_comp is used directly."
            ),
        )
        declare(
            "eval_comp_treshold",
            1.0,
            types=float,
            desc="Threshold for Wold's R criterion (used when eval_n_comp_strategy='wold').",
        )
        declare(
            "eval_n_comp_strategy",
            "wold",
            values=("wold", "exhaustive"),
            types=str,
            desc=(
                "Strategy used to estimate the optimal number of components when "
                "eval_n_comp=True. "
                "'wold': increments n_comp until the cross-validation error stops "
                "decreasing (early stopping via Wold's R criterion). "
                "'exhaustive': evaluates all values in range_n_comp_exhaustive and "
                "picks the global minimum."
            ),
        )
        declare(
            "range_n_comp_exhaustive",
            None,
            types=(list, tuple, np.ndarray, type(None)),
            desc=(
                "Search range [n_min, n_max] (inclusive) for the exhaustive strategy. "
                "Only used when eval_n_comp=True and eval_n_comp_strategy='exhaustive'. "
                "If None, defaults to [1, max(1, n_features // 10)]."
            ),
        )
        declare(
            "cat_kernel_comps",
            None,
            types=list,
            desc="Number of components for PLS categorical kernel",
        )

    def _compute_pls(self, X, y):
        _pls = pls(self.options["n_comp"])
        self.coeff_pls = 0
        if np.shape(X)[0] < self.options["n_comp"] + 1:
            raise ValueError(
                "ValueError: The database should be at least "
                + str(self.options["n_comp"] + 1)
                + " points (currently "
                + str(np.shape(X)[0])
                + ")."
            )
        else:
            if np.shape(X)[1] == 1:
                self.coeff_pls = np.atleast_2d(np.array([1]))
            else:
                self.coeff_pls = abs(_pls.fit(X.copy(), y.copy()).x_rotations_)
        return X, y

    def _check_param(self):
        """
        Validates KPLS-specific parameters before calling base class validation.
        """
        if (
            self.options["categorical_kernel"]
            not in [
                MixIntKernelType.EXP_HOMO_HSPHERE,
                MixIntKernelType.HOMO_HSPHERE,
            ]
            and self._use_pls
        ):
            if self.options["cat_kernel_comps"] is not None:
                raise ValueError("cat_kernel_comps option is for homoscedastic kernel.")

        if not self.options["eval_n_comp"]:
            if (
                not isinstance(self.options["n_comp"], int)
                or self.options["n_comp"] < 1
            ):
                raise ValueError(
                    "n_comp must be a positive integer when eval_n_comp=False."
                )

        if (
            self.options["eval_n_comp"]
            and self.options["eval_n_comp_strategy"] == "exhaustive"
            and self.options["range_n_comp_exhaustive"] is not None
        ):
            r = self.options["range_n_comp_exhaustive"]
            if len(r) != 2 or int(r[0]) >= int(r[1]):
                raise ValueError(
                    "range_n_comp_exhaustive must be a two-element array-like "
                    "[n_min, n_max] with n_min < n_max."
                )

        super()._check_param()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_exhaustive_range(self):
        """
        Returns (n_min, n_max) for the exhaustive search.
        Falls back to [1, max(1, n_features // 10)] when the option is None.
        """
        r = self.options["range_n_comp_exhaustive"]
        if r is None:
            n_features = self.training_points[None][0][0].shape[1]
            n_max = max(1, n_features // 10)
            return 1, n_max
        return int(r[0]), int(r[1])

    def _calculate_press_error(self, X, y, n_comp, k_fold, original_nt):
        """
        Computes the k-fold cross-validation PRESS error for a given n_comp.

        Parameters
        ----------
        X : ndarray, shape (original_nt, n_features) — full training inputs, never mutated
        y : ndarray, shape (original_nt, n_outputs)  — full training outputs, never mutated
        n_comp : int — number of PLS components to use
        k_fold : int — number of folds
        original_nt : int — size of the full dataset; used to compute fold width
                      and the PRESS normalisation factor. Passed explicitly so
                      this method does not depend on self.nt, which may have
                      been overwritten by a previous fold.

        Returns np.inf if training fails for any fold.
        """
        nbk = int(original_nt / k_fold)
        press = 0.0
        for fold in range(k_fold):
            todel = np.arange(fold * nbk, (fold + 1) * nbk)
            Xfold = np.delete(X, todel, axis=0)
            yfold = np.delete(y, todel, axis=0)
            Xtest = X[fold * nbk : (fold + 1) * nbk, :]
            ytest = y[fold * nbk : (fold + 1) * nbk, :]

            # Reset theta0 before each fold so optimisation starts from the
            # same point regardless of what _new_train set in the previous fold
            self._theta0 = [0.1]
            self.nt = len(Xfold)
            self.training_points[None][0][0] = Xfold
            self.training_points[None][0][1] = yfold
            try:
                self._new_train()
            except ValueError:
                return np.inf
            ye = self._predict_values(Xtest)
            press += np.sum(np.power((1 / original_nt) * (ye - ytest), 2))
        return press

    def _estimate_number_of_components(self):
        """
        Unified component-count estimator.

        - strategy='wold'      : increments n_comp one at a time and stops as
                                 soon as the PRESS error increases relative to
                                 the previous step (Wold's R criterion).
        - strategy='exhaustive': evaluates every integer in
                                 range_n_comp_exhaustive (or the auto-derived
                                 range) and returns the global minimum.
        """
        strategy = self.options["eval_n_comp_strategy"]
        eval_comp_treshold = self.options["eval_comp_treshold"]

        X = self.training_points[None][0][0].copy()
        y = self.training_points[None][0][1].copy()
        original_nt = len(X)

        k_fold = min(4, original_nt)
        best_comp = 1
        best_error = np.inf

        # Need at least 2 folds to have both a train and a validation split
        if k_fold < 2:
            self.options["n_comp"] = best_comp
            return

        if strategy == "wold":
            n_comp = 0
            prev_error = np.inf
            while True:
                n_comp += 1
                self.options["n_comp"] = n_comp
                error = self._calculate_press_error(X, y, n_comp, k_fold, original_nt)

                if error == np.inf:
                    # Training failed: revert to last good value
                    n_comp -= 1
                    break

                if n_comp > 1 and error / prev_error > eval_comp_treshold:
                    # Error increased beyond threshold: keep previous n_comp
                    n_comp -= 1
                    break

                best_comp = n_comp
                prev_error = error

        elif strategy == "exhaustive":
            n_min, n_max = self._get_exhaustive_range()
            for n_comp in range(n_min, n_max + 1):
                self.options["n_comp"] = n_comp
                error = self._calculate_press_error(X, y, n_comp, k_fold, original_nt)
                if error < best_error:
                    best_error = error
                    best_comp = n_comp

        # Restore training data and state
        self.training_points[None][0][0] = X
        self.training_points[None][0][1] = y
        self.nt = original_nt
        self._theta0 = [0.1]
        self.options["n_comp"] = best_comp

    # ------------------------------------------------------------------

    def _componentwise_distance(self, dx, theta=None, return_derivative=False):
        d = componentwise_distance_PLS(
            dx,
            self.options["corr"],
            self.options["n_comp"],
            self.coeff_pls,
            power=self._pow_exp_power,
            theta=theta,
            return_derivative=return_derivative,
        )
        return d

    def _train(self):
        """
        Train the model.

        If eval_n_comp=True the optimal number of components is estimated first
        using the configured strategy, then the model is trained with that value.
        If eval_n_comp=False the user-supplied n_comp is used directly.
        """
        if self.options["eval_n_comp"]:
            self._estimate_number_of_components()

        self._new_train()
