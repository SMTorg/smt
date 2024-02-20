"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

import numpy as np

from sklearn.cross_decomposition import PLSRegression as pls

from smt.surrogate_models.krg_based import KrgBased
from smt.utils.kriging import componentwise_distance_PLS


class KPLS(KrgBased):
    name = "KPLS"

    def _initialize(self):
        super(KPLS, self)._initialize()
        declare = self.options.declare
        declare("n_comp", 1, types=int, desc="Number of principal components")
        # KPLS used only with "abs_exp" and "squar_exp" correlations
        declare(
            "corr",
            "squar_exp",
            values=("abs_exp", "squar_exp", "pow_exp"),
            desc="Correlation function type",
            types=(str),
        )
        declare(
            "eval_n_comp",
            False,
            types=(bool),
            values=(True, False),
            desc="n_comp evaluation flag",
        )
        declare(
            "eval_comp_treshold",
            1.0,
            types=(float),
            desc="n_comp evaluation treshold for Wold's R criterion",
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

    def _componentwise_distance(self, dx, opt=0, theta=None, return_derivative=False):
        d = componentwise_distance_PLS(
            dx,
            self.options["corr"],
            self.options["n_comp"],
            self.coeff_pls,
            power=self.options["pow_exp_power"],
            theta=theta,
            return_derivative=return_derivative,
        )
        return d

    def _estimate_number_of_components(self):
        """
        self.options[n_comp] value from user is ignored and replaced by an estimated one wrt Wold's R criterion.
        """
        eval_comp_treshold = self.options["eval_comp_treshold"]
        X = self.training_points[None][0][0]
        y = self.training_points[None][0][1]
        k_fold = 4
        nbk = int(self.nt / k_fold)
        press_m = 0.0
        press_m1 = 0.0
        self.options["n_comp"] = 0
        nextcomp = True
        while nextcomp:
            self.options["n_comp"] += 1
            press_m = press_m1
            press_m1 = 0
            self.options["theta0"] = [0.1]
            for fold in range(k_fold):
                self.nt = len(X) - nbk
                todel = np.arange(fold * nbk, (fold + 1) * nbk)
                Xfold = np.copy(X)
                Xfold = np.delete(X, todel, axis=0)
                yfold = np.copy(y)
                yfold = np.delete(y, todel, axis=0)
                Xtest = np.copy(X)[fold * nbk : (fold + 1) * nbk, :]
                ytest = np.copy(y)[fold * nbk : (fold + 1) * nbk, :]

                self.training_points[None][0][0] = Xfold
                self.training_points[None][0][1] = yfold
                try:
                    self._new_train()
                except ValueError:
                    self.options["n_comp"] -= 1
                    nextcomp = False
                    break
                ye = self._predict_values(Xtest)
                press_m1 = press_m1 + np.sum(np.power((1 / len(X)) * (ye - ytest), 2))
            if self.options["n_comp"] > 1 and press_m1 / press_m > eval_comp_treshold:
                self.options["n_comp"] -= 1
                nextcomp = False
        self.training_points[None][0][0] = X
        self.training_points[None][0][1] = y
        self.nt = len(X)
        self.options["theta0"] = [0.1]

    def _train(self):
        """
        Train the model
        """
        # outputs['sol'] = self.sol

        if self.options["eval_n_comp"]:
            self._estimate_number_of_components()
        self._new_train()
