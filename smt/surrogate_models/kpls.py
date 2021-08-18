"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

import numpy as np

from packaging import version
from sklearn import __version__ as sklversion

if version.parse(sklversion) < version.parse("0.22"):
    from sklearn.cross_decomposition.pls_ import PLSRegression as pls
else:
    from sklearn.cross_decomposition import PLSRegression as pls

from smt.surrogate_models.krg_based import KrgBased
from smt.utils.kriging_utils import componentwise_distance_PLS


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
            values=("abs_exp", "squar_exp"),
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
        self.name = "KPLS"

    def _compute_pls(self, X, y):
        _pls = pls(self.options["n_comp"])
        # As of sklearn 0.24.1 zeroed outputs raises an exception while sklearn 0.23 returns zeroed x_rotations
        # For now the try/except below is a workaround to restore the 0.23 behaviour
        try:
            self.coeff_pls = _pls.fit(X.copy(), y.copy()).x_rotations_
        except StopIteration:
            self.coeff_pls = np.zeros((X.shape[1], self.options["n_comp"]))
        return X, y

    def _componentwise_distance(self, dx, opt=0, theta=None, return_derivative=False):
        d = componentwise_distance_PLS(
            dx,
            self.options["corr"],
            self.options["n_comp"],
            self.coeff_pls,
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
                # We are doing hyperparameters hot_start for the k-fold
                # as the problem is the same and the data are mainly similar
                if fold == 0:
                    super(KPLS, self)._initialize()
                else:
                    super(KPLS, self)._initialize()
                    self.options["theta0"] = theta_opt

                try:
                    self._new_train()
                except ValueError:
                    self.options["n_comp"] -= 1
                    nextcomp = False
                    break

                ye = self._predict_values(Xtest)
                press_m1 = press_m1 + np.sum(np.power((1 / len(X)) * (ye - ytest), 2))

                theta_opt = self.options["theta0"]
                theta_opt = self.optimal_theta

            if self.options["n_comp"] > 1 and press_m1 / press_m > eval_comp_treshold:
                self.options["n_comp"] -= 1
                nextcomp = False

        self.training_points[None][0][0] = X
        self.training_points[None][0][1] = y
        self.nt = len(X)
        super(KPLS, self)._initialize()

    def _train(self):
        """
        Train the model
        """
        # outputs['sol'] = self.sol

        if self.options["eval_n_comp"]:
            self._estimate_number_of_components()
        self._new_train()
