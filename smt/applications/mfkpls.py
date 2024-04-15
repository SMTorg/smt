# -*- coding: utf-8 -*-
"""
Created on Fri May 04 10:26:49 2018

@author: Mostafa Meliani <melimostafa@gmail.com>
Multi-Fidelity co-Kriging: recursive formulation with autoregressive model of order 1 (AR1)
Partial Least Square decomposition added on highest fidelity level
Adapted on March 2020 by Nathalie Bartoli to the new SMT version
Adapted on January 2021 by Andres Lopez-Lopera to the new SMT version
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression as pls
from sklearn.metrics.pairwise import check_pairwise_arrays

from smt.applications import MFK
from smt.utils.kriging import componentwise_distance_PLS


class MFKPLS(MFK):
    """
    Multi-Fidelity model + PLS (done on the highest fidelity level)
    """

    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        # Like KPLS, MFKPLS used only with "abs_exp" and "squar_exp" correlations
        declare(
            "corr",
            "squar_exp",
            values=("abs_exp", "squar_exp"),
            desc="Correlation function type",
            types=(str),
        )
        declare(
            "hyper_opt",
            "Cobyla",
            values=("Cobyla"),
            desc="Optimiser for hyperparameters optimisation",
            types=str,
        )
        declare("n_comp", 1, types=int, desc="Number of principal components")
        self.name = "MFKPLS"

    def _differences(self, X, Y):
        """
        Overrides differences function for MFK
        Compute the manhattan_distances
        """
        X, Y = check_pairwise_arrays(X, Y)
        D = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
        D = np.abs(D, D)
        return D.reshape((-1, X.shape[1]))

    def _componentwise_distance(self, dx, opt=0):
        d = componentwise_distance_PLS(
            dx,
            self.options["corr"],
            self.options["n_comp"],
            self.coeff_pls,
            power=self.options["pow_exp_power"],
        )
        return d

    def _compute_pls(self, X, y):
        _pls = pls(self.options["n_comp"])
        # As of sklearn 0.24.1 PLS with zeroed outputs raises an exception while sklearn 0.23 returns zeroed x_rotations
        # For now the try/except below is a workaround to restore the 0.23 behaviour
        try:
            self.coeff_pls = _pls.fit(X.copy(), y.copy()).x_rotations_
        except StopIteration:
            self.coeff_pls = np.zeros((X.shape[1], self.options["n_comp"]))

        return X, y

    def _get_theta(self, i):
        return np.sum(self.optimal_theta[i] * self.coeff_pls**2, axis=1)
