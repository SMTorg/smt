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
