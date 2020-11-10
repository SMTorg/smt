"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""
import warnings
import numpy as np

from packaging import version
from sklearn import __version__ as sklversion

if version.parse(sklversion) < version.parse("0.22"):
    from sklearn.cross_decomposition.pls_ import PLSRegression as pls
else:
    from sklearn.cross_decomposition import PLSRegression as pls

from smt.surrogate_models.krg_based import KrgBased
from smt.utils.kriging_utils import componentwise_distance_PLS, componentwise_distance


class KPLSK(KrgBased):
    def _initialize(self):
        super(KPLSK, self)._initialize()
        declare = self.options.declare
        declare("n_comp", 1, types=int, desc="Number of principal components")
        self.name = "KPLSK"
        # KPLSK used only with "squar_exp" correlations
        declare(
            "corr",
            "squar_exp",
            values=("squar_exp"),
            desc="Correlation function type",
            types=(str),
        )

    def _compute_pls(self, X, y):
        _pls = pls(self.options["n_comp"])
        self.coeff_pls = _pls.fit(X.copy(), y.copy()).x_rotations_

        return X, y

    def _componentwise_distance(self, dx, opt=0):
        if opt == 0:
            # Kriging step
            d = componentwise_distance(dx, self.options["corr"], self.nx)
        else:
            # KPLS step
            d = componentwise_distance_PLS(
                dx, self.options["corr"], self.options["n_comp"], self.coeff_pls
            )
        return d
