"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

from __future__ import division
import warnings
import numpy as np
from sklearn.cross_decomposition.pls_ import PLSRegression as pls
from smt.surrogate_models.krg_based import KrgBased
from smt.utils.kriging_utils import componentwise_distance_PLS, componentwise_distance

"""
The KPLS class.
"""


class KPLSK(KrgBased):

    """
    - KPLSK
    """

    def _initialize(self):
        super(KPLSK, self)._initialize()
        declare = self.options.declare
        declare("n_comp", 1, types=int, desc="Number of principal components")
        self.name = "KPLSK"

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
