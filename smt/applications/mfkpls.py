# -*- coding: utf-8 -*-
"""
Created on Fri May 04 10:26:49 2018

@author: Mostafa Meliani <melimostafa@gmail.com>
Multi-Fidelity co-Kriging: recursive formulation with autoregressive model of order 1 (AR1)
Partial Least Square decomposition added on highest fidelity level
Adapted March 2020 by Nathalie Bartoli to the new SMT version
"""

from copy import deepcopy
from sys import exit
import numpy as np
from scipy.linalg import solve_triangular
from scipy import linalg
from packaging import version
from sklearn import __version__ as sklversion

if version.parse(sklversion) < version.parse("0.22"):
    from sklearn.cross_decomposition.pls_ import PLSRegression as pls
else:
    from sklearn.cross_decomposition import PLSRegression as pls
from sklearn.metrics.pairwise import manhattan_distances

from smt.utils.kriging_utils import (
    cross_distances,
    componentwise_distance,
    standardization,
)
from smt.applications import MFK
from smt.utils.kriging_utils import componentwise_distance_PLS


class MFKPLS(MFK):
    def _initialize(self):
        super(MFKPLS, self)._initialize()
        declare = self.options.declare
        declare("n_comp", 1, types=int, desc="Number of principal components")
        self.name = "MFKPLS"

    def _componentwise_distance(self, dx, opt=0):
        d = componentwise_distance_PLS(
            dx, self.options["corr"], self.options["n_comp"], self.coeff_pls
        )
        return d

    def _compute_pls(self, X, y):
        _pls = pls(self.options["n_comp"])
        self.coeff_pls = _pls.fit(X.copy(), y.copy()).x_rotations_

        return X, y
