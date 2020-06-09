# -*- coding: utf-8 -*-
"""
Created on Fri May 04 10:26:49 2018

@author: Mostafa Meliani <melimostafa@gmail.com>
Multi-Fidelity co-Kriging: recursive formulation with autoregressive model of order 1 (AR1)
Partial Least Square decomposition added on highest fidelity level
KPLSK model combined PLS followed by a Krging model in the initial dimension
Adapted March 2020 by Nathalie Bartoli to the new SMT version
"""

import numpy as np
from copy import deepcopy

from scipy.linalg import solve_triangular
from packaging import version
from sklearn import __version__ as sklversion

if version.parse(sklversion) < version.parse("0.22"):
    from sklearn.cross_decomposition.pls_ import PLSRegression as pls
else:
    from sklearn.cross_decomposition import PLSRegression as pls
from sklearn.metrics.pairwise import manhattan_distances

from smt.utils.kriging_utils import (
    l1_cross_distances,
    componentwise_distance,
    componentwise_distance_PLS,
    standardization,
)
from smt.applications import MFKPLS


class MFKPLSK(MFKPLS):
    def _initialize(self):
        super(MFKPLSK, self)._initialize()
        self.name = "MFKPLSK"

    def _componentwise_distance(self, dx, opt=0):
        # Modif for KPLSK model
        if opt == 0:
            # Kriging step
            d = componentwise_distance(dx, self.options["corr"], self.nx)
        else:
            # KPLS step
            d = super(MFKPLSK, self)._componentwise_distance(dx, opt)

        return d

    def _new_train(self):
        """
        Overrides KrgBased implementation
        Trains the Multi-Fidelity model + PLS (done on the highest fidelity level) + Kriging  (MFKPLSK)
        """
        # Modif for KPLSK model
        self.n_comp = self.options["n_comp"]
        self.theta0 = self.options["theta0"]

        self._new_train_init()

        for lvl in range(self.nlvl):
            self.options["n_comp"] = self.n_comp
            self.options["theta0"] = self.theta0
            self._new_train_iteration(lvl)

        self._new_train_finalize(lvl)

    def _get_theta(self, i):
        return self.optimal_theta[i]
