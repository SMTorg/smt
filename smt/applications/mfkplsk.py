# -*- coding: utf-8 -*-
"""
Created on Fri May 04 10:26:49 2018

@author: Mostafa Meliani <melimostafa@gmail.com>
Multi-Fidelity co-Kriging: recursive formulation with autoregressive model of order 1 (AR1)
Partial Least Square decomposition added on highest fidelity level
KPLSK model combined PLS followed by a Krging model in the initial dimension
Adapted on March 2020 by Nathalie Bartoli to the new SMT version
Adapted on January 2021 by Andres Lopez-Lopera to the new SMT version
"""

from smt.utils.kriging_utils import componentwise_distance
from smt.applications import MFKPLS


class MFKPLSK(MFKPLS):
    def _initialize(self):
        super(MFKPLSK, self)._initialize()
        declare = self.options.declare
        # Like KPLSK, MFKPLSK used only with "squar_exp" correlations
        declare(
            "corr",
            "squar_exp",
            values=("squar_exp"),
            desc="Correlation function type",
            types=(str),
        )
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
        self._new_train_init()
        self.n_comp = self.options["n_comp"]
        theta0 = self.options["theta0"].copy()
        noise0 = self.options["noise0"].copy()

        for lvl in range(self.nlvl):
            self._new_train_iteration(lvl)
            self.options["n_comp"] = self.n_comp
            self.options["theta0"] = theta0
            self.options["noise0"] = noise0

        self._new_train_finalize(lvl)

    def _get_theta(self, i):
        return self.optimal_theta[i]
