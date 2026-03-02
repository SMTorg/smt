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

import numpy as np

from smt.applications import MFKPLS
from smt.surrogate_models.krg_based import compute_n_param
from smt.surrogate_models.krg_based.distances import componentwise_distance


class MFKPLSK(MFKPLS):
    @property
    def _is_kplsk_style(self) -> bool:
        return True

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
        declare(
            "hyper_opt",
            "Cobyla",
            values=("Cobyla"),
            desc="Optimiser for hyperparameters optimisation",
            types=str,
        )
        self.name = "MFKPLSK"

    def _componentwise_distance(self, dx):
        if getattr(self, "_pls_pass", False):
            # PLS step (reduced space for first optimization pass)
            d = super(MFKPLSK, self)._componentwise_distance(dx)
        else:
            # Full Kriging step (prediction and second optimization pass)
            d = componentwise_distance(
                dx, self.options["corr"], self.nx, power=self.options["pow_exp_power"]
            )
        return d

    def _handle_theta0_out_of_bounds(self, theta0_i, i, theta_bounds):
        """Clamp theta0 to bounds like KPLSK."""
        if theta0_i - theta_bounds[1] > 0:
            return theta_bounds[1] - 1e-10
        else:
            return theta_bounds[0] + 1e-10

    def _run_optimization(self, D):
        """Two-pass optimization: PLS space then full Kriging space."""
        # First pass: optimize in reduced PLS space
        self._pls_pass = True
        self.kplsk_second_loop = False
        _, _, best_theta = self._optimize_hyperparam(D)

        # Project PLS theta back to full Kriging space
        if self.options["eval_noise"]:
            theta = best_theta[:-1]
        else:
            theta = best_theta

        if self.options["corr"] == "squar_exp":
            self.options["theta0"] = (theta * self.coeff_pls**2).sum(1)
        else:
            self.options["theta0"] = (theta * np.abs(self.coeff_pls)).sum(1)
        self.n_param = compute_n_param(
            self.design_space,
            self.options["categorical_kernel"],
            self.nx,
            None,
            None,
        )
        self.options["n_comp"] = int(self.n_param)
        self.best_iteration_fail = None

        # Second pass: optimize in full Kriging space (no multistart)
        self._pls_pass = False
        self.kplsk_second_loop = True
        return self._optimize_hyperparam(
            D,
            use_multistart=False,
            limit=10 * self.options["n_comp"],
        )

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

        self._reinterpolate(lvl)

    def _get_theta(self, i):
        return self.optimal_theta[i]
