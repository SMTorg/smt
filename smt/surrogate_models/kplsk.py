"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

import numpy as np

from smt.surrogate_models import KPLS
from smt.surrogate_models.krg_based import compute_n_param
from smt.surrogate_models.krg_based.distances import (
    componentwise_distance,
    componentwise_distance_PLS,
)


class KPLSK(KPLS):
    name = "KPLSK"

    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        # KPLSK used only with "squar_exp" correlations
        declare(
            "corr",
            "squar_exp",
            values=("squar_exp"),
            desc="Correlation function type",
            types=(str),
        )
        # KPLSK does not evaluate n_comp during optimization, so it is set to False
        declare(
            "eval_n_comp",
            False,
            types=(bool),
            values=(False),
            desc="n_comp evaluation flag",
        )

    def _componentwise_distance(self, dx, theta=None, return_derivative=False):
        if getattr(self, "_pls_pass", False):
            # PLS step (reduced space for first optimization pass)
            d = componentwise_distance_PLS(
                dx,
                self.options["corr"],
                self.options["n_comp"],
                self.coeff_pls,
                power=self.options["pow_exp_power"],
                theta=theta,
                return_derivative=return_derivative,
            )
        else:
            # Full Kriging step (prediction and second optimization pass)
            d = componentwise_distance(
                dx,
                self.options["corr"],
                self.nx,
                power=self.options["pow_exp_power"],
                theta=theta,
                return_derivative=return_derivative,
            )
        return d

    def _handle_theta0_out_of_bounds(self, theta0_i, i, theta_bounds):
        """KPLSK clamps theta0 to bounds instead of random initialization."""
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
