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

    def _componentwise_distance(self, dx, opt=0, theta=None, return_derivative=False):
        if opt == 0:
            # Kriging step
            d = componentwise_distance(
                dx,
                self.options["corr"],
                self.nx,
                power=self.options["pow_exp_power"],
                theta=theta,
                return_derivative=return_derivative,
            )
        else:
            # KPLS step
            d = componentwise_distance_PLS(
                dx,
                self.options["corr"],
                self.options["n_comp"],
                self.coeff_pls,
                power=self.options["pow_exp_power"],
                theta=theta,
                return_derivative=return_derivative,
            )
        return d

    # --- Polymorphic hook overrides ---

    @property
    def _n_outer_iterations(self):
        """KPLSK uses two-pass optimization: PLS space then full Kriging space."""
        return 1

    def _handle_theta0_out_of_bounds(self, theta0_i, i, theta_bounds):
        """KPLSK clamps theta0 to bounds instead of random initialization."""
        if theta0_i - theta_bounds[1] > 0:
            return theta_bounds[1] - 1e-10
        else:
            return theta_bounds[0] + 1e-10

    def _should_sample_multistart(self, ii):
        """Only sample LHS multistart points in the first (PLS) loop."""
        return ii == 1

    def _finalize_outer_loop(
        self,
        best_optimal_rlf_value,
        best_optimal_par,
        best_optimal_theta,
        exit_function,
    ):
        """KPLSK two-pass: after PLS loop, update theta0 from PLS coefficients
        and continue to full Kriging loop. After Kriging loop, return."""
        if self.options["eval_noise"]:
            # best_optimal_theta contains [theta, noise] if eval_noise = True
            theta = best_optimal_theta[:-1]
        else:
            # best_optimal_theta contains [theta] if eval_noise = False
            theta = best_optimal_theta

        if exit_function:
            return True, exit_function, None

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
        new_limit = 10 * self.options["n_comp"]
        self.best_iteration_fail = None
        return False, True, new_limit
