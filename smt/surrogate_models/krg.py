"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

from smt.surrogate_models.krg_based import KrgBased
from smt.utils.kriging_utils import componentwise_distance


class KRG(KrgBased):
    name = "Kriging"

    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        declare(
            "corr",
            "squar_exp",
            values=("abs_exp", "squar_exp", "matern52", "matern32"),
            desc="Correlation function type",
            types=(str),
        )

    def _componentwise_distance(self, dx, opt=0, theta=None, return_derivative=False):
        d = componentwise_distance(
            dx,
            self.options["corr"],
            self.nx,
            theta=theta,
            return_derivative=return_derivative,
        )
        return d
