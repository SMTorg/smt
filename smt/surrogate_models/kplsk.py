"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

from smt.surrogate_models import KPLS
from smt.utils.kriging_utils import componentwise_distance_PLS, componentwise_distance


class KPLSK(KPLS):
    name = "KPLSK"

    def _initialize(self):
        super(KPLSK, self)._initialize()
        declare = self.options.declare
        # KPLSK used only with "squar_exp" correlations
        declare(
            "corr",
            "squar_exp",
            values=("squar_exp"),
            desc="Correlation function type",
            types=(str),
        )
        self.name = "KPLSK"

    def _componentwise_distance(self, dx, opt=0, theta=None, return_derivative=False):
        if opt == 0:
            # Kriging step
            d = componentwise_distance(
                dx,
                self.options["corr"],
                self.nx,
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
                theta=theta,
                return_derivative=return_derivative,
            )
        return d
