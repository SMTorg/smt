"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

import numpy as np
from smt.surrogate_models import KPLS
from smt.utils.kriging_utils import ge_compute_pls


class GEKPLS(KPLS):
    name = "GEKPLS"

    def _initialize(self):
        super(GEKPLS, self)._initialize()
        declare = self.options.declare
        # Like KPLS, GEKPLS used only with "abs_exp" and "squar_exp" correlations
        declare(
            "corr",
            "squar_exp",
            values=("abs_exp", "squar_exp"),
            desc="Correlation function type",
            types=(str),
        )
        declare(
            "xlimits",
            types=np.ndarray,
            desc="Lower/upper bounds in each dimension - ndarray [nx, 2]",
        )
        declare("delta_x", 1e-4, types=(int, float), desc="Step used in the FOTA")
        declare(
            "extra_points",
            0,
            types=int,
            desc="Number of extra points per training point",
        )
        self.supports["training_derivatives"] = True

    def _compute_pls(self, X, y):
        if 0 in self.training_points[None]:
            self.coeff_pls, XX, yy = ge_compute_pls(
                X.copy(),
                y.copy(),
                self.options["n_comp"],
                self.training_points,
                self.options["delta_x"],
                self.options["xlimits"],
                self.options["extra_points"],
            )
            if self.options["extra_points"] != 0:
                self.nt *= self.options["extra_points"] + 1
                X = np.vstack((X, XX))
                y = np.vstack((y, yy))

        return X, y
