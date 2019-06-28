"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

from __future__ import division
import warnings
import numpy as np
from smt.surrogate_models.krg_based import KrgBased
from smt.utils.kriging_utils import componentwise_distance_PLS, ge_compute_pls

"""
The KPLS class.
"""


class GEKPLS(KrgBased):

    """
    - GEKPLS
    """

    def _initialize(self):
        super(GEKPLS, self)._initialize()
        declare = self.options.declare
        declare(
            "xlimits",
            types=np.ndarray,
            desc="Lower/upper bounds in each dimension - ndarray [nx, 2]",
        )
        declare("n_comp", 1, types=int, desc="Number of principal components")
        declare(
            "theta0", [1e-2], types=(list, np.ndarray), desc="Initial hyperparameters"
        )
        declare("delta_x", 1e-4, types=(int, float), desc="Step used in the FOTA")
        declare(
            "extra_points",
            0,
            types=int,
            desc="Number of extra points per training point",
        )
        self.supports["training_derivatives"] = True

        self.name = "GEKPLS"

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

    def _componentwise_distance(self, dx, opt=0):

        d = componentwise_distance_PLS(
            dx, self.options["corr"], self.options["n_comp"], self.coeff_pls
        )
        return d
