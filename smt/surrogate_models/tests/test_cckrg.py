"""
Author: Lisa Pretsch <<lisa.pretsch@tum.de>>

This package is distributed under New BSD license.
"""

import unittest
import random

import numpy as np

from smt.surrogate_models import CoopCompKRG
from smt.problems import TensorProduct
from smt.sampling_methods import LHS

try:
    import matplotlib

    matplotlib.use("Agg")
    NO_MATPLOTLIB = False
except ImportError:
    NO_MATPLOTLIB = True

from smt.utils.sm_test_case import SMTestCase


class TestCCKRG(SMTestCase):
    @staticmethod
    def run_cckrg_example():
        import numpy as np

        from smt.surrogate_models import CoopCompKRG
        from smt.problems import TensorProduct
        from smt.sampling_methods import LHS

        # The problem is the exponential problem with dimension 10
        ndim = 10
        prob = TensorProduct(ndim=ndim, func="exp")

        # Example with three random components
        ncomp = 3

        # Initial sampling
        samp = LHS(xlimits=prob.xlimits, seed=42)
        xt = samp(50)
        yt = prob(xt)

        # Cooperative components Kriging model fit
        # comp_var is auto-computed from ncomp and seed
        sm = CoopCompKRG(ncomp=ncomp)
        sm.set_training_values(xt, yt)
        sm.train()

        # Prediction as for ordinary Kriging
        xpoint = (-5 + np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])) / 10.0
        print(sm.predict_values(xpoint))
        print(sm.predict_variances(xpoint))

    def test_cckrg_explicit_comp_var(self):
        # The problem is the exponential problem with dimension 10
        ndim = 10
        prob = TensorProduct(ndim=ndim, func="exp")

        ncomp = 3

        # Initial sampling
        samp = LHS(xlimits=prob.xlimits, seed=42)
        xt = samp(50)
        yt = prob(xt)

        # Explicit design variable to component allocation
        comps = [*range(ncomp)]
        vars = [*range(ndim)]
        random.shuffle(vars)
        comp_var = np.full((ndim, ncomp), False)
        for c in comps:
            comp_size = int(ndim / ncomp)
            start = c * comp_size
            end = (c + 1) * comp_size
            if c + 1 == ncomp:
                end = max((c + 1) * comp_size, ndim)
            comp_var[vars[start:end], c] = True

        # Cooperative components Kriging model fit with explicit comp_var
        sm = CoopCompKRG(comp_var=comp_var)
        sm.set_training_values(xt, yt)
        sm.train()

        # Prediction as for ordinary Kriging
        xpoint = (-5 + np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])) / 10.0
        print(sm.predict_values(xpoint))
        print(sm.predict_variances(xpoint))

    # run scripts are used in documentation as documentation is not always rebuild
    # make a test run by pytest to test the run scripts
    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_run_cckrg_example(self):
        self.run_cckrg_example()

    def test_ego_with_cckrg(self):
        from smt.applications import EGO
        from smt.design_space import DesignSpace
        from smt.problems import Sphere

        ndim = 10
        prob = Sphere(ndim=ndim)
        design_space = DesignSpace(prob.xlimits)

        surrogate = CoopCompKRG(design_space=design_space, ncomp=2, print_global=False)

        ego = EGO(
            n_iter=10,
            criterion="EI",
            n_doe=50,
            surrogate=surrogate,
            seed=42,
            verbose=True,
        )

        x_opt, y_opt, _, x_data, y_data = ego.optimize(fun=prob)
        # Check that EGO improved over the initial DOE
        y_doe_best = np.min(y_data[:50])
        print(f"Best DOE value: {y_doe_best:.4e}, best EGO value: {y_opt.item():.4e}")
        self.assertLess(y_opt.item(), y_doe_best)


if __name__ == "__main__":
    unittest.main()
