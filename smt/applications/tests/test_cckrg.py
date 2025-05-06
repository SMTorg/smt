"""
Author: Lisa Pretsch <<lisa.pretsch@tum.de>>

This package is distributed under New BSD license.
"""

import unittest

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
        import random

        import numpy as np

        from smt.applications import CoopCompKRG
        from smt.problems import TensorProduct
        from smt.sampling_methods import LHS

        # The problem is the exponential problem with dimension 10
        ndim = 10
        prob = TensorProduct(ndim=ndim, func="exp")

        # Example with three random components
        # (use physical components if available)
        ncomp = 3

        # Initial sampling
        samp = LHS(xlimits=prob.xlimits, random_state=42)
        np.random.seed(0)
        xt = samp(50)
        yt = prob(xt)
        np.random.seed(1)

        # Random design variable to component allocation
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

        # Cooperative components Kriging model fit
        sm = CoopCompKRG()
        for active_coop_comp in comps:
            sm.set_training_values(xt, yt)
            sm.train(active_coop_comp, comp_var)

        # Prediction as for ordinary Kriging
        xpoint = (-5 + np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])) / 10.0
        print(sm.predict_values(xpoint))
        print(sm.predict_variances(xpoint))

    # run scripts are used in documentation as documentation is not always rebuild
    # make a test run by pytest to test the run scripts
    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_run_cckrg_example(self):
        self.run_cckrg_example()


if __name__ == "__main__":
    unittest.main()
