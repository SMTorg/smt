import unittest

import numpy as np

from smt.problems import Sphere
from smt.sampling_methods import LHS
from smt.surrogate_models import GPX
from smt.surrogate_models.gpx import GPX_AVAILABLE


class TestGPX(unittest.TestCase):
    @unittest.skipIf(not GPX_AVAILABLE, "GPX not available")
    def test_gpx(self):
        ndim = 2
        num = 50
        problem = Sphere(ndim=ndim)
        xlimits = problem.xlimits
        sampling = LHS(xlimits=xlimits, criterion="ese")

        xt = sampling(num)
        yt = problem(xt)

        sm = GPX()
        sm.set_training_values(xt, yt)
        sm.train()

        xe = sampling(10)
        ye = problem(xe)

        ytest = sm.predict_values(xe)
        e_error = np.linalg.norm(ytest - ye) / np.linalg.norm(ye)
        self.assertLessEqual(e_error, 2e-2)

        vars = sm.predict_variances(xt)
        self.assertLessEqual(np.linalg.norm(vars), 1e-6)


if __name__ == "__main__":
    unittest.main()
