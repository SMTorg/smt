import unittest
import numpy as np
from smt.sampling_methods import LHS
from smt.problems import Sphere
from smt.surrogate_models import GPX


class TestGPX(unittest.TestCase):
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
