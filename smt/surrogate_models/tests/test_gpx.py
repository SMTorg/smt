import tempfile
import unittest

import numpy as np

from smt.problems import Sphere
from smt.sampling_methods import LHS
from smt.surrogate_models import GPX, KRG
from smt.surrogate_models.gpx import GPX_AVAILABLE


class TestGPX(unittest.TestCase):
    @unittest.skipIf(not GPX_AVAILABLE, "GPX not available")
    def test_gpx(self):
        ndim = 2
        num = 20
        problem = Sphere(ndim=ndim)
        xlimits = problem.xlimits
        sampling = LHS(xlimits=xlimits, criterion="ese")

        xt = sampling(num)
        yt = problem(xt)

        gpx = GPX(print_global=False, seed=42)
        gpx.set_training_values(xt, yt)
        gpx.train()

        xe = sampling(10)
        ye = problem(xe)

        # Prediction should be pretty good
        gpx_y = gpx.predict_values(xe)
        e_error = np.linalg.norm(gpx_y - ye) / np.linalg.norm(ye)
        self.assertLessEqual(e_error, 1e-3)

        gpx_var = gpx.predict_variances(xe)
        self.assertLessEqual(np.linalg.norm(gpx_var), 1e-3)

    @unittest.skipIf(not GPX_AVAILABLE, "GPX not available")
    def test_save_load(self):
        ndim = 2
        num = 20
        problem = Sphere(ndim=ndim)
        xlimits = problem.xlimits
        sampling = LHS(xlimits=xlimits, criterion="ese")

        xt = sampling(num)
        yt = problem(xt)

        gpx = GPX(print_global=False, seed=42)
        gpx.set_training_values(xt, yt)
        gpx.train()

        with tempfile.NamedTemporaryFile(suffix=".json") as fp:
            gpx.save(fp.name)
            gpx2 = GPX.load(fp.name)

        xe = sampling(10)
        ye = problem(xe)

        gpx_y = gpx2.predict_values(xe)
        e_error = np.linalg.norm(gpx_y - ye) / np.linalg.norm(ye)
        self.assertLessEqual(e_error, 1e-3)

    @unittest.skipIf(not GPX_AVAILABLE, "GPX not available")
    def test_gpx_vs_krg(self):
        ndim = 3
        num = 30
        problem = Sphere(ndim=ndim)
        xlimits = problem.xlimits
        sampling = LHS(xlimits=xlimits, criterion="ese", seed=42)

        xt = sampling(num)
        yt = problem(xt)

        gpx = GPX(print_global=False, seed=42)
        gpx.set_training_values(xt, yt)
        gpx.train()

        xe = sampling(10)

        gpx_y = gpx.predict_values(xe)
        gpx_var = gpx.predict_variances(xe)

        # Compare against KRG
        krg = KRG(print_global=False)
        krg.set_training_values(xt, yt)
        krg.train()

        krg_y = krg.predict_values(xe)
        np.testing.assert_allclose(gpx_y, krg_y, rtol=1e-4, atol=1e-2)

        krg_var = krg.predict_variances(xe)
        np.testing.assert_allclose(gpx_var, krg_var, rtol=1e-4, atol=1e-2)

        for kx in range(ndim):
            dy = gpx.predict_derivatives(xe, kx)
            krg_dy = krg.predict_derivatives(xe, kx)
            np.testing.assert_allclose(dy, krg_dy, rtol=1e-2, atol=1e-3)

            dvar = gpx.predict_variance_derivatives(xe, kx)
            krg_dvar = krg.predict_variance_derivatives(xe, kx)
            np.testing.assert_allclose(dvar, krg_dvar, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
