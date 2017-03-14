from __future__ import print_function, division
import numpy as np
import unittest

from six.moves import range

from smt.problems import CantileverBeam, Carre, ReducedProblem, RobotArm, Rosenbrock
from smt.problems import TensorProduct, TorsionVibration, WaterFlow, WeldedBeam, WingWeight
from smt.utils.sm_test_case import SMTestCase


class Test(SMTestCase):

    def run_test(self, problem):
        # Test xlimits
        ndim = problem.options['ndim']
        xlimits = problem.xlimits
        self.assertEqual(xlimits.shape, (ndim, 2))

        # Test evaluation of multiple points at once
        x = np.zeros((10, ndim))
        for ind in range(10):
            x[ind, :] = 0.5 * (xlimits[:, 0] + xlimits[:, 1])
        y = problem(x)
        ny = y.shape[1]
        self.assertEqual(x.shape[0], y.shape[0])

        # Test derivatives
        x = np.zeros((4, ndim))
        x[0, :] = 0.2 * xlimits[:, 0] + 0.8 * xlimits[:, 1]
        x[1, :] = 0.4 * xlimits[:, 0] + 0.6 * xlimits[:, 1]
        x[2, :] = 0.6 * xlimits[:, 0] + 0.4 * xlimits[:, 1]
        x[3, :] = 0.8 * xlimits[:, 0] + 0.2 * xlimits[:, 1]
        y0 = problem(x)
        dydx_FD = np.zeros(4)
        dydx_AN = np.zeros(4)

        print()
        h = 1e-5
        for iy in range(ny):
            for idim in range(ndim):
                x[:, idim] += h
                y = problem(x)
                x[:, idim] -= h
                dydx_FD[:] = (y[:, iy] - y0[:, iy]) / h
                dydx_AN[:] = problem(x, idim)[:, iy]
                abs_rms_error = np.linalg.norm(dydx_FD - dydx_AN)
                rel_rms_error = np.linalg.norm(dydx_FD - dydx_AN) / np.linalg.norm(dydx_FD)
                msg = '{:16s} iy {:2} dim {:2} of {:2} abs {:16.9e} rel {:16.9e}'
                print(msg.format(problem.options['name'], iy, idim, ndim, abs_rms_error, rel_rms_error))
                self.assertTrue(rel_rms_error < 1e-3 or abs_rms_error < 1e-5)

    def test_carre(self):
        self.run_test(Carre(ndim=1))
        self.run_test(Carre(ndim=3))

    def test_exp(self):
        self.run_test(TensorProduct(name='TP-exp', ndim=1, func='exp'))
        self.run_test(TensorProduct(name='TP-exp', ndim=3, func='exp'))

    def test_tanh(self):
        self.run_test(TensorProduct(name='TP-tanh', ndim=1, func='tanh'))
        self.run_test(TensorProduct(name='TP-tanh', ndim=3, func='tanh'))

    def test_cos(self):
        self.run_test(TensorProduct(name='TP-cos', ndim=1, func='cos'))
        self.run_test(TensorProduct(name='TP-cos', ndim=3, func='cos'))

    def test_gaussian(self):
        self.run_test(TensorProduct(name='TP-gaussian', ndim=1, func='gaussian'))
        self.run_test(TensorProduct(name='TP-gaussian', ndim=3, func='gaussian'))

    def test_rosenbrock(self):
        self.run_test(Rosenbrock(ndim=2))
        self.run_test(Rosenbrock(ndim=3))

    def test_cantilever_beam(self):
        self.run_test(CantileverBeam(ndim=3))
        self.run_test(CantileverBeam(ndim=6))
        self.run_test(CantileverBeam(ndim=9))
        self.run_test(CantileverBeam(ndim=12))

    def test_robot_arm(self):
        self.run_test(RobotArm(ndim=2))
        self.run_test(RobotArm(ndim=4))
        self.run_test(RobotArm(ndim=6))

    def test_torsion_vibration(self):
        self.run_test(TorsionVibration(ndim=15))
        self.run_test(ReducedProblem(TorsionVibration(ndim=15), 3))

    def test_water_flow(self):
        self.run_test(WaterFlow(ndim=8))
        self.run_test(ReducedProblem(WaterFlow(ndim=8), 3))

    def test_welded_beam(self):
        self.run_test(WeldedBeam(ndim=3))

    def test_wing_weight(self):
        self.run_test(WingWeight(ndim=10))
        self.run_test(ReducedProblem(WingWeight(ndim=10), 3))


if __name__ == '__main__':
    unittest.main()
