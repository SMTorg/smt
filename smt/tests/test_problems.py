"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

import numpy as np
import unittest

from smt.problems import (
    CantileverBeam,
    Sphere,
    ReducedProblem,
    RobotArm,
    Rosenbrock,
    Branin,
    LpNorm,
)
from smt.problems import (
    TensorProduct,
    TorsionVibration,
    WaterFlow,
    WeldedBeam,
    WingWeight,
)
from smt.problems import (
    NdimCantileverBeam,
    NdimRobotArm,
    NdimRosenbrock,
    NdimStepFunction,
)
from smt.utils.sm_test_case import SMTestCase


class Test(SMTestCase):
    def run_test(self, problem):
        problem.options["return_complex"] = True

        # Test xlimits
        ndim = problem.options["ndim"]
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
        x = np.zeros((4, ndim), complex)
        x[0, :] = 0.2 * xlimits[:, 0] + 0.8 * xlimits[:, 1]
        x[1, :] = 0.4 * xlimits[:, 0] + 0.6 * xlimits[:, 1]
        x[2, :] = 0.6 * xlimits[:, 0] + 0.4 * xlimits[:, 1]
        x[3, :] = 0.8 * xlimits[:, 0] + 0.2 * xlimits[:, 1]
        y0 = problem(x)
        dydx_FD = np.zeros(4)
        dydx_CS = np.zeros(4)
        dydx_AN = np.zeros(4)

        print()
        h = 1e-5
        ch = 1e-16
        for iy in range(ny):
            for idim in range(ndim):
                x[:, idim] += h
                y_FD = problem(x)
                x[:, idim] -= h

                x[:, idim] += complex(0, ch)
                y_CS = problem(x)
                x[:, idim] -= complex(0, ch)

                dydx_FD[:] = (y_FD[:, iy] - y0[:, iy]) / h
                dydx_CS[:] = np.imag(y_CS[:, iy]) / ch
                dydx_AN[:] = problem(x, idim)[:, iy]

                abs_rms_error_FD = np.linalg.norm(dydx_FD - dydx_AN)
                rel_rms_error_FD = np.linalg.norm(dydx_FD - dydx_AN) / np.linalg.norm(
                    dydx_FD
                )

                abs_rms_error_CS = np.linalg.norm(dydx_CS - dydx_AN)
                rel_rms_error_CS = np.linalg.norm(dydx_CS - dydx_AN) / np.linalg.norm(
                    dydx_CS
                )

                msg = (
                    "{:16s} iy {:2} dim {:2} of {:2} "
                    + "abs_FD {:16.9e} rel_FD {:16.9e} abs_CS {:16.9e} rel_CS {:16.9e}"
                )
                print(
                    msg.format(
                        problem.options["name"],
                        iy,
                        idim,
                        ndim,
                        abs_rms_error_FD,
                        rel_rms_error_FD,
                        abs_rms_error_CS,
                        rel_rms_error_CS,
                    )
                )
                self.assertTrue(rel_rms_error_FD < 1e-3 or abs_rms_error_FD < 1e-5)

    def test_sphere(self):
        self.run_test(Sphere(ndim=1))
        self.run_test(Sphere(ndim=3))

    def test_exp(self):
        self.run_test(TensorProduct(name="TP-exp", ndim=1, func="exp"))
        self.run_test(TensorProduct(name="TP-exp", ndim=3, func="exp"))

    def test_tanh(self):
        self.run_test(TensorProduct(name="TP-tanh", ndim=1, func="tanh"))
        self.run_test(TensorProduct(name="TP-tanh", ndim=3, func="tanh"))

    def test_cos(self):
        self.run_test(TensorProduct(name="TP-cos", ndim=1, func="cos"))
        self.run_test(TensorProduct(name="TP-cos", ndim=3, func="cos"))

    def test_gaussian(self):
        self.run_test(TensorProduct(name="TP-gaussian", ndim=1, func="gaussian"))
        self.run_test(TensorProduct(name="TP-gaussian", ndim=3, func="gaussian"))

    def test_branin(self):
        self.run_test(Branin(ndim=2))

    def test_lp_norm(self):
        self.run_test(LpNorm(ndim=2))

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
        self.run_test(ReducedProblem(TorsionVibration(ndim=15), dims=[5, 10, 12, 13]))

    def test_water_flow(self):
        self.run_test(WaterFlow(ndim=8))
        self.run_test(ReducedProblem(WaterFlow(ndim=8), dims=[0, 1, 6]))

    def test_welded_beam(self):
        self.run_test(WeldedBeam(ndim=3))

    def test_wing_weight(self):
        self.run_test(WingWeight(ndim=10))
        self.run_test(ReducedProblem(WingWeight(ndim=10), dims=[0, 2, 3, 5]))

    def test_ndim_cantilever_beam(self):
        self.run_test(NdimCantileverBeam(ndim=1))
        self.run_test(NdimCantileverBeam(ndim=2))

    def test_ndim_robot_arm(self):
        self.run_test(NdimRobotArm(ndim=1))
        self.run_test(NdimRobotArm(ndim=2))

    def test_ndim_rosenbrock(self):
        self.run_test(NdimRosenbrock(ndim=1))
        self.run_test(NdimRosenbrock(ndim=2))

    def test_ndim_step_function(self):
        self.run_test(NdimStepFunction(ndim=1))
        self.run_test(NdimStepFunction(ndim=2))


if __name__ == "__main__":
    unittest.main()
