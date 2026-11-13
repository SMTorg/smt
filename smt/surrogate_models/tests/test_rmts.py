"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest

import numpy as np

from smt.surrogate_models import RMTB, RMTC, RMTS
from smt.utils.misc import compute_relative_error
from smt.utils.silence import Silence
from smt.utils.sm_test_case import SMTestCase


def function_test_1d(x):
    # function xsinx
    x = np.reshape(x, (-1,))
    y = np.zeros(x.shape)
    y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
    return y.reshape((-1, 1))


class TestRMTS(SMTestCase):
    def setUp(self):
        # xt = np.random.rand(5, 1) * 10.0
        self.xt = np.array(
            [[3.6566495, 4.64266046, 7.23645433, 6.04862594, 8.85571712]]
        ).T
        self.yt = function_test_1d(self.xt)

        self.xlimits = np.array([[0.0, 25.0]])
        smref = RMTS(xlimits=self.xlimits, spline_type="B-Spline", print_global=False)
        smref.set_training_values(self.xt, self.yt)
        with Silence():
            smref.train()

        self.xref = np.array([[0.0, 6.25, 12.5, 18.75, 25.0]]).T
        self.yref = smref.predict_values(self.xref)

        self.sms = {}

    def test_linear_search(self):
        for ls in ["bracketed", "cubic", "quadratic", "null"]:
            self.sms[ls] = RMTS(
                xlimits=self.xlimits,
                spline_type="B-Spline",
                line_search=ls,
                print_global=False,
            )
            self.sms[ls].set_training_values(self.xt, self.yt)

            with Silence():
                self.sms[ls].train()

            error = compute_relative_error(self.sms[ls], self.xref, self.yref)
            self.assert_error(error, 0.0, 1e-1)

    def test_linear_solver(self):
        from smt.utils.linear_solvers import KrylovSolver

        for ls in [
            "krylov-dense",
            "dense-lu",
            "dense-chol",
            "lu",
            "ilu",
            "krylov",
            "krylov-lu",
            "krylov-mg",
            "gs",
            "jacobi",
            "mg",
            "null",
        ]:
            self.sms[ls] = RMTS(
                xlimits=self.xlimits,
                spline_type="B-Spline",
                solver=ls,
                print_global=False,
            )
            self.sms[ls].set_training_values(self.xt, self.yt)

            # Test different Krylov variants for coverage
            if ls == "krylov":
                for solver_type in ["bicgstab", "gmres"]:
                    # Directly pass a solver instance to test internal Krylov algorithms
                    custom_solver = KrylovSolver(solver=solver_type)
                    self.sms[ls + "_" + solver_type] = RMTS(
                        xlimits=self.xlimits,
                        spline_type="B-Spline",
                        solver=custom_solver,
                        print_global=False,
                    )
                    self.sms[ls + "_" + solver_type].set_training_values(
                        self.xt, self.yt
                    )
                    with Silence():
                        self.sms[ls + "_" + solver_type].train()

            with Silence():
                self.sms[ls].train()

            error = compute_relative_error(self.sms[ls], self.xref, self.yref)
            self.assert_error(error, 0.0, 1.1)

    def test_rmts_spline_type_equivalence_bspline(self):
        xq = np.linspace(0.0, 25.0, 20).reshape(-1, 1)

        # 1. RMTB baseline
        sm_b = RMTB(xlimits=self.xlimits, order=4, num_ctrl_pts=12, print_global=False)
        sm_b.set_training_values(self.xt, self.yt)
        with Silence():
            sm_b.train()

        yq_ref = sm_b.predict_values(xq)
        dyq_ref = sm_b.predict_derivatives(xq, 0)
        bs_ref = sm_b.get_bspline()

        # 2. Test RMTS with default and multiple aliases
        aliases = [
            None,  # default should be B-Spline
            "B-Spline",
            "b-spline",
            "bspline",
            "b_spline",
            "b",
        ]
        for alias in aliases:
            kwargs = {
                "xlimits": self.xlimits,
                "order": 4,
                "num_ctrl_pts": 12,
                "print_global": False,
            }
            if alias is not None:
                kwargs["spline_type"] = alias
            sm_s = RMTS(**kwargs)
            sm_s.set_training_values(self.xt, self.yt)
            with Silence():
                sm_s.train()

            # Exact numerical equivalence
            np.testing.assert_allclose(
                sm_s.sol_coeff,
                sm_b.sol_coeff,
                atol=1e-15,
                err_msg=f"Coeff mismatch for spline_type={alias!r}",
            )
            np.testing.assert_allclose(
                sm_s.predict_values(xq),
                yq_ref,
                atol=1e-15,
                err_msg=f"Prediction mismatch for spline_type={alias!r}",
            )
            np.testing.assert_allclose(
                sm_s.predict_derivatives(xq, 0),
                dyq_ref,
                atol=1e-15,
                err_msg=f"Derivative mismatch for spline_type={alias!r}",
            )

            # Representation equivalence
            bs = sm_s.get_bspline()
            self.assertEqual(bs["degree"], bs_ref["degree"])
            np.testing.assert_allclose(
                bs["coefficients"], bs_ref["coefficients"], atol=1e-15
            )
            for k1, k2 in zip(bs["knots"], bs_ref["knots"]):
                np.testing.assert_allclose(k1, k2, atol=1e-15)

            # SciPy export equivalence
            sp_nd = sm_s.to_scipy(physical=True)
            sp_1d = sm_s.to_scipy(physical=True, as_1d=True)
            np.testing.assert_allclose(sp_nd(xq), yq_ref[:, 0], atol=1e-12)
            np.testing.assert_allclose(sp_1d(xq[:, 0]), yq_ref[:, 0], atol=1e-12)

    def test_rmts_spline_type_equivalence_cubic_hermite(self):
        xq = np.linspace(0.0, 25.0, 20).reshape(-1, 1)

        # 1. RMTC baseline
        sm_c = RMTC(xlimits=self.xlimits, num_elements=6, print_global=False)
        sm_c.set_training_values(self.xt, self.yt)
        with Silence():
            sm_c.train()

        yq_ref = sm_c.predict_values(xq)
        dyq_ref = sm_c.predict_derivatives(xq, 0)

        # 2. Test RMTS with multiple aliases
        aliases = [
            "Cubic_Hermite_Spline",
            "cubic_hermite_spline",
            "Cubic_Hermite",
            "cubic_hermite",
            "hermite",
            "c",
        ]
        for alias in aliases:
            sm_s = RMTS(
                xlimits=self.xlimits,
                spline_type=alias,
                num_elements=6,
                print_global=False,
            )
            sm_s.set_training_values(self.xt, self.yt)
            with Silence():
                sm_s.train()

            np.testing.assert_allclose(
                sm_s.sol_coeff,
                sm_c.sol_coeff,
                atol=1e-15,
                err_msg=f"Coeff mismatch for spline_type={alias!r}",
            )
            np.testing.assert_allclose(
                sm_s.predict_values(xq),
                yq_ref,
                atol=1e-15,
                err_msg=f"Prediction mismatch for spline_type={alias!r}",
            )
            np.testing.assert_allclose(
                sm_s.predict_derivatives(xq, 0),
                dyq_ref,
                atol=1e-15,
                err_msg=f"Derivative mismatch for spline_type={alias!r}",
            )

    def test_rmts_subclasses_hierarchy_and_errors(self):
        # 1. Class hierarchy and names
        self.assertTrue(issubclass(RMTB, RMTS))
        self.assertTrue(issubclass(RMTC, RMTS))
        self.assertEqual(RMTB.name, "RMTB")
        self.assertEqual(RMTC.name, "RMTC")
        self.assertEqual(RMTS.name, "RMTS")

        import pytest

        with pytest.warns(FutureWarning, match="RMTB is deprecated"):
            sm_b = RMTB(xlimits=self.xlimits, print_global=False)
        with pytest.warns(FutureWarning, match="RMTC is deprecated"):
            sm_c = RMTC(xlimits=self.xlimits, print_global=False)
        sm_s = RMTS(xlimits=self.xlimits, print_global=False)

        self.assertIsInstance(sm_b, RMTS)
        self.assertIsInstance(sm_c, RMTS)
        self.assertIsInstance(sm_s, RMTS)

        self.assertEqual(sm_b.options["spline_type"], "B-Spline")
        self.assertEqual(sm_c.options["spline_type"], "Cubic_Hermite_Spline")
        self.assertEqual(sm_s.options["spline_type"], "B-Spline")

        # 2. Unsupported spline_type error
        with self.assertRaises(ValueError):
            sm_bad = RMTS(
                xlimits=self.xlimits, spline_type="unsupported", print_global=False
            )
            sm_bad.set_training_values(self.xt, self.yt)
            sm_bad.train()

        # 3. get_bspline and to_scipy on Cubic Hermite raises ValueError
        sm_c.set_training_values(self.xt, self.yt)
        with Silence():
            sm_c.train()
        with self.assertRaises(ValueError):
            sm_c.get_bspline()
        with self.assertRaises(ValueError):
            sm_c.to_scipy()

    def test_rmts_2d_and_multioutput_equivalence(self):
        rng = np.random.default_rng(42)
        xlimits = np.array([[-1.0, 2.0], [0.0, 3.0]])
        xt = rng.uniform([-1.0, 0.0], [2.0, 3.0], size=(30, 2))
        yt = np.column_stack(
            [np.sin(xt[:, 0]) * np.cos(xt[:, 1]), xt[:, 0] ** 2 + xt[:, 1]]
        )
        xq = rng.uniform([-1.0, 0.0], [2.0, 3.0], size=(10, 2))

        # B-Spline 2D multi-output
        b_ref = RMTB(xlimits=xlimits, order=4, num_ctrl_pts=8, print_global=False)
        b_ref.set_training_values(xt, yt)
        with Silence():
            b_ref.train()

        b_s = RMTS(
            xlimits=xlimits,
            spline_type="B-Spline",
            order=4,
            num_ctrl_pts=8,
            print_global=False,
        )
        b_s.set_training_values(xt, yt)
        with Silence():
            b_s.train()

        np.testing.assert_allclose(b_s.sol_coeff, b_ref.sol_coeff, atol=1e-15)
        np.testing.assert_allclose(
            b_s.predict_values(xq), b_ref.predict_values(xq), atol=1e-15
        )
        np.testing.assert_allclose(
            b_s.predict_derivatives(xq, 0), b_ref.predict_derivatives(xq, 0), atol=1e-15
        )
        np.testing.assert_allclose(
            b_s.predict_derivatives(xq, 1), b_ref.predict_derivatives(xq, 1), atol=1e-15
        )

        dy_dyt_ref = b_ref.predict_output_derivatives(xq)
        dy_dyt_s = b_s.predict_output_derivatives(xq)
        np.testing.assert_allclose(dy_dyt_s[None], dy_dyt_ref[None], atol=1e-15)

        # Cubic Hermite 2D multi-output
        c_ref = RMTC(xlimits=xlimits, num_elements=4, print_global=False)
        c_ref.set_training_values(xt, yt)
        with Silence():
            c_ref.train()

        c_s = RMTS(
            xlimits=xlimits,
            spline_type="Cubic_Hermite",
            num_elements=4,
            print_global=False,
        )
        c_s.set_training_values(xt, yt)
        with Silence():
            c_s.train()

        np.testing.assert_allclose(c_s.sol_coeff, c_ref.sol_coeff, atol=1e-15)
        np.testing.assert_allclose(
            c_s.predict_values(xq), c_ref.predict_values(xq), atol=1e-15
        )
        np.testing.assert_allclose(
            c_s.predict_derivatives(xq, 0), c_ref.predict_derivatives(xq, 0), atol=1e-15
        )
        np.testing.assert_allclose(
            c_s.predict_derivatives(xq, 1), c_ref.predict_derivatives(xq, 1), atol=1e-15
        )

    def test_rmts_edge_cases_and_option_validation(self):
        # 1. Conflicting spline_type on deprecated subclasses raises ValueError
        import pytest

        with pytest.warns(FutureWarning), self.assertRaises(ValueError):
            RMTB(
                xlimits=self.xlimits,
                spline_type="Cubic_Hermite_Spline",
                print_global=False,
            )
        with pytest.warns(FutureWarning), self.assertRaises(ValueError):
            RMTC(xlimits=self.xlimits, spline_type="B-Spline", print_global=False)

        # 2. Invalid approx_order raises ValueError at training time
        sm_bad = RMTS(xlimits=self.xlimits, approx_order=3, print_global=False)
        sm_bad.set_training_values(self.xt, self.yt)
        with self.assertRaises(ValueError):
            sm_bad.train()

        # 3. num_elements accepts tuple in multi-D
        xlimits_2d = np.array([[0.0, 1.0], [0.0, 1.0]])
        xt_2d = np.array([[0.1, 0.2], [0.8, 0.3], [0.4, 0.9], [0.6, 0.6]])
        yt_2d = np.sum(xt_2d, axis=1, keepdims=True)
        sm_tuple = RMTS(
            xlimits=xlimits_2d,
            spline_type="Cubic_Hermite_Spline",
            num_elements=(3, 4),
            print_global=False,
        )
        sm_tuple.set_training_values(xt_2d, yt_2d)
        with Silence():
            sm_tuple.train()
        self.assertEqual(sm_tuple.predict_values(xt_2d).shape, (4, 1))

        # 4. Zero outputs & nonlinear_maxiter=0 still initialize self.mtx properly
        for maxiter in (0, 10):
            sm_zero = RMTS(
                xlimits=self.xlimits,
                spline_type="B-Spline",
                nonlinear_maxiter=maxiter,
                print_global=False,
            )
            sm_zero.set_training_values(self.xt, np.zeros_like(self.yt))
            with Silence():
                sm_zero.train()
            self.assertIsNotNone(sm_zero.mtx)
            np.testing.assert_allclose(
                sm_zero.predict_values(self.xref), 0.0, atol=1e-15
            )
            dydt = sm_zero.predict_output_derivatives(self.xref)
            self.assertEqual(
                dydt[None].shape, (self.xref.shape[0], self.xt.shape[0], 1)
            )

        # 5. extrapolate=False yields zero derivatives outside domain bounds
        for st, extra in [
            ("B-Spline", {"order": 4, "num_ctrl_pts": 8}),
            ("Cubic_Hermite_Spline", {"num_elements": 4}),
        ]:
            sm_noext = RMTS(
                xlimits=self.xlimits,
                spline_type=st,
                extrapolate=False,
                print_global=False,
                **extra,
            )
            sm_noext.set_training_values(self.xt, self.yt)
            with Silence():
                sm_noext.train()
            x_out = np.array([[-2.0], [30.0]])
            np.testing.assert_allclose(
                sm_noext.predict_derivatives(x_out, 0), 0.0, atol=1e-15
            )

    def test_cpp_regression_rmtb_and_rmtc(self):
        """Regression test against outputs captured from legacy C++ RMTB/RMTC (v2.14.1)."""
        xt1 = np.array([[0.0], [0.2], [0.45], [0.7], [0.9], [1.0]])
        yt1 = np.sin(np.pi * xt1) + 0.3 * xt1**2
        dyt1 = np.pi * np.cos(np.pi * xt1) + 0.6 * xt1
        xq1 = np.array([[-0.15], [0.0], [0.32], [0.81], [1.0], [1.15]])

        gx, gy = np.meshgrid(
            np.linspace(-0.9, 1.85, 5), np.linspace(0.1, 2.85, 5), indexing="ij"
        )
        xt2 = np.column_stack([gx.ravel(), gy.ravel()])
        yt2 = np.sin(xt2[:, 0:1]) * np.cos(0.5 * xt2[:, 1:2]) + 0.1 * xt2[:, 0:1] ** 2
        xq2 = np.array([[-0.5, 0.5], [0.4, 1.7], [1.8, 2.8]])
        xlimits2 = np.array([[-1.0, 2.0], [0.0, 3.0]])
        xq5 = np.array([[0.1], [0.32], [0.64], [0.92]])

        # 1. 1D RMTB with linear extrapolation (values + derivatives)
        m1 = RMTS(
            spline_type="B-Spline",
            xlimits=np.array([[0.0, 1.0]]),
            order=4,
            num_ctrl_pts=8,
            solver="lu",
            extrapolate=True,
            print_global=False,
        )
        m1.set_training_values(xt1, yt1)
        with Silence():
            m1.train()
        np.testing.assert_allclose(
            m1.predict_values(xq1)[:, 0],
            [
                -0.27273781948971215,
                0.09655845513571548,
                0.7924673650751136,
                0.7338276743279101,
                0.3970248843612719,
                0.11729560486786994,
            ],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            m1.predict_derivatives(xq1, 0)[:, 0],
            [
                2.461975164169518,
                2.461975164169518,
                1.562036598772621,
                -1.5506832813359392,
                -1.8648618632893477,
                -1.8648618632893477,
            ],
            atol=1e-10,
        )

        # 2. 2D RMTB with heterogeneous order/control points (values + d/dx0 + d/dx1)
        m2 = RMTS(
            spline_type="B-Spline",
            xlimits=xlimits2,
            order=[3, 4],
            num_ctrl_pts=[6, 7],
            solver="lu",
            extrapolate=False,
            print_global=False,
        )
        m2.set_training_values(xt2, yt2)
        with Silence():
            m2.train()
        np.testing.assert_allclose(
            m2.predict_values(xq2)[:, 0],
            [-0.40664063417733537, 0.2633303338825932, 0.4964903790285152],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            m2.predict_derivatives(xq2, 0)[:, 0],
            [0.7657427834062432, 0.6843714270416118, 0.31970418861234734],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            m2.predict_derivatives(xq2, 1)[:, 0],
            [0.05571878916680414, -0.14267784418981344, -0.473667415972054],
            atol=1e-10,
        )

        # 3. 1D RMTC with linear extrapolation (values + derivatives)
        m3 = RMTS(
            spline_type="Cubic_Hermite_Spline",
            xlimits=np.array([[0.0, 1.0]]),
            num_elements=4,
            solver="lu",
            extrapolate=True,
            print_global=False,
        )
        m3.set_training_values(xt1, yt1)
        with Silence():
            m3.train()
        np.testing.assert_allclose(
            m3.predict_values(xq1)[:, 0],
            [
                -0.27005994950814566,
                0.09626623597348857,
                0.7890605235628528,
                0.7346428766152102,
                0.39687736778180827,
                0.11919809275140386,
            ],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            m3.predict_derivatives(xq1, 0)[:, 0],
            [
                2.442174569877561,
                2.442174569877561,
                1.5556241437394094,
                -1.5555962570443358,
                -1.8511951668693638,
                -1.8511951668693638,
            ],
            atol=1e-10,
        )

        # 4. 2D RMTC with heterogeneous element grid (values + d/dx0 + d/dx1)
        m4 = RMTS(
            spline_type="Cubic_Hermite_Spline",
            xlimits=xlimits2,
            num_elements=[3, 4],
            solver="lu",
            extrapolate=False,
            print_global=False,
        )
        m4.set_training_values(xt2, yt2)
        with Silence():
            m4.train()
        np.testing.assert_allclose(
            m4.predict_values(xq2)[:, 0],
            [-0.4290241712364965, 0.26770311995129303, 0.5648493039197746],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            m4.predict_derivatives(xq2, 0)[:, 0],
            [0.835045836162696, 0.6698121636972664, 0.6065540865437227],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            m4.predict_derivatives(xq2, 1)[:, 0],
            [-0.15015358233264217, -0.13512685489649076, -1.104460624995153],
            atol=1e-10,
        )

        # 5. 1D RMTB & RMTC with training derivatives (values + derivatives)
        m5 = RMTS(
            spline_type="B-Spline",
            xlimits=np.array([[0.0, 1.0]]),
            order=4,
            num_ctrl_pts=8,
            approx_order=2,
            grad_weight=1.0,
            solver="lu",
            extrapolate=False,
            print_global=False,
        )
        m5.set_training_values(xt1, yt1)
        m5.set_training_derivatives(xt1, dyt1, 0)
        with Silence():
            m5.train()
        np.testing.assert_allclose(
            m5.predict_values(xq5)[:, 0],
            [
                0.31099546625271723,
                0.8741115868593925,
                1.0283534718690972,
                0.5035198693286369,
            ],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            m5.predict_derivatives(xq5, 0)[:, 0],
            [
                3.0529515312201974,
                1.8781847728476493,
                -0.9567897459379866,
                -2.490048135448547,
            ],
            atol=1e-10,
        )

        m6 = RMTS(
            spline_type="Cubic_Hermite_Spline",
            xlimits=np.array([[0.0, 1.0]]),
            num_elements=4,
            approx_order=2,
            grad_weight=1.0,
            solver="lu",
            extrapolate=False,
            print_global=False,
        )
        m6.set_training_values(xt1, yt1)
        m6.set_training_derivatives(xt1, dyt1, 0)
        with Silence():
            m6.train()
        np.testing.assert_allclose(
            m6.predict_values(xq5)[:, 0],
            [
                0.3147092809128741,
                0.8764240811612661,
                1.0267538649596586,
                0.5002392615093597,
            ],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            m6.predict_derivatives(xq5, 0)[:, 0],
            [
                3.0277795796882003,
                1.855778329731941,
                -0.9548454059459718,
                -2.487951254443737,
            ],
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
