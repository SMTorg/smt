"""
Comprehensive tests for PCE and SparsePCE surrogate models.
"""

import os
import pickle
import tempfile
import unittest

import numpy as np
import pytest
import scipy.stats as st

from smt.design_space import CategoricalVariable, DesignSpace, FloatVariable
from smt.surrogate_models import PCE, SparsePCE
from smt.surrogate_models.pce_backend import compute_ols_and_loo, parse_distributions
from smt.utils.checks import _enable_jax_x64


class TestPCEBackendAndAskey(unittest.TestCase):
    def test_askey_orthonormality_and_derivatives(self):
        specs = [
            ("uniform", -2.0, 3.0),
            ("normal", 1.5, 0.7),
            ("gamma", 2.5, 1.3, -1.0),
            ("exponential", 0.8, 0.2),
            ("beta", 2.2, 4.5, -1.0, 4.0),
            ("beta", 0.5, 0.5, -1.0, 1.0),  # Chebyshev / arcsine boundary test
            st.uniform(loc=-1.0, scale=4.0),
            st.norm(loc=-0.5, scale=2.0),
            st.gamma(a=3.0, scale=0.5, loc=1.0),
            st.expon(loc=0.0, scale=2.0),
            st.beta(a=1.5, b=3.0, loc=2.0, scale=5.0),
        ]

        max_deg = 7
        for idx, spec in enumerate(specs):
            marg = parse_distributions(nx=1, distributions=spec)[0]
            self.assertTrue(marg.is_askey, msg=f"Spec {idx} should resolve to Askey.")
            x_q, w_q = marg.gauss_quadrature_1d(max_deg + 2)
            self.assertAlmostEqual(float(np.sum(w_q)), 1.0, places=13)

            V = marg.eval_1d(x_q, max_deg, with_deriv=False)
            gram = (V * w_q[:, None]).T @ V
            np.testing.assert_allclose(
                gram, np.eye(max_deg + 1), atol=1e-11, err_msg=f"Failed for spec {spec}"
            )

            # Check analytic derivatives against central finite differences
            pts = x_q[1:-1][:3]
            eps = 1e-6
            _, dV = marg.eval_1d(pts, max_deg, with_deriv=True)
            V_plus = marg.eval_1d(pts + eps, max_deg)
            V_minus = marg.eval_1d(pts - eps, max_deg)
            fd = (V_plus - V_minus) / (2.0 * eps)
            np.testing.assert_allclose(dV, fd, atol=1e-6, rtol=1e-6)

    def test_non_askey_scipy_marginal_fallback(self):
        rv_logn = st.lognorm(s=0.4, scale=np.exp(0.5))
        sm = PCE(
            distributions=[rv_logn], degree=3, method="quadrature", print_global=False
        )
        xq, wq = sm.quadrature_points(order=6)
        yq = xq[:, 0] ** 2
        sm.options["weights"] = wq
        sm.set_training_values(xq, yq)
        sm.train()

        # Lognormal exact E[X^2] = exp(2*mu + 2*sigma^2)
        mu, s = 0.5, 0.4
        expected_ex2 = np.exp(2.0 * mu + 2.0 * s * s)
        self.assertAlmostEqual(float(sm.mean()[0]), expected_ex2, places=6)

        # Analytic derivative vs FD
        x_test = np.array([[1.2], [2.5]])
        eps = 1e-6
        dy = sm.predict_derivatives(x_test, 0)
        fd = (sm.predict_values(x_test + eps) - sm.predict_values(x_test - eps)) / (
            2.0 * eps
        )
        np.testing.assert_allclose(dy, fd, atol=1e-5)

        with self.assertRaises(NotImplementedError):
            sm.get_jax_predictor()

    def test_analytical_loo_matches_bruteforce_cv(self):
        rng = np.random.default_rng(123)
        X = rng.uniform(-1.0, 1.0, size=(30, 3))
        # Design matrix [1, x0, x1, x0^2, x0*x1]
        A = np.column_stack(
            [np.ones(30), X[:, 0], X[:, 1], X[:, 0] ** 2, X[:, 0] * X[:, 1]]
        )
        y = 1.5 + 2.0 * X[:, 0] - 0.5 * X[:, 1] ** 2 + 0.3 * np.sin(X[:, 2])
        var_y = float(np.var(y))

        _, loo_analytic, mloo_analytic = compute_ols_and_loo(A, y, var_y)

        # Brute-force N-fold CV
        sq_errs = []
        for i in range(len(y)):
            mask = np.arange(len(y)) != i
            c_i = np.linalg.lstsq(A[mask], y[mask], rcond=None)[0]
            pred_i = float(A[i] @ c_i)
            sq_errs.append((y[i] - pred_i) ** 2)
        loo_bf = float(np.mean(sq_errs)) / var_y
        self.assertAlmostEqual(loo_analytic, loo_bf, places=11)

        t_corr = (30 / (30 - 5)) * (1.0 + np.trace(np.linalg.inv(A.T @ A)))
        self.assertAlmostEqual(mloo_analytic, t_corr * loo_bf, places=11)


class TestPCEModelsAndUQ(unittest.TestCase):
    def test_exact_polynomial_moments_and_derivatives(self):
        # f(x0, x1) = 3 + 2*x0 + x0*x1^2 with X0 ~ U(0, 2), X1 ~ N(1, 0.5^2)
        def fun(x):
            return (3.0 + 2.0 * x[:, 0] + x[:, 0] * (x[:, 1] ** 2))[:, None]

        # Exact analytical moments:
        # E[X0]=1, Var[X0]=1/3, E[X0^2]=4/3
        # E[X1]=1, Var[X1]=0.25, E[X1^2]=1.25, E[X1^4]=1 + 6*1*0.25 + 3*0.25^2 = 2.6875
        exact_mean = 3.0 + 2.0 * 1.0 + 1.0 * 1.25  # 6.25
        # Y = 3 + X0 * (2 + X1^2), Var(X0) = 1/3, E[X0^2]=4/3, Var(2+X1^2) = E[X1^4] - E[X1^2]^2
        ex_x1_sq_var = 2.6875 - 1.25**2  # 1.125
        exact_var = (
            (4.0 / 3.0) * (2.0 + 1.25) ** 2
            + (4.0 / 3.0) * ex_x1_sq_var
            - (1.0 * 3.25) ** 2
        )

        sm = PCE(
            degree=3,
            distributions=[("uniform", 0.0, 2.0), ("normal", 1.0, 0.5)],
            method="ols",
            print_global=False,
        )
        xt = sm.sample_design(60, sampling="sobol", seed=42)
        yt = fun(xt)
        sm.set_training_values(xt, yt)
        sm.train()

        rng = np.random.default_rng(1)
        x_test = np.column_stack([rng.uniform(0.0, 2.0, 25), rng.normal(1.0, 0.5, 25)])
        np.testing.assert_allclose(sm.predict_values(x_test), fun(x_test), atol=1e-10)
        self.assertAlmostEqual(float(sm.mean()[0]), exact_mean, places=10)
        self.assertAlmostEqual(float(sm.variance()[0]), exact_var, places=10)
        self.assertAlmostEqual(float(sm.std()[0]), np.sqrt(exact_var), places=10)

        # Exact derivatives
        dy0 = 2.0 + x_test[:, 1] ** 2
        dy1 = 2.0 * x_test[:, 0] * x_test[:, 1]
        np.testing.assert_allclose(
            sm.predict_derivatives(x_test, 0)[:, 0], dy0, atol=1e-10
        )
        np.testing.assert_allclose(
            sm.predict_derivatives(x_test, 1)[:, 0], dy1, atol=1e-10
        )

    def test_ishigami_sparse_pce_and_sobol_indices(self):
        a, b = 7.0, 0.1

        def ishigami(x):
            y = (
                np.sin(x[:, 0])
                + a * (np.sin(x[:, 1]) ** 2)
                + b * (x[:, 2] ** 4) * np.sin(x[:, 0])
            )
            return y[:, None]

        # Exact analytical moments & Sobol indices for U([-pi, pi]^3)
        exact_mean = a / 2.0
        exact_var = (a**2) / 8.0 + (b * np.pi**4) / 5.0 + (b**2 * np.pi**8) / 18.0 + 0.5
        v1 = 0.5 * (1.0 + (b * np.pi**4) / 5.0) ** 2
        v2 = (a**2) / 8.0
        v13 = (b**2 * np.pi**8) * (1.0 / 18.0 - 1.0 / 50.0)
        exact_s1 = np.array([v1 / exact_var, v2 / exact_var, 0.0])
        exact_st = np.array([(v1 + v13) / exact_var, v2 / exact_var, v13 / exact_var])

        xlimits = np.tile(np.array([[-np.pi, np.pi]]), (3, 1))
        for method in ("lars", "omp", "lasso"):
            sm = SparsePCE(
                xlimits=xlimits,
                method=method,
                degree=range(4, 10),
                q_norm=0.75,
                print_global=False,
            )
            xt = sm.sample_design(220, sampling="sobol", seed=123)
            sm.set_training_values(xt, ishigami(xt))
            sm.train()

            self.assertLess(sm.multi_indices.shape[0], sm.n_basis_candidate_)
            self.assertAlmostEqual(float(sm.mean()[0]), exact_mean, delta=0.1)
            self.assertAlmostEqual(float(sm.variance()[0]), exact_var, delta=0.5)
            self.assertGreater(float(sm.q2_loo_[0]), 0.98)

            sobol = sm.sobol_indices(second_order=True)
            np.testing.assert_allclose(sobol["first_order"], exact_s1, atol=0.03)
            np.testing.assert_allclose(sobol["total"], exact_st, atol=0.03)
            self.assertAlmostEqual(
                sobol["second_order"][0, 2], v13 / exact_var, delta=0.03
            )
            self.assertAlmostEqual(sobol["second_order"][0, 1], 0.0, delta=0.02)

            s_02_closed = sm.subset_sobol([0, 2], closed=True)
            self.assertAlmostEqual(s_02_closed, (v1 + v13) / exact_var, delta=0.03)

    def test_conditional_expectation_openturns_style(self):
        # Coupled test problem with non-zero mean inputs:
        # X0 ~ U(0, 2), X1 ~ N(1.5, 0.5), X2 ~ U(-1, 1)
        # Y = 2 + x0^2 + 3 * x0 * x1 + 0.8 * (x1^2) * x2 + 4 * x2^2
        # Analytical E[Y | X0 = x0]:
        # E[X1] = 1.5, E[X2] = 0, E[X2^2] = 1/3
        # => E[Y | X0] = 2 + x0^2 + 3 * 1.5 * x0 + 4 * (1/3) = (10/3) + 4.5*x0 + x0^2
        def fun(x):
            return (
                2.0
                + x[:, 0] ** 2
                + 3.0 * x[:, 0] * x[:, 1]
                + 0.8 * (x[:, 1] ** 2) * x[:, 2]
                + 4.0 * x[:, 2] ** 2
            )[:, None]

        sm = PCE(
            degree=3,
            method="ols",
            distributions=[
                ("uniform", 0.0, 2.0),
                ("normal", 1.5, 0.5),
                ("uniform", -1.0, 1.0),
            ],
            print_global=False,
        )
        xt = sm.sample_design(80, sampling="sobol", seed=42)
        sm.set_training_values(xt, fun(xt))
        sm.train()

        cond_0 = sm.conditional_expectation([0])
        self.assertEqual(cond_0.nx, 1)
        self.assertEqual(cond_0.ny, 1)

        x0_grid = np.linspace(0.0, 2.0, 25).reshape(-1, 1)
        exact_e_y_x0 = 10.0 / 3.0 + 4.5 * x0_grid + x0_grid**2
        np.testing.assert_allclose(
            cond_0.predict_values(x0_grid), exact_e_y_x0, atol=1e-9
        )

        # Law of Total Variance identity: Var(E[Y | X_u]) == S_u^closed * Var(Y)
        s0_closed = sm.subset_sobol([0], closed=True)
        self.assertAlmostEqual(
            float(cond_0.variance()[0]), float(s0_closed * sm.variance()[0]), places=12
        )
        self.assertAlmostEqual(float(cond_0.mean()[0]), float(sm.mean()[0]), places=12)

        # Exact derivative of conditional expectation
        exact_d_x0 = 4.5 + 2.0 * x0_grid
        np.testing.assert_allclose(
            cond_0.predict_derivatives(x0_grid, 0), exact_d_x0, atol=1e-9
        )

        # Non-sequential reordering of conditioning variables [2, 0]
        cond_20 = sm.conditional_expectation([2, 0])
        self.assertEqual(cond_20.nx, 2)
        x20 = np.column_stack([np.linspace(-1.0, 1.0, 10), np.linspace(0.0, 2.0, 10)])
        exact_e_y_x20 = (
            2.0
            + x20[:, 1:2] ** 2
            + 4.5 * x20[:, 1:2]
            + 0.8 * 2.5 * x20[:, 0:1]
            + 4.0 * x20[:, 0:1] ** 2
        )
        np.testing.assert_allclose(
            cond_20.predict_values(x20), exact_e_y_x20, atol=1e-9
        )
        self.assertAlmostEqual(
            float(cond_20.variance()[0]),
            float(sm.subset_sobol([2, 0], closed=True) * sm.variance()[0]),
            places=12,
        )

    def test_high_dim_sparse_recovery_and_max_interaction(self):
        # 12D additive/pairwise sparse function estimated with only 120 samples
        def fun(x):
            return (
                1.0 + 3.0 * x[:, 0] - 2.0 * x[:, 3] ** 2 + 1.5 * x[:, 7] * x[:, 11]
            )[:, None]

        xlimits = np.tile(np.array([[-1.0, 1.0]]), (12, 1))
        sm = SparsePCE(
            xlimits=xlimits,
            method="lars",
            degree=[1, 2, 3],
            q_norm=1.0,
            max_interaction=2,
            print_global=False,
        )
        xt = sm.sample_design(120, sampling="lhs", seed=7)
        sm.set_training_values(xt, fun(xt))
        sm.train()

        xe = sm.sample_design(50, sampling="mc", seed=99)
        np.testing.assert_allclose(sm.predict_values(xe), fun(xe), atol=1e-8)
        self.assertTrue(np.all(np.sum(sm.multi_indices > 0, axis=1) <= 2))
        sobol = sm.sobol_indices()
        # Unused dimensions must have near-zero Sobol importance
        for inactive_dim in (1, 2, 4, 5, 6, 8, 9, 10):
            self.assertAlmostEqual(float(sobol["total"][inactive_dim]), 0.0, delta=1e-8)

    def test_multi_output_and_quadrature(self):
        xlimits = np.array([[-1.0, 1.0], [-1.0, 1.0]])

        def fun_multi(x):
            y0 = 1.0 + 2.0 * x[:, 0] + x[:, 0] ** 2
            y1 = -0.5 + 3.0 * x[:, 1] - x[:, 0] * x[:, 1]
            return np.column_stack([y0, y1])

        # Quadrature method
        sm_q = PCE(xlimits=xlimits, degree=3, method="quadrature", print_global=False)
        xq, wq = sm_q.quadrature_points(order=4)
        sm_q.options["weights"] = wq
        sm_q.set_training_values(xq, fun_multi(xq))
        sm_q.train()

        xe = sm_q.sample_design(20, sampling="halton", seed=5)
        np.testing.assert_allclose(sm_q.predict_values(xe), fun_multi(xe), atol=1e-11)
        np.testing.assert_allclose(sm_q.mean(), [4.0 / 3.0, -0.5], atol=1e-12)

        # SparsePCE multi-output with distinct sparsity per output
        sm_sp = SparsePCE(xlimits=xlimits, degree=range(1, 5), print_global=False)
        xt = sm_sp.sample_design(35, sampling="sobol", seed=11)
        sm_sp.set_training_values(xt, fun_multi(xt))
        sm_sp.train()
        np.testing.assert_allclose(sm_sp.predict_values(xe), fun_multi(xe), atol=1e-10)

        sobol_all = sm_sp.sobol_indices()
        self.assertEqual(sobol_all["first_order"].shape, (2, 2))
        self.assertAlmostEqual(float(sobol_all["total"][1, 0]), 0.0, delta=1e-10)
        sobol_out1 = sm_sp.sobol_indices(output=1)
        self.assertEqual(sobol_out1["first_order"].shape, (2,))
        cond_0 = sm_sp.conditional_expectation([0])
        np.testing.assert_allclose(
            cond_0.predict_values(xe[:, :1]),
            np.column_stack([1.0 + 2.0 * xe[:, 0] + xe[:, 0] ** 2, np.full(20, -0.5)]),
            atol=1e-10,
        )

    def test_design_space_caching_persistence_and_edge_cases(self):
        ds = DesignSpace([FloatVariable(0.0, 2.0), FloatVariable(-1.0, 1.0)])
        xt = np.array([[0.1, -0.5], [0.5, 0.0], [1.0, 0.5], [1.5, -0.2], [1.9, 0.8]])
        yt = 2.0 * xt[:, 0:1] - 0.5 * xt[:, 1:2] ** 2

        with tempfile.TemporaryDirectory() as tmpdir:
            sm1 = PCE(design_space=ds, degree=2, data_dir=tmpdir, print_global=False)
            sm1.set_training_values(xt, yt)
            sm1.train()

            sm2 = PCE(design_space=ds, degree=2, data_dir=tmpdir, print_global=False)
            sm2.set_training_values(xt, yt)
            sm2.train()  # cache hit
            np.testing.assert_allclose(sm1.predict_values(xt), sm2.predict_values(xt))

            pkl_path = os.path.join(tmpdir, "pce.pkl")
            sm1.save(pkl_path)
            sm_loaded = PCE.load(pkl_path)
            np.testing.assert_allclose(
                sm_loaded.predict_values(xt), sm1.predict_values(xt)
            )
            np.testing.assert_allclose(sm_loaded.mean(), sm1.mean())

        # Edge cases: single training point, constant target, rank-deficient OLS, degree 0
        sm_one = SparsePCE(print_global=False)
        sm_one.set_training_values(np.array([[0.2, 0.3]]), np.array([[5.0]]))
        sm_one.train()
        np.testing.assert_allclose(
            sm_one.predict_values(np.array([[0.9, -1.0]])), [[5.0]]
        )

        sm_const = SparsePCE(print_global=False)
        sm_const.set_training_values(xt, np.full((len(xt), 1), 3.25))
        sm_const.train()
        np.testing.assert_allclose(sm_const.mean(), [3.25])
        np.testing.assert_allclose(sm_const.variance(), [0.0])

        sm_under = PCE(degree=3, method="ols", print_global=False)
        sm_under.set_training_values(xt, yt)
        sm_under.train()
        self.assertTrue(np.isnan(sm_under.loo_error_[0]))
        self.assertEqual(sm_under.predict_values(xt).shape, (5, 1))

        # Validation errors
        ds_cat = DesignSpace(
            [CategoricalVariable(values=["a", "b"]), FloatVariable(0.0, 1.0)]
        )
        sm_bad_ds = PCE(design_space=ds_cat, print_global=False)
        sm_bad_ds.set_training_values(
            np.array([[0.0, 0.1], [1.0, 0.9]]), np.array([[1.0], [2.0]])
        )
        with self.assertRaises(ValueError):
            sm_bad_ds.train()

        with self.assertRaises(RuntimeError):
            PCE().conditional_expectation([0])
        with self.assertRaises(ValueError):
            sm_const.conditional_expectation([])
        with self.assertRaises(ValueError):
            sm_const.conditional_expectation([5])
        with self.assertRaises(ValueError):
            sm_const.conditional_expectation([0, 0])
        with self.assertRaises(ValueError):
            sm_const.sobol_indices(output=2)
        with self.assertRaises(ValueError):
            sm_const.sample_design(0)
        with self.assertRaises(ValueError):
            sm_const.sample_design(5, sampling="unknown")
        with self.assertRaises(ValueError):
            sm_const.quadrature_points(order=0)
        with self.assertRaises(ValueError):
            sm_const.quadrature_points(order=[2, 2, 2])
        with self.assertRaises(ValueError):
            sm_const.quadrature_points(order=[100, 100, 100])

    def test_distribution_variants_anisotropy_and_edge_solvers(self):
        # Test dict/string/rate specs, xlimits uniform fallback, anisotropy weights, q-norm list
        sm = PCE(
            xlimits=np.array([[0.0, 2.0], [0.0, 5.0]]),
            distributions=[
                {"type": "uniform"},
                {"type": "gamma", "shape": 2.0, "rate": 1.5, "loc": 0.0},
            ],
            degree=[1, 2, 3],
            q_norm=[0.75, 1.0],
            anisotropy_weights=[1.0, 2.0],
            method="lars",
            print_global=False,
        )
        xt = sm.sample_design(64, sampling="sobol", seed=21)
        yt = (1.0 + 2.0 * xt[:, 0] ** 2 + 0.5 * xt[:, 1])[:, None]
        sm.set_training_values(xt, yt)
        sm.train()
        self.assertLess(sm.loo_error_[0], 1e-6)
        self.assertGreater(sm.subset_sobol([0], closed=False), 0.1)

        # Rank-deficient duplicate training points + reg=0.0 fallback + small N sparse fit
        x_dup = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.5, 0.2]])
        y_dup = np.array([[1.0], [1.0], [2.0], [2.0], [1.5]])
        sm_dup = PCE(degree=2, method="ols", reg=0.0, print_global=False)
        sm_dup.set_training_values(x_dup, y_dup)
        sm_dup.train()
        self.assertEqual(sm_dup.predict_values(x_dup).shape, (5, 1))

        # Tiny 2-point dataset with sparse OMP
        sm_tiny = SparsePCE(method="omp", degree=2, print_global=False)
        sm_tiny.set_training_values(
            np.array([[0.0, 0.0], [1.0, 0.5]]), np.array([[1.0], [3.0]])
        )
        sm_tiny.train()
        self.assertEqual(sm_tiny.predict_values(np.array([[0.5, 0.25]])).shape, (1, 1))

        # Non-Askey marginal with use_jax=True gracefully falls back to CPU
        sm_logn = PCE(
            distributions=[st.lognorm(s=0.3), {"type": "exponential", "rate": 2.0}],
            degree=2,
            use_jax=True,
            print_global=False,
        )
        xt_l = sm_logn.sample_design(32, sampling="halton", seed=3)
        sm_logn.set_training_values(xt_l, (xt_l[:, 0] + xt_l[:, 1])[:, None])
        sm_logn.train()
        self.assertEqual(sm_logn.predict_values(xt_l).shape, (32, 1))

        # Validation of bad distribution specs
        for bad_dist in [
            ("normal", 0.0, -1.0),
            ("gamma", -1.0, 1.0),
            {"type": "gamma", "shape": 2.0, "scale": 1.0, "rate": 1.0},
            {"type": "exponential", "scale": 1.0, "rate": 1.0},
            ("beta", 0.0, 1.0),
            ("unknown_dist", 1.0),
        ]:
            with self.assertRaises(ValueError):
                parse_distributions(nx=1, distributions=bad_dist)


class TestPCEJAXSupport(unittest.TestCase):
    def test_pce_and_conditional_jax_predictor(self):
        import subprocess
        import sys
        import textwrap

        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        sm = SparsePCE(
            distributions=[
                ("uniform", -1.0, 2.0),
                ("normal", 0.5, 0.4),
                ("gamma", 2.0, 0.5),
            ],
            degree=[1, 2, 3],
            print_global=False,
        )
        xt = sm.sample_design(70, sampling="sobol", seed=42)
        yt = (
            1.5
            + 2.0 * xt[:, 0:1]
            - 0.7 * (xt[:, 1:2] ** 2)
            + 0.4 * xt[:, 0:1] * xt[:, 2:3]
        )
        sm.set_training_values(xt, yt)
        sm.train()

        x_eval = sm.sample_design(12, sampling="halton", seed=9)
        y_cpu = sm.predict_values(x_eval)
        dy1_cpu = sm.predict_derivatives(x_eval, 1)

        prev_x64 = bool(jax.config.jax_enable_x64)
        with _enable_jax_x64(jax):
            pred_jax = sm.get_jax_predictor()
            np.testing.assert_allclose(
                np.asarray(pred_jax(jnp.asarray(x_eval))), y_cpu, atol=1e-11
            )

            pt = jnp.asarray(x_eval[0])
            self.assertEqual(pred_jax(pt).ndim, 0)
            g_ad = np.asarray(jax.grad(pred_jax)(pt))
            g_cpu = np.array(
                [sm.predict_derivatives(x_eval[:1], k)[0, 0] for k in range(3)]
            )
            np.testing.assert_allclose(g_ad, g_cpu, atol=1e-10)

            # Conditional expectation inherits JAX compatibility
            cond = sm.conditional_expectation([0])
            pred_cond_jax = cond.get_jax_predictor()
            x0 = x_eval[:, :1]
            np.testing.assert_allclose(
                np.asarray(pred_cond_jax(jnp.asarray(x0))),
                cond.predict_values(x0),
                atol=1e-11,
            )

            # Constant PCE JAX grad returns exact zeros
            sm_c = PCE(degree=0, print_global=False)
            sm_c.set_training_values(xt, np.full((len(xt), 1), 2.5))
            sm_c.train()
            np.testing.assert_allclose(
                np.asarray(jax.grad(sm_c.get_jax_predictor())(pt)),
                np.zeros(3),
                atol=1e-14,
            )
        self.assertEqual(bool(jax.config.jax_enable_x64), prev_x64)

        sm.options["use_jax"] = True
        np.testing.assert_allclose(sm.predict_values(x_eval), y_cpu, atol=1e-11)
        np.testing.assert_allclose(
            sm.predict_derivatives(x_eval, 1), dy1_cpu, atol=1e-11
        )
        self.assertEqual(bool(jax.config.jax_enable_x64), prev_x64)

        # Pickle roundtrip strips JIT closures cleanly
        loaded = pickle.loads(pickle.dumps(sm))
        np.testing.assert_allclose(loaded.predict_values(x_eval), y_cpu, atol=1e-11)

        # Pristine subprocess check: neither use_jax nor get_jax_predictor mutates x64
        code = textwrap.dedent(
            """
            import warnings
            import jax
            import jax.numpy as jnp
            import numpy as np
            from smt.surrogate_models import PCE

            assert not jax.config.jax_enable_x64
            sm = PCE(degree=2, use_jax=True, print_global=False)
            xt = np.array([[0.1, 0.2], [0.8, 0.3], [0.4, 0.9], [0.2, 0.7], [0.9, 0.8], [0.5, 0.5]])
            yt = 1.0 + 2.0 * xt[:, :1] - 0.5 * (xt[:, 1:2] ** 2)
            sm.set_training_values(xt, yt)
            sm.train()
            x_eval = np.array([[0.3, 0.6], [0.7, 0.4]])
            y_j = sm.predict_values(x_eval)
            dy_j = sm.predict_derivatives(x_eval, 1)
            sm.options["use_jax"] = False
            np.testing.assert_allclose(y_j, sm.predict_values(x_eval), atol=1e-11)
            np.testing.assert_allclose(dy_j, sm.predict_derivatives(x_eval, 1), atol=1e-11)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pred = sm.get_jax_predictor()
                _ = jax.jit(pred)(jnp.asarray(x_eval))
                assert not any("float64" in str(item.message) for item in w)
            assert not jax.config.jax_enable_x64
            """
        )
        subprocess.check_call([sys.executable, "-c", code])


if __name__ == "__main__":
    unittest.main()
