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

from smt.design_space import (
    CategoricalVariable,
    DesignSpace,
    FloatVariable,
    IntegerVariable,
)
from smt.surrogate_models import PCE, SparsePCE
from smt.surrogate_models.pce import (
    _infer_nx_for_design,
    _normalize_degree_sequence,
    _normalize_method_name,
    _normalize_qnorm_sequence,
)
from smt.surrogate_models.pce_backend import (
    MarginalSpec,
    assemble_basis_deriv_matrix,
    compute_ols_and_loo,
    fit_pce_at_basis,
    generate_multi_indices,
    parse_distributions,
    select_adaptive_pce,
)
from smt.surrogate_models.rbf_idw_backend import _enable_x64 as _enable_jax_x64


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

        # Non-Askey marginal evaluates accurately on CPU and rejects JAX predictor
        sm_logn = PCE(
            distributions=[st.lognorm(s=0.3), {"type": "exponential", "rate": 2.0}],
            degree=2,
            print_global=False,
        )
        xt_l = sm_logn.sample_design(32, sampling="halton", seed=3)
        sm_logn.set_training_values(xt_l, (xt_l[:, 0] + xt_l[:, 1])[:, None])
        sm_logn.train()
        self.assertEqual(sm_logn.predict_values(xt_l).shape, (32, 1))
        with self.assertRaises(NotImplementedError):
            sm_logn.get_jax_predictor()

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

        # Bug regression: per-dim tuple and single-spec list disambiguation
        m_tuple = parse_distributions(nx=2, distributions=("normal", "uniform"))
        self.assertEqual([m.family for m in m_tuple], ["hermite", "legendre"])
        m_list_3d = parse_distributions(nx=3, distributions=["uniform", -1.0, 2.0])
        self.assertEqual(len(m_list_3d), 3)
        self.assertTrue(
            all(m.family == "legendre" and m.bounds == (-1.0, 2.0) for m in m_list_3d)
        )
        sm_inf = PCE(distributions=("normal", "uniform"), print_global=False)
        self.assertEqual(sm_inf.sample_design(4, sampling="mc", seed=0).shape, (4, 2))

        # Validation: data_dir=None allowed post-init; booleans rejected cleanly
        sm_val = PCE(print_global=False)
        sm_val.options["data_dir"] = None
        for bad_kw in [
            {"max_interaction": True},
            {"max_basis": False},
            {"reg": -1.0},
            {"max_basis": 0},
        ]:
            bad_sm = PCE(print_global=False, **bad_kw)
            bad_sm.set_training_values(
                np.array([[0.0], [0.5], [1.0]]), np.array([[0.0], [1.0], [2.0]])
            )
            with self.assertRaises(ValueError):
                bad_sm.train()

    def test_stieltjes_empirical_moments_solvers_and_regressions(self):
        # 1. Discretized Stieltjes marginal on lognormal preserves exact polynomial linearity
        rv_ln = st.lognorm(s=0.4)
        sm_st = PCE(
            distributions=[("stieltjes", rv_ln)],
            degree=2,
            method="quadrature",
            print_global=False,
        )
        xq, wq = sm_st.quadrature_points(order=6)
        sm_st.options["weights"] = wq
        sm_st.set_training_values(xq, (2.0 * xq[:, 0] - 1.5)[:, None])
        sm_st.train()
        self.assertAlmostEqual(
            float(sm_st.mean()[0]), 2.0 * rv_ln.mean() - 1.5, places=10
        )
        self.assertAlmostEqual(float(sm_st.variance()[0]), 4.0 * rv_ln.var(), places=10)
        x_chk = np.array([[0.8], [1.5], [2.3]])
        np.testing.assert_allclose(
            sm_st.predict_values(x_chk), 2.0 * x_chk - 1.5, atol=1e-10
        )
        np.testing.assert_allclose(
            sm_st.predict_derivatives(x_chk, 0), np.full((3, 1), 2.0), atol=1e-10
        )

        # 2. Data-driven empirical / aPCE marginal matches exact sample moments
        rng = np.random.default_rng(77)
        samples = rng.weibull(2.2, size=120)
        sm_emp = PCE(
            distributions=[{"type": "empirical", "data": samples}],
            degree=3,
            method="quadrature",
            print_global=False,
        )
        xq_e, wq_e = sm_emp.quadrature_points(order=5)
        sm_emp.options["weights"] = wq_e
        sm_emp.set_training_values(xq_e, (3.0 * xq_e[:, 0] + 1.0)[:, None])
        sm_emp.train()
        self.assertAlmostEqual(
            float(sm_emp.mean()[0]), 3.0 * float(np.mean(samples)) + 1.0, places=10
        )
        self.assertAlmostEqual(
            float(sm_emp.variance()[0]), 9.0 * float(np.var(samples)), places=10
        )

        # 3. Anisotropy weights < 1.0 bug fix (V_cache and quadrature order)
        sm_aniso = PCE(
            degree=2,
            q_norm=1.0,
            anisotropy_weights=[0.5, 2.0],
            xlimits=[[-1.0, 1.0], [-1.0, 1.0]],
            method="quadrature",
            print_global=False,
        )
        xq_a, wq_a = sm_aniso.quadrature_points()
        # floor(2 / 0.5) = 4 -> order 5 on dim 0; floor(2 / 2.0) = 1 -> order 2 on dim 1 -> 10 pts
        self.assertEqual(xq_a.shape, (10, 2))
        sm_aniso.options["weights"] = wq_a
        sm_aniso.set_training_values(xq_a, (xq_a[:, 0] ** 4 + xq_a[:, 1])[:, None])
        sm_aniso.train()
        np.testing.assert_allclose(
            sm_aniso.predict_values(np.array([[0.6, -0.3]])),
            [[0.6**4 - 0.3]],
            atol=1e-11,
        )
        sm_aniso_ols = PCE(
            degree=[1, 2],
            q_norm=1.0,
            anisotropy_weights=[0.5, 2.0],
            method="ols",
            print_global=False,
        )
        xt_a = sm_aniso_ols.sample_design(24, sampling="sobol", seed=4, nx=2)
        sm_aniso_ols.set_training_values(xt_a, (xt_a[:, 0] ** 4 + xt_a[:, 1])[:, None])
        sm_aniso_ols.train()
        np.testing.assert_allclose(
            sm_aniso_ols.predict_values(np.array([[0.6, -0.3]])),
            [[0.6**4 - 0.3]],
            atol=1e-10,
        )

        # 4. Multi-output mean_mloo guard and q_norm max_basis fallback
        xt_mo = np.array([[0.0], [0.25], [0.75], [1.0]])
        yt_mo = np.column_stack([np.ones(4), xt_mo[:, 0]])
        sm_mo = PCE(degree=[4, 1], method="ols", print_global=False)
        sm_mo.set_training_values(xt_mo, yt_mo)
        sm_mo.train()
        self.assertEqual(sm_mo.degree_, 1)

        sm_qfall = SparsePCE(
            degree=6,
            q_norm=[1.0, 0.35],
            max_basis=40,
            method="omp",
            print_global=False,
        )
        xt_qf = sm_qfall.sample_design(60, sampling="sobol", seed=5, nx=5)
        sm_qfall.set_training_values(
            xt_qf, (1.0 + 2.0 * xt_qf[:, 0] - xt_qf[:, 3])[:, None]
        )
        sm_qfall.train()
        self.assertAlmostEqual(sm_qfall.q_norm_, 0.35)

        # 5. Subspace Pursuit ('sp'), BCS ('bcs'), and Weighted Least Squares
        sm_sp_tpl = SparsePCE(degree=3, print_global=False)
        xt_sp = sm_sp_tpl.sample_design(64, sampling="sobol", seed=11, nx=4)
        y_sp = (
            2.0
            + 3.0 * xt_sp[:, 0]
            - 1.5 * xt_sp[:, 2] ** 2
            + 0.8 * xt_sp[:, 1] * xt_sp[:, 3]
        )[:, None]
        xt_test = sm_sp_tpl.sample_design(40, sampling="halton", seed=19, nx=4)
        y_test = (
            2.0
            + 3.0 * xt_test[:, 0]
            - 1.5 * xt_test[:, 2] ** 2
            + 0.8 * xt_test[:, 1] * xt_test[:, 3]
        )[:, None]
        for solver in ("sp", "bcs"):
            sm_s = SparsePCE(degree=3, method=solver, print_global=False)
            sm_s.set_training_values(xt_sp, y_sp)
            sm_s.train()
            np.testing.assert_allclose(sm_s.predict_values(xt_test), y_test, atol=1e-8)

        # Weighted regression exact polynomial fit under positive importance weights
        sm_w = PCE(
            degree=2,
            method="ols",
            xlimits=[[-1.0, 1.0], [-1.0, 1.0]],
            weights=np.linspace(0.5, 2.0, 16),
            print_global=False,
        )
        xt_w = sm_w.sample_design(16, sampling="halton", seed=2, nx=2)
        y_w = (1.0 + 2.0 * xt_w[:, 0] ** 2 - 0.5 * xt_w[:, 1])[:, None]
        sm_w.set_training_values(xt_w, y_w)
        sm_w.train()
        np.testing.assert_allclose(
            sm_w.predict_values(np.array([[0.3, -0.4]])),
            [[1.0 + 2.0 * 0.09 + 0.2]],
            atol=1e-10,
        )

        # 6. Analytical higher moments: skewness() and kurtosis()
        # Exponential(scale=1): Y = X -> skew = 2, excess kurtosis = 6
        sm_exp = PCE(
            distributions=["exponential"],
            degree=2,
            method="quadrature",
            print_global=False,
        )
        xq_x, wq_x = sm_exp.quadrature_points(order=3)
        sm_exp.options["weights"] = wq_x
        sm_exp.set_training_values(xq_x, xq_x)
        sm_exp.train()
        self.assertAlmostEqual(float(sm_exp.skewness()[0]), 2.0, places=10)
        self.assertAlmostEqual(float(sm_exp.kurtosis(fisher=True)[0]), 6.0, places=10)
        self.assertAlmostEqual(float(sm_exp.kurtosis(fisher=False)[0]), 9.0, places=10)

        # Quadratic Normal Y = Z + 0.5 * Z^2 -> var = 1.5, mu3 = 4.0, excess kurt = 20/3
        sm_qn = PCE(
            distributions=["normal"], degree=2, method="quadrature", print_global=False
        )
        xq_n, wq_n = sm_qn.quadrature_points(order=4)
        sm_qn.options["weights"] = wq_n
        sm_qn.set_training_values(xq_n, xq_n + 0.5 * (xq_n**2))
        sm_qn.train()
        mom = sm_qn.moments()
        self.assertAlmostEqual(float(mom["mean"][0]), 0.5, places=11)
        self.assertAlmostEqual(float(mom["variance"][0]), 1.5, places=11)
        self.assertAlmostEqual(float(mom["skewness"][0]), 4.0 / (1.5**1.5), places=10)
        self.assertAlmostEqual(float(mom["kurtosis"][0]), 20.0 / 3.0, places=10)

        # 7. Christoffel WLS ('weights="christoffel"'), weighted constant target, and small empirical moments
        sm_cw = SparsePCE(
            degree=3, method="lars", weights="christoffel", print_global=False
        )
        sm_cw.set_training_values(xt_sp, y_sp)
        sm_cw.train()
        np.testing.assert_allclose(sm_cw.predict_values(xt_test), y_test, atol=1e-8)
        cw = sm_cw.christoffel_weights(xt_test)
        self.assertEqual(cw.shape, (xt_test.shape[0],))
        self.assertTrue(np.all(np.isfinite(cw)) and np.all(cw > 0.0))

        # Constant output under non-uniform weights with sparse solver ('omp')
        sm_wconst = PCE(
            degree=2,
            method="omp",
            xlimits=[[-1.0, 1.0], [-1.0, 1.0]],
            weights=np.linspace(0.5, 2.0, 12),
            print_global=False,
        )
        xt_wc = sm_wconst.sample_design(12, sampling="halton", seed=3, nx=2)
        sm_wconst.set_training_values(xt_wc, np.full((12, 1), 3.25))
        sm_wconst.train()
        np.testing.assert_allclose(
            sm_wconst.predict_values(np.array([[0.2, -0.5]])), [[3.25]], atol=1e-12
        )

        # High-degree empirical marginal with few distinct samples computes moments safely
        small_emp = np.array([0.0, 0.2, 0.5, 0.9, 1.4, 2.1])
        sm_emp_small = PCE(
            distributions=[{"type": "empirical", "data": small_emp}],
            degree=3,
            method="ols",
            print_global=False,
        )
        xt_es = np.linspace(0.0, 2.1, 8).reshape(-1, 1)
        sm_emp_small.set_training_values(xt_es, 1.0 + xt_es + 0.5 * xt_es**2)
        sm_emp_small.train()
        mom_es = sm_emp_small.moments()
        self.assertAlmostEqual(
            float(mom_es["mean"][0]),
            float(np.mean(1.0 + small_emp + 0.5 * small_emp**2)),
            places=10,
        )
        self.assertTrue(np.isfinite(mom_es["kurtosis"][0]))

    def test_validation_edge_cases_and_full_coverage(self):
        # 1. Helper validators in pce.py
        with self.assertRaises(ValueError):
            _normalize_method_name("unknown_solver")
        for bad_deg in ([1.5], "bad", [], [-1]):
            with self.assertRaises(ValueError):
                _normalize_degree_sequence(bad_deg)
        for bad_q in (0, "bad", [], [0.0], [1.5]):
            with self.assertRaises(ValueError):
                _normalize_qnorm_sequence(bad_q)
        self.assertEqual(_normalize_qnorm_sequence(1), [1.0])

        ds2 = DesignSpace([FloatVariable(0.0, 1.0), FloatVariable(0.0, 1.0)])
        self.assertEqual(_infer_nx_for_design(marginals=[None, None]), 2)
        self.assertEqual(_infer_nx_for_design(design_space=ds2), 2)
        with self.assertRaises(ValueError):
            _infer_nx_for_design()

        # 2. Option validation errors in PCE._validate_options
        xt2 = np.array([[0.1, 0.2], [0.8, 0.3], [0.4, 0.9], [0.2, 0.7], [0.9, 0.6]])
        yt2 = (1.0 + xt2[:, :1]).copy()
        invalid_option_cases = [
            {"loo_target": -0.1},
            {"weights": "invalid_mode"},
            {"method": "quadrature", "weights": "christoffel"},
            {"weights": [-1.0, 1.0, 1.0, 1.0, 1.0]},
        ]
        for bad_opts in invalid_option_cases:
            sm_bad = PCE(degree=1, print_global=False, **bad_opts)
            sm_bad.set_training_values(xt2, yt2)
            with self.assertRaises(ValueError):
                sm_bad.train()

        sm_bad_jd = PCE(degree=1, print_global=False)
        sm_bad_jd.set_training_values(xt2, yt2)
        sm_bad_jd.options._dict["jax_dtype"] = "float16"
        with self.assertRaises(ValueError):
            sm_bad_jd.train()
        sm_bad_jd.train_called = True
        sm_bad_jd._is_trained = True
        with self.assertRaises(ValueError):
            sm_bad_jd.get_jax_predictor()

        # 3. Constant (degree=0) PCE model edge cases across moments, Sobol, and derivatives
        sm0 = PCE(degree=0, method="ols", print_global=False)
        sm0.set_training_values(xt2, np.full((5, 2), [3.0, -1.0]))
        sm0.train()
        np.testing.assert_allclose(sm0.mean(), [3.0, -1.0])
        np.testing.assert_allclose(sm0.variance(), [0.0, 0.0])
        np.testing.assert_allclose(sm0.skewness(), [0.0, 0.0])
        np.testing.assert_allclose(sm0.kurtosis(), [0.0, 0.0])
        mom0 = sm0.moments()
        np.testing.assert_allclose(mom0["skewness"], [0.0, 0.0])
        np.testing.assert_allclose(
            sm0.predict_derivatives(xt2[:2], 0), np.zeros((2, 2))
        )
        s_sub0 = sm0.subset_sobol([0])
        np.testing.assert_allclose(s_sub0, [0.0, 0.0])
        self.assertEqual(sm0.subset_sobol([0], output=1), 0.0)
        cond0 = sm0.conditional_expectation([0])
        np.testing.assert_allclose(cond0.predict_values(xt2[:2, :1]), [[3.0, -1.0]] * 2)

        # 4. _central_moments_3_4: max_deg_j == 0 skip AND n_active > 60 QMC fallback
        sm_skip = PCE(degree=2, method="omp", print_global=False)
        # Train cleanly with enough samples so only var 0 is active
        xt_sk = sm_skip.sample_design(25, sampling="sobol", seed=2, nx=2)
        sm_skip.set_training_values(xt_sk, 1.0 + 2.0 * xt_sk[:, :1] + xt_sk[:, :1] ** 2)
        sm_skip.train()
        mom_sk = sm_skip.moments()
        self.assertTrue(np.all(np.isfinite(mom_sk["kurtosis"])))

        # 3D degree=6 has C(9,3)-1 = 83 > 60 non-constant basis terms
        sm_qmc_mom = PCE(degree=6, method="ols", print_global=False)
        xt_qmc = sm_qmc_mom.sample_design(110, sampling="sobol", seed=3, nx=3)
        y_qmc = (1.0 + xt_qmc[:, 0] + 0.5 * xt_qmc[:, 1] ** 2 - 0.3 * xt_qmc[:, 2])[
            :, None
        ]
        sm_qmc_mom.set_training_values(xt_qmc, y_qmc)
        sm_qmc_mom.train()
        self.assertGreater(sm_qmc_mom.multi_indices.shape[0] - 1, 60)
        mom_qmc = sm_qmc_mom.moments(fisher=True)
        self.assertTrue(np.isfinite(mom_qmc["skewness"][0]))
        self.assertTrue(np.isfinite(mom_qmc["kurtosis"][0]))

        # 5. christoffel_weights 1D input and invalid shape
        sm1d = PCE(degree=2, method="ols", print_global=False)
        x1 = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
        sm1d.set_training_values(x1, 1.0 + 2.0 * x1)
        sm1d.train()
        cw1 = sm1d.christoffel_weights(np.array([0.1, 0.5, 0.9]))
        self.assertEqual(cw1.shape, (3,))
        with self.assertRaises(ValueError):
            sm1d.christoffel_weights(np.ones((2, 3)))

        # 6. Error branches on Sobol, subset_sobol, conditional_expectation, quadrature_points, sample_design
        with self.assertRaises(ValueError):
            sm0.sobol_indices(output="0")
        with self.assertRaises(ValueError):
            sm0.subset_sobol(123)
        with self.assertRaises(ValueError):
            sm0.conditional_expectation(["0"])
        with self.assertRaises(ValueError):
            sm0.conditional_expectation([0, 0])
        with self.assertRaises(ValueError):
            sm0.conditional_expectation([5])

        sm_qp = PCE(degree=3, print_global=False)
        xq_def, wq_def = sm_qp.quadrature_points(nx=2)
        self.assertEqual(xq_def.shape, (16, 2))
        self.assertAlmostEqual(float(np.sum(wq_def)), 1.0, places=12)
        with self.assertRaises(ValueError):
            PCE(anisotropy_weights=[-1.0, 1.0], print_global=False).quadrature_points(
                nx=2
            )
        with self.assertRaises(ValueError):
            sm_qp.quadrature_points(order="bad", nx=2)
        with self.assertRaises(ValueError):
            sm_qp.quadrature_points(nx=10)
        with self.assertRaises(ValueError):
            sm_qp.sample_design(0, nx=2)
        with self.assertRaises(ValueError):
            sm_qp.sample_design(4, sampling="invalid", nx=2)

        # 7. MarginalSpec.d_reduced_dx, ppf, gauss_quadrature_1d, recurrence_coefficients, and all dict/tuple specs
        rv_ln = st.lognorm(s=0.4)
        [m_cdf] = parse_distributions(1, distributions=[rv_ln])
        dz = m_cdf.d_reduced_dx(np.array([-1.0, 1.0, 1e100]))
        self.assertEqual(dz[0], 0.0)
        self.assertEqual(dz[2], 0.0)
        self.assertGreater(dz[1], 0.0)
        x_1pt, w_1pt = m_cdf.gauss_quadrature_1d(1)
        self.assertEqual(x_1pt.shape, (1,))
        np.testing.assert_allclose(w_1pt, [1.0])
        with self.assertRaises(ValueError):
            m_cdf.gauss_quadrature_1d(0)

        m_emp_list = parse_distributions(
            1, distributions=[{"type": "empirical", "data": np.linspace(0.0, 2.0, 12)}]
        )
        _ = m_emp_list[0].ppf(np.array([0.25, 0.75]))
        with self.assertRaises(ValueError):
            MarginalSpec(family="stieltjes", offset=0.0, scale=1.0, bounds=(-1, 1)).ppf(
                np.array([0.5])
            )
        with self.assertRaises(ValueError):
            MarginalSpec(
                family="stieltjes", offset=0.0, scale=1.0, bounds=(-1, 1)
            ).recurrence_coefficients(2)
        with self.assertRaises(ValueError):
            MarginalSpec(
                family="unknown", offset=0.0, scale=1.0, bounds=(-1, 1)
            ).recurrence_coefficients(2)

        # Sample design on beta, stieltjes, and empirical marginals (covers ppf branches)
        sm_mix = PCE(
            distributions=[
                ("beta", 2.0, 3.0, 0.0, 1.0),
                ("stieltjes", st.weibull_min(c=2.0)),
                {"type": "empirical", "data": np.linspace(1.0, 5.0, 20)},
            ],
            print_global=False,
        )
        x_mix = sm_mix.sample_design(8, sampling="halton", seed=3)
        self.assertEqual(x_mix.shape, (8, 3))

        # All dict and tuple marginal variants
        valid_specs = [
            m_cdf,
            {"type": "uniform", "bounds": (0.0, 2.0)},
            {"type": "normal", "mean": 1.0, "std": 0.5},
            {"type": "gamma", "shape": 2.0, "rate": 0.5},
            {"type": "exponential", "rate": 2.0},
            {"type": "beta", "alpha": 2.0, "beta": 3.0},
            {"type": "stieltjes", "dist": st.gamma(a=2.0)},
            ("uniform",),
            ("normal",),
            ("gamma", 2.0),
            ("exponential",),
            ("exponential", 1.5),
            ("exponential", 1.5, 0.2),
        ]
        parsed_valid = parse_distributions(len(valid_specs), distributions=valid_specs)
        self.assertEqual(len(parsed_valid), len(valid_specs))

        # Empty / invalid marginal specifications & non-FloatVariable DesignSpace
        invalid_specs = [
            (),
            ("uniform", 1.0),
            ("normal", 1.0),
            ("gamma",),
            ("exponential", 1.0, 2.0, 3.0),
            ("beta", 1.0),
            ("stieltjes",),
            ("empirical",),
            ("unknown_dist", 1.0),
            12345,
            {"type": "normal", "std": -1.0},
            {"type": "gamma", "shape": 2.0, "scale": 1.0, "rate": 1.0},
            {"type": "gamma", "shape": -1.0},
            {"type": "exponential", "scale": 1.0, "rate": 1.0},
            {"type": "exponential", "scale": -1.0},
            {"type": "beta", "alpha": 0.0, "beta": 1.0},
            {"type": "stieltjes"},
            {"type": "stieltjes", "dist": "not_an_rv"},
            {"type": "empirical"},
            {"type": "empirical", "data": [1.0]},
            {"type": "empirical", "data": [2.0, 2.0, 2.0]},
            {"type": "unknown_type"},
            {"type": "uniform", "bounds": (2.0, 1.0)},
            ("empirical", [0.0, 1.0, 2.0]),  # with degree > unique pts in recurrence
        ]
        for bad_spec in invalid_specs[:-1]:
            with self.assertRaises(ValueError):
                parse_distributions(1, distributions=[bad_spec])
        [m_emp3] = parse_distributions(1, distributions=[invalid_specs[-1]])
        with self.assertRaises(ValueError):
            m_emp3.recurrence_coefficients(5)
        with self.assertRaises(ValueError):
            parse_distributions(2, distributions=np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            parse_distributions(1, distributions=[None], xlimits=np.zeros((2, 2)))

        ds_int = DesignSpace([IntegerVariable(0, 3)])
        with self.assertRaises(ValueError):
            parse_distributions(1, design_space=ds_int)

        class _WeirdRV:
            def cdf(self, x):
                return np.zeros_like(x)

            def ppf(self, u):
                return np.full_like(u, np.inf)

        with self.assertRaises(ValueError):
            parse_distributions(1, distributions=[("stieltjes", _WeirdRV())])

        # 8. Backend helpers & solver edge branches
        for bad_args in (
            (0, 1, 1.0, None, None),
            (1, -1, 1.0, None, None),
            (1, 1, 1.5, None, None),
            (1, 1, 1.0, -1, None),
            (2, 1, 1.0, None, [1.0]),
        ):
            with self.assertRaises(ValueError):
                generate_multi_indices(*bad_args)

        with self.assertRaises(OverflowError):
            generate_multi_indices(ndim=4, degree=4, max_interaction=1, max_basis=3)

        np.testing.assert_allclose(
            assemble_basis_deriv_matrix(np.zeros((2, 2), dtype=int), {}, None, 0, 3),
            np.zeros((3, 2)),
        )
        np.testing.assert_allclose(
            assemble_basis_deriv_matrix(
                np.zeros((2, 2), dtype=int), {}, np.ones((3, 1)), 0, 3
            ),
            np.zeros((3, 2)),
        )

        with self.assertRaises(ValueError):
            compute_ols_and_loo(np.zeros((0, 1)), np.zeros(0), 1.0)
        c1, l1, ml1 = compute_ols_and_loo(np.ones((1, 1)), np.array([2.0]), 1.0)
        self.assertTrue(np.isnan(l1) and np.isnan(ml1))
        # High-leverage row where h_i == 1.0
        A_lev = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 5.0]])
        _, l_lev, _ = compute_ols_and_loo(A_lev, np.array([1.0, 2.0, 3.0]), 1.0)
        self.assertTrue(np.isinf(l_lev))
        # Rank-deficient with reg > 0 and reg == 0
        A_rank = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        _, l_r1, _ = compute_ols_and_loo(
            A_rank, np.array([1.0, 2.0, 3.0, 4.0]), 1.0, reg=1e-6
        )
        _, l_r0, _ = compute_ols_and_loo(
            A_rank, np.array([1.0, 2.0, 3.0, 4.0]), 1.0, reg=0.0
        )
        self.assertTrue(np.isinf(l_r1) and np.isinf(l_r0))

        # fit_pce_at_basis edge cases (quadrature defaults & bad weights, unknown solver)
        Phi_test = np.column_stack([np.ones(4), np.array([-1.0, -0.3, 0.3, 1.0])])
        y_const = np.ones((4, 1))
        fit_q0 = fit_pce_at_basis(Phi_test, y_const, method="quadrature", weights=None)
        self.assertEqual(float(fit_q0["loo"][0]), 0.0)
        with self.assertRaises(ValueError):
            fit_pce_at_basis(Phi_test, y_const, method="quadrature", weights=[1.0, 1.0])
        with self.assertRaises(ValueError):
            fit_pce_at_basis(
                Phi_test, y_const, method="quadrature", weights=[0.0, 0.0, 0.0, 0.0]
            )
        with self.assertRaises(ValueError):
            fit_pce_at_basis(Phi_test, y_const, method="ols", weights="bad")
        with self.assertRaises(ValueError):
            fit_pce_at_basis(Phi_test, np.arange(4.0)[:, None], method="unknown_method")

        # Multi-output sparse solvers: 'sp', 'bcs', 'lasso'
        xt_mo_sp = np.linspace(0.0, 1.0, 16)[:, None]
        xt_mo_sp = np.hstack([xt_mo_sp, (xt_mo_sp**2) % 1.0, np.sin(3.0 * xt_mo_sp)])
        yt_mo_sp = np.column_stack(
            [1.0 + 2.0 * xt_mo_sp[:, 0], -0.5 + xt_mo_sp[:, 1] ** 2]
        )
        for meth in ("sp", "bcs", "lasso"):
            sm_m = SparsePCE(degree=2, method=meth, print_global=False)
            sm_m.set_training_values(xt_mo_sp, yt_mo_sp)
            sm_m.train()
            self.assertEqual(sm_m.coefficients.shape[1], 2)

        # select_adaptive_pce: max_basis overflow warning on 2nd degree and ValueError on 1st
        [m_u] = parse_distributions(1, distributions=[("uniform", 0.0, 1.0)])
        with self.assertWarns(UserWarning):
            sel_ok = select_adaptive_pce(
                marginals=[m_u, m_u, m_u],
                xt=np.random.default_rng(0).uniform(0, 1, size=(12, 3)),
                yt=np.ones((12, 1)),
                degrees=[1, 3],
                q_norms=[1.0],
                method="ols",
                early_stop=False,
                max_basis=5,
            )
            self.assertEqual(sel_ok["degree"], 1)
        with self.assertRaises(ValueError):
            select_adaptive_pce(
                marginals=[m_u, m_u, m_u],
                xt=np.random.default_rng(0).uniform(0, 1, size=(12, 3)),
                yt=np.ones((12, 1)),
                degrees=[2],
                q_norms=[1.0],
                method="ols",
                max_basis=4,
            )
        with self.assertRaises(ValueError):
            select_adaptive_pce(
                marginals=[m_u, m_u],
                xt=np.zeros((4, 2)),
                yt=np.ones((4, 1)),
                degrees=[1],
                q_norms=[1.0],
                method="ols",
                anisotropy_weights=[-1.0, 1.0],
            )


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
            dy1_jax = np.asarray(
                jax.vmap(jax.grad(pred_jax))(jnp.asarray(x_eval))[:, 1]
            )
            np.testing.assert_allclose(dy1_jax, dy1_cpu[:, 0], atol=1e-10)

            # Conditional expectation inherits JAX compatibility (including 0D scalars)
            cond = sm.conditional_expectation([0])
            pred_cond_jax = cond.get_jax_predictor()
            x0 = x_eval[:, :1]
            np.testing.assert_allclose(
                np.asarray(pred_cond_jax(jnp.asarray(x0))),
                cond.predict_values(x0),
                atol=1e-11,
            )
            self.assertEqual(pred_cond_jax(float(x0[0, 0])).ndim, 0)
            np.testing.assert_allclose(
                float(jax.grad(pred_cond_jax)(float(x0[0, 0]))),
                cond.predict_derivatives(x0[:1], 0)[0, 0],
                atol=1e-10,
            )
            with self.assertRaises(ValueError):
                pred_jax(0.5)

            # Constant PCE JAX grad returns exact zeros
            sm_c = PCE(degree=0, print_global=False)
            sm_c.set_training_values(xt, np.full((len(xt), 1), 2.5))
            sm_c.train()
            np.testing.assert_allclose(
                np.asarray(jax.grad(sm_c.get_jax_predictor())(pt)),
                np.zeros(3),
                atol=1e-14,
            )

            # Explicit float32 predictor target and jax.dtype handling
            pred_fp32 = sm.get_jax_predictor(dtype="float32")
            y_fp32 = pred_fp32(jnp.asarray(x_eval, dtype=jnp.float32))
            self.assertEqual(y_fp32.dtype, jnp.float32)
            np.testing.assert_allclose(np.asarray(y_fp32), y_cpu, atol=1e-4)

            pred_jnp_dt = sm.get_jax_predictor(dtype=jnp.float64)
            self.assertEqual(pred_jnp_dt(jnp.asarray(x_eval)).dtype, jnp.float64)
            with self.assertRaises(ValueError):
                sm.get_jax_predictor(dtype="bfloat16_invalid")

            # Multi-output JAX evaluation (1D and batch)
            yt_multi = np.hstack([yt, 2.0 * yt])
            sm_m = PCE(degree=3, print_global=False)
            sm_m.set_training_values(xt, yt_multi)
            sm_m.train()
            pred_m = sm_m.get_jax_predictor()
            self.assertEqual(pred_m(pt).shape, (2,))
            np.testing.assert_allclose(
                np.asarray(pred_m(jnp.asarray(x_eval))),
                sm_m.predict_values(x_eval),
                atol=1e-11,
            )

            # Discretized Stieltjes marginal also compiles to pure JAX predictor
            sm_sj = PCE(
                distributions=[("stieltjes", st.lognorm(s=0.4))],
                degree=2,
                method="quadrature",
                print_global=False,
            )
            xq_s, wq_s = sm_sj.quadrature_points(order=5)
            sm_sj.options["weights"] = wq_s
            sm_sj.set_training_values(xq_s, 1.2 + 0.7 * (xq_s**2))
            sm_sj.train()
            pred_sj = sm_sj.get_jax_predictor()
            x_sj = np.array([[0.7], [1.4], [2.1]])
            np.testing.assert_allclose(
                np.asarray(pred_sj(jnp.asarray(x_sj))),
                sm_sj.predict_values(x_sj),
                atol=1e-10,
            )
            np.testing.assert_allclose(
                float(jax.grad(pred_sj)(1.4)),
                2.0 * 0.7 * 1.4,
                atol=1e-10,
            )
        self.assertEqual(bool(jax.config.jax_enable_x64), prev_x64)

        # Untrained models reject get_jax_predictor()
        for model_cls in (PCE, SparsePCE):
            with self.assertRaises(RuntimeError):
                model_cls(print_global=False).get_jax_predictor()

        # Pickle roundtrip works cleanly
        loaded = pickle.loads(pickle.dumps(sm))
        np.testing.assert_allclose(loaded.predict_values(x_eval), y_cpu, atol=1e-11)

        # Pristine subprocess check: get_jax_predictor does not mutate x64 or warn
        code = textwrap.dedent(
            """
            import warnings
            import jax
            import jax.numpy as jnp
            import numpy as np
            from smt.surrogate_models import PCE

            assert not jax.config.jax_enable_x64
            sm = PCE(degree=2, print_global=False)
            xt = np.array([[0.1, 0.2], [0.8, 0.3], [0.4, 0.9], [0.2, 0.7], [0.9, 0.8], [0.5, 0.5]])
            yt = 1.0 + 2.0 * xt[:, :1] - 0.5 * (xt[:, 1:2] ** 2)
            sm.set_training_values(xt, yt)
            sm.train()
            x_eval = np.array([[0.3, 0.6], [0.7, 0.4]])
            y_cpu = sm.predict_values(x_eval)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pred = sm.get_jax_predictor()
                y_jax = jax.jit(pred)(jnp.asarray(x_eval))
                assert not any("float64" in str(item.message) for item in w)
            np.testing.assert_allclose(np.asarray(y_jax), y_cpu, atol=1e-5)
            assert not jax.config.jax_enable_x64
            """
        )
        subprocess.check_call([sys.executable, "-c", code])


if __name__ == "__main__":
    unittest.main()
