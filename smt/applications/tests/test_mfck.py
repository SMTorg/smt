# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:17:04 2024

@author: mcastano

Unit tests for the MFCK application, covering the original behaviour plus:

  * the parameter-vector layout helpers and the log10 <-> model space mapping,
  * the analytical gradients of the marginal likelihood (finite differences,
    both parameter spaces, several kernels and numbers of fidelity levels),
  * the gradient-based optimisers and their guards,
  * the sequential (level-wise) hyper-parameter estimation and the
    level-restriction context manager,
  * the optimisation report and the multi-start machinery,
  * the heteroscedastic-noise branch: scaling, auxiliary noise model,
    noise prediction and target transformation,
  * the prediction API (shapes, variance scaling, single fidelity level).
"""

import unittest

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    NO_MATPLOTLIB = False
except ImportError:
    NO_MATPLOTLIB = True

try:
    import nlopt  # noqa: F401

    NO_NLOPT = False
except ImportError:
    NO_NLOPT = True

from copy import deepcopy

from smt.applications.mfck import MFCK
from smt.problems import TensorProduct
from smt.sampling_methods import FullFactorial
from smt.utils.silence import Silence
from smt.utils.sm_test_case import SMTestCase

print_output = False

# scipy checks the TNC / COBYLA evaluation limit *after* the call, so the
# recorded number of likelihood evaluations can exceed the requested budget by
# a few units depending on the scipy version and the platform.
OPT_BUDGET_SLACK = 5


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def forrester_hf(x):
    return ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)


def forrester_lf(x):
    return 0.5 * forrester_hf(x) + (x - 0.5) * 10.0 - 5


def toy_levels(x, level):
    """Smooth, mildly correlated levels for the multi-level tests."""
    total = np.sum(x, axis=1, keepdims=True)
    return (
        np.sin(3.0 * total) + 0.3 * level * total + 0.1 * level * np.cos(7.0 * x[:, :1])
    )


def build_untrained_model(
    n_levels=2,
    dim=1,
    n_points=(24, 14, 9),
    corr="squar_exp",
    eval_noise=True,
    nested=False,
    seed=3,
    **options,
):
    """
    Sets a model up exactly as train() does, but without the optimisation, so
    that the likelihood and its gradient can be probed at chosen points.
    """
    from smt.utils.misc import standardization

    rng = np.random.default_rng(seed)
    x_low = rng.uniform(0.0, 1.0, (n_points[0], dim))
    xt, yt = [], []
    for level in range(n_levels):
        x = (
            x_low[: n_points[level]]
            if nested
            else rng.uniform(0.0, 1.0, (n_points[level], dim))
        )
        xt.append(x)
        yt.append(toy_levels(x, level) + 0.01 * rng.standard_normal((x.shape[0], 1)))

    options = {k: v for k, v in options.items() if v is not None}
    model = MFCK(
        theta0=[0.7] * dim,
        corr=corr,
        eval_noise=eval_noise,
        noise0=[1e-3],
        print_global=False,
        **options,
    )
    for level in range(n_levels - 1):
        model.set_training_values(xt[level], yt[level], name=level)
    model.set_training_values(xt[-1], yt[-1])

    model.lvl = n_levels
    model.X = xt
    model.y = np.vstack(yt)
    model._check_param_het_safe()
    (_, _, model.X_offset, model.y_mean, model.X_scale, model.y_std) = standardization(
        np.concatenate(xt, axis=0), np.concatenate(yt, axis=0)
    )
    model.X_norma_all = [(x - model.X_offset) / model.X_scale for x in xt]
    model.y_norma_all = np.vstack([(f - model.y_mean) / model.y_std for f in yt])
    return model


def well_conditioned_point(model, rng):
    """
    A point of the optimiser space in the region where the covariance matrix is
    reasonably conditioned; sampling uniformly over the full bounds produces
    likelihoods whose finite differences are meaningless.
    """
    lower, upper = model._build_hyperparameter_vectors()[1:]
    x = np.empty_like(lower)
    for level in range(model.lvl):
        i_sigma, sl_theta, i_rho = model._param_indices(level)
        x[i_sigma] = rng.uniform(-0.5, 0.5)
        x[sl_theta] = rng.uniform(-1.0, 0.5, model.nx)
        if i_rho is not None:
            x[i_rho] = rng.uniform(-2.0, 2.0)
    if model._has_noise_params():
        x[-model.lvl :] = rng.uniform(-4.0, -1.0)
    return np.clip(x, lower, upper)


def richardson_gradient(func, x, eps=1e-4):
    """Central differences with Richardson extrapolation, O(h^4)."""
    grad = np.zeros_like(x)
    for i in range(x.size):
        h = eps * max(abs(x[i]), 1e-2)

        def central(step):
            xp, xm = np.array(x), np.array(x)
            xp[i] += step
            xm[i] -= step
            return (func(xp) - func(xm)) / (2.0 * step)

        grad[i] = (4.0 * central(h / 2.0) - central(h)) / 3.0
    return grad


def max_relative_error(analytic, numeric):
    scale = np.maximum(np.abs(analytic), np.abs(numeric))
    scale = np.where(scale < 1e-8, 1.0, scale)
    return float(np.max(np.abs(analytic - numeric) / scale))


def train_forrester(n_hf=8, n_lf=16, seed=1380, **options):
    """A small trained two-fidelity model, used by many tests."""
    rng = np.random.default_rng(seed)
    x_hf = np.sort(rng.uniform(0.0, 1.0, (n_hf, 1)), axis=0)
    x_lf = np.sort(rng.uniform(0.0, 1.0, (n_lf, 1)), axis=0)
    options.setdefault("theta0", [1.0])
    options.setdefault("n_start", 1)
    options.setdefault("print_global", False)
    model = MFCK(**options)
    model.set_training_values(x_lf, forrester_lf(x_lf), name=0)
    model.set_training_values(x_hf, forrester_hf(x_hf))
    with Silence():
        model.train()
    return model


def train_het_noise(n_hf=12, n_lf=20, seed=5, **options):
    """A trained model with heteroscedastic noise and the auxiliary noise model."""
    rng = np.random.default_rng(seed)
    x_hf = np.sort(rng.uniform(0.0, 1.0, (n_hf, 1)), axis=0)
    x_lf = np.sort(rng.uniform(0.0, 1.0, (n_lf, 1)), axis=0)
    tau2_hf = 0.01 + 0.2 * np.sin(3.0 * x_hf.ravel()) ** 2
    tau2_lf = 0.02 + 0.3 * np.sin(3.0 * x_lf.ravel()) ** 2
    y_hf = forrester_hf(x_hf) + np.sqrt(tau2_hf)[:, None] * rng.standard_normal(
        x_hf.shape
    )
    y_lf = forrester_lf(x_lf) + np.sqrt(tau2_lf)[:, None] * rng.standard_normal(
        x_lf.shape
    )
    options.setdefault("theta0", [1.0])
    options.setdefault("n_start", 1)
    options.setdefault("print_global", False)
    model = MFCK(use_het_noise=True, noise0=[tau2_lf, tau2_hf], **options)
    model.set_training_values(x_lf, y_lf, name=0)
    model.set_training_values(x_hf, y_hf)
    with Silence():
        model.train()
    return model, dict(x_lf=x_lf, x_hf=x_hf, tau2_lf=tau2_lf, tau2_hf=tau2_hf)


class TestMFCK(SMTestCase):
    def setUp(self):
        self.nt = 100
        self.ne = 100
        self.ndim = 3

    # ------------------------------------------------------------------
    # original tests
    # ------------------------------------------------------------------
    def test_mfck(self):
        self.problems = ["exp"]  # , "tanh", "cos"]

        for fname in self.problems:
            prob = TensorProduct(ndim=self.ndim, func=fname)
            sampling = FullFactorial(xlimits=prob.xlimits, clip=True)

            xt = sampling(self.nt)
            yt = prob(xt)
            for i in range(self.ndim):
                yt = np.concatenate((yt, prob(xt, kx=i)), axis=1)

            y_lf = 2 * prob(xt) + 2
            x_lf = deepcopy(xt)

            sm = MFCK(hyper_opt="Cobyla", eval_noise=False)
            if sm.options.is_declared("xlimits"):
                sm.options["xlimits"] = prob.xlimits
            sm.options["print_global"] = False

            sm.set_training_values(xt, yt[:, 0])
            sm.set_training_values(x_lf, y_lf[:, 0], name=0)

            with Silence():
                sm.train()

            m = sm.predict_values(xt)

            num = np.linalg.norm(m[:, 0] - yt[:, 0])
            den = np.linalg.norm(yt[:, 0])

            t_error = num / den

            self.assert_error(t_error, 0.0, 3e-5, 1e-5)

    def test_mfck_error_branches(self):
        """Covers ValueError in eta."""
        sm = MFCK()
        with self.assertRaises(ValueError):
            sm.eta(j=5, jp=2, rho=[1.0, 1.0])

    # ------------------------------------------------------------------
    # parameter-vector layout
    # ------------------------------------------------------------------
    def test_parameter_layout(self):
        """Block layout: [sigma_0 l_0 | sigma_k l_k rho_k | tau_0^2 ...]."""
        for dim in (1, 3):
            for n_levels in (1, 2, 3):
                sm = build_untrained_model(n_levels=n_levels, dim=dim)
                n_kernel = sm._n_kernel_params()
                self.assertEqual(n_kernel, (dim + 1) + (n_levels - 1) * (dim + 2))

                theta, lower, upper = sm._build_hyperparameter_vectors()
                expected = n_kernel + n_levels  # eval_noise=True
                self.assertEqual(theta.size, expected)
                self.assertEqual(lower.size, expected)
                self.assertEqual(upper.size, expected)
                self.assertTrue(np.all(lower <= theta))
                self.assertTrue(np.all(theta <= upper))

                # blocks tile the kernel vector exactly, without overlap
                covered = np.zeros(n_kernel, dtype=int)
                for level in range(n_levels):
                    block = sm._level_block_slice(level)
                    covered[block] += 1
                    i_sigma, sl_theta, i_rho = sm._param_indices(level)
                    self.assertEqual(i_sigma, block.start)
                    self.assertEqual(sl_theta.stop - sl_theta.start, dim)
                    if level == 0:
                        self.assertIsNone(i_rho)
                    else:
                        self.assertEqual(i_rho, block.stop - 1)
                self.assertTrue(np.all(covered == 1))

                # rho is the only parameter kept in linear scale
                mask = sm._log_scale_mask()
                self.assertEqual(mask.size, expected)
                self.assertEqual(int(np.sum(~mask)), n_levels - 1)
                for level in range(1, n_levels):
                    self.assertFalse(mask[sm._param_indices(level)[2]])

    def test_transform_optimizer_param(self):
        """log10 -> model space, rho untouched, noise transformed."""
        sm = build_untrained_model(n_levels=3, dim=2)
        rng = np.random.default_rng(0)
        x = well_conditioned_point(sm, rng)
        param = sm._transform_optimizer_param(x)

        self.assertEqual(param.size, x.size)
        mask = sm._log_scale_mask()
        np.testing.assert_allclose(param[mask], 10.0 ** x[mask], rtol=1e-12)
        np.testing.assert_allclose(param[~mask], x[~mask], rtol=1e-12)
        self.assertTrue(np.all(param[mask] > 0.0))
        # the input vector must not be modified in place
        self.assertFalse(np.allclose(param, x))

    def test_eta_matrix(self):
        """_eta_matrix / _eta_grad_matrix against eta() and finite differences."""
        sm = build_untrained_model(n_levels=3, dim=1)
        rho = np.array([1.3, -0.7])
        eta = sm._eta_matrix(rho)
        for j in range(sm.lvl):
            for kappa in range(j, sm.lvl):
                self.assertAlmostEqual(eta[j, kappa], sm.eta(j, kappa, rho), places=12)
        self.assertTrue(np.allclose(np.tril(eta, -1), 0.0))  # eta = 0 for j > kappa

        for m in range(sm.lvl - 1):
            analytic = sm._eta_grad_matrix(rho, m)
            step = 1e-6
            rho_p, rho_m = rho.copy(), rho.copy()
            rho_p[m] += step
            rho_m[m] -= step
            numeric = (sm._eta_matrix(rho_p) - sm._eta_matrix(rho_m)) / (2 * step)
            np.testing.assert_allclose(analytic, numeric, atol=1e-7)

        # rho = 0 must not break the derivative (no division by rho)
        analytic = sm._eta_grad_matrix(np.array([0.0, 0.0]), 0)
        self.assertTrue(np.all(np.isfinite(analytic)))
        self.assertAlmostEqual(analytic[0, 1], 1.0, places=12)

    # ------------------------------------------------------------------
    # gradients
    # ------------------------------------------------------------------
    def test_gradient_value_matches_likelihood(self):
        """The vectorised assembly and the block-wise one agree."""
        rng = np.random.default_rng(1)
        for n_levels in (1, 2, 3):
            for eval_noise in (True, False):
                sm = build_untrained_model(
                    n_levels=n_levels,
                    dim=2,
                    eval_noise=eval_noise,
                    nugget=None if eval_noise else 1e-6,
                )
                for _ in range(2):
                    param = sm._transform_optimizer_param(
                        well_conditioned_point(sm, rng)
                    )
                    reference = sm.neg_log_likelihood(param)
                    value = sm.neg_log_likelihood_grad(param)[0]
                    self.assertAlmostEqual(
                        value / max(1.0, abs(reference)),
                        reference / max(1.0, abs(reference)),
                        places=8,
                    )

    def test_gradient_finite_differences(self):
        """Analytical gradient against Richardson central differences."""
        rng = np.random.default_rng(42)
        cases = [
            dict(n_levels=1, dim=1),
            dict(n_levels=2, dim=1),
            dict(n_levels=2, dim=3),
            dict(n_levels=2, dim=2, nested=True),
            dict(n_levels=3, dim=2),
        ]
        for case in cases:
            sm = build_untrained_model(**case)
            x = well_conditioned_point(sm, rng)

            # model space
            param = sm._transform_optimizer_param(x)
            _, grad = sm.neg_log_likelihood_grad(param)
            numeric = richardson_gradient(sm.neg_log_likelihood, param)
            self.assertLess(max_relative_error(grad, numeric), 1e-5, msg=str(case))

            # optimiser space (includes the log10 chain rule)
            _, grad_log = sm.neg_log_likelihood_scipy_grad(x)
            numeric_log = richardson_gradient(sm.neg_log_likelihood_scipy, x)
            self.assertLess(
                max_relative_error(grad_log, numeric_log), 1e-5, msg=str(case)
            )

    def test_gradient_kernels(self):
        """Every kernel exposing dR/dtheta is differentiated correctly."""
        rng = np.random.default_rng(7)
        for corr in ("squar_exp", "abs_exp", "matern32", "matern52"):
            sm = build_untrained_model(n_levels=2, dim=2, corr=corr)
            x = well_conditioned_point(sm, rng)
            _, grad = sm.neg_log_likelihood_scipy_grad(x)
            numeric = richardson_gradient(sm.neg_log_likelihood_scipy, x)
            self.assertLess(max_relative_error(grad, numeric), 1e-5, msg=corr)

    def test_gradient_with_regularization(self):
        """The lambda term contributes 2 lambda theta to the gradient."""
        rng = np.random.default_rng(11)
        sm = build_untrained_model(n_levels=2, dim=1)
        sm.options["lambda"] = 1e-2
        x = well_conditioned_point(sm, rng)
        param = sm._transform_optimizer_param(x)
        _, grad = sm.neg_log_likelihood_grad(param)
        numeric = richardson_gradient(sm.neg_log_likelihood, param)
        self.assertLess(max_relative_error(grad, numeric), 1e-5)

    def test_gradient_guards(self):
        """Unsupported kernel or model rejects the gradient-based optimisers."""
        sm = build_untrained_model(n_levels=2, dim=1, corr="squar_exp")
        sm.options["hyper_opt"] = "TNC"
        sm._check_gradient_support()  # supported: no exception

        sm.options["corr"] = "squar_sin_exp"
        with self.assertRaises(ValueError):
            sm._check_gradient_support()

        sm.options["corr"] = "squar_exp"
        sm._supports_gradient = False
        with self.assertRaises(ValueError):
            sm._check_gradient_support()
        with self.assertRaises(NotImplementedError):
            sm.neg_log_likelihood_grad(np.ones(sm._n_kernel_params() + sm.lvl))

    def test_gradient_based_optimizers(self):
        """TNC and L-BFGS train and reach a likelihood at least as good."""
        reference = train_forrester(
            hyper_opt="Cobyla", eval_noise=True, noise0=[1e-3], opt_max_eval=200
        )
        optimizers = ["TNC"] if NO_NLOPT else ["TNC", "Lbfgs-nlopt"]
        for hyper_opt in optimizers:
            sm = train_forrester(
                hyper_opt=hyper_opt,
                eval_noise=True,
                noise0=[1e-3],
                opt_max_eval=200,
            )
            self.assertTrue(np.all(np.isfinite(sm.optimal_theta)))
            reference_nll = reference.opt_report["joint_nll"]
            self.assertLessEqual(
                sm.opt_report["joint_nll"],
                reference_nll + 1e-6 * max(1.0, abs(reference_nll)),
                msg=f"{hyper_opt} did worse than COBYLA",
            )
            # the budget is honoured; how far below it a gradient run stops is
            # optimiser- and platform-dependent, so it is not asserted here
            self.assertLessEqual(sm.opt_report["n_eval"], 200 + OPT_BUDGET_SLACK)

    # ------------------------------------------------------------------
    # sequential estimation
    # ------------------------------------------------------------------
    def test_restricted_levels(self):
        """The context manager exposes a sub-model and restores everything."""
        sm = build_untrained_model(n_levels=3, dim=1)
        saved = dict(
            lvl=sm.lvl,
            n_x=len(sm.X_norma_all),
            n_y=sm.y_norma_all.shape[0],
            X=[x.copy() for x in sm.X_norma_all],
        )
        n_rows = sum(x.shape[0] for x in sm.X_norma_all[:2])

        with sm._restricted_levels(2):
            self.assertEqual(sm.lvl, 2)
            self.assertEqual(len(sm.X_norma_all), 2)
            self.assertEqual(sm.y_norma_all.shape[0], n_rows)
            n_kernel_sub = sm._n_kernel_params(2)
            param = np.concatenate([np.full(n_kernel_sub, 0.0), np.full(2, -3.0)])
            value = sm.neg_log_likelihood_scipy(param)
            self.assertTrue(np.isfinite(value))
            self.assertEqual(sm.K.shape, (n_rows, n_rows))

        self.assertEqual(sm.lvl, saved["lvl"])
        self.assertEqual(len(sm.X_norma_all), saved["n_x"])
        self.assertEqual(sm.y_norma_all.shape[0], saved["n_y"])
        for before, after in zip(saved["X"], sm.X_norma_all):
            np.testing.assert_allclose(before, after)

        # a no-op restriction is allowed
        with sm._restricted_levels(sm.lvl):
            self.assertEqual(sm.lvl, saved["lvl"])

    def test_stage_objective_gradient(self):
        """A stage objective returns the gradient of its free block only."""
        sm = build_untrained_model(n_levels=3, dim=2)
        rng = np.random.default_rng(3)
        x_full = well_conditioned_point(sm, rng)
        n_kernel = sm._n_kernel_params(sm.lvl)
        for level in range(sm.lvl):
            block = sm._level_block_slice(level)
            free = np.asarray(
                list(range(block.start, block.stop)) + [n_kernel + level], dtype=int
            )
            with sm._restricted_levels(level + 1):
                objective = sm._make_stage_objective(x_full, level)
                x_free = x_full[free]
                value, grad = objective(x_free, with_grad=True)
                self.assertEqual(grad.size, x_free.size)
                self.assertAlmostEqual(value, objective(x_free), places=10)
                numeric = richardson_gradient(lambda z: objective(z), x_free)
            self.assertLess(max_relative_error(grad, numeric), 1e-5)

    def test_sequential_training(self):
        """sequential_opt trains level by level and reports the stages."""
        joint = train_forrester(
            sequential_opt=False, eval_noise=True, noise0=[1e-3], opt_max_eval=60
        )
        sequential = train_forrester(
            sequential_opt=True, eval_noise=True, noise0=[1e-3], opt_max_eval=60
        )

        self.assertEqual(joint.opt_report["mode"], "joint")
        self.assertEqual(sequential.opt_report["mode"], "sequential")
        stages = sequential.opt_report["stages"]
        self.assertEqual([s["stage"] for s in stages], ["level0", "level1"])
        # level 0: (sigma, theta, tau^2); level 1 adds rho
        self.assertEqual(stages[0]["n_free_params"], 3)
        self.assertEqual(stages[1]["n_free_params"], 4)
        self.assertEqual(
            sequential.opt_report["n_eval"], sum(s["n_eval"] for s in stages)
        )
        for report in (joint.opt_report, sequential.opt_report):
            self.assertTrue(np.isfinite(report["joint_nll"]))
            self.assertGreater(len(report["history"]), 0)
            self.assertGreater(report["training_time"], 0.0)

        self.assertEqual(sequential.optimal_theta.size, joint.optimal_theta.size)
        x = np.linspace(0, 1, 20).reshape(-1, 1)
        self.assertTrue(np.all(np.isfinite(sequential.predict_values(x))))

    def test_sequential_refine(self):
        """sequential_refine appends a joint polishing stage."""
        sm = train_forrester(
            sequential_opt=True,
            sequential_refine=True,
            eval_noise=True,
            noise0=[1e-3],
            opt_max_eval=60,
        )
        stages = sm.opt_report["stages"]
        self.assertEqual(stages[-1]["stage"], "refine")
        self.assertEqual(stages[-1]["n_free_params"], sm.optimal_theta.size)
        # the refinement cannot leave the likelihood worse than the cascade
        self.assertLessEqual(stages[-1]["nll"], stages[-2]["nll"] + 1e-6)

    def test_sequential_single_level(self):
        """With one fidelity level the sequential flag falls back to joint."""
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0.0, 1.0, (15, 1)), axis=0)
        sm = MFCK(
            theta0=[1.0],
            n_start=1,
            sequential_opt=True,
            eval_noise=True,
            noise0=[1e-3],
            print_global=False,
            opt_max_eval=40,
        )
        sm.set_training_values(x, forrester_hf(x))
        with Silence():
            sm.train()
        self.assertEqual(sm.lvl, 1)
        self.assertEqual(sm.opt_report["mode"], "joint")

    # ------------------------------------------------------------------
    # optimisation machinery
    # ------------------------------------------------------------------
    def test_multistart_sampling(self):
        """_sample_starts never asks LHS for a single point (it would hang)."""
        sm = build_untrained_model(n_levels=2, dim=1)
        _, lower, upper = sm._build_hyperparameter_vectors()
        self.assertEqual(sm._sample_starts(0, lower, upper), [])
        for n_samples in (1, 3):
            starts = sm._sample_starts(n_samples, lower, upper)
            self.assertEqual(len(starts), n_samples)
            for start in starts:
                self.assertTrue(np.all(start >= lower))
                self.assertTrue(np.all(start <= upper))

    def test_n_start_one_terminates(self):
        """Regression: n_start=1 used to build a degenerate LHS and hang."""
        sm = train_forrester(n_start=1, opt_max_eval=30, eval_noise=True, noise0=[1e-3])
        self.assertTrue(np.all(np.isfinite(sm.optimal_theta)))

    def test_opt_max_eval_is_respected(self):
        """The evaluation budget bounds the number of likelihood calls."""
        for budget in (20, 40):
            sm = train_forrester(
                n_start=1, opt_max_eval=budget, eval_noise=True, noise0=[1e-3]
            )
            self.assertLessEqual(sm.opt_report["n_eval"], budget + OPT_BUDGET_SLACK)
            self.assertEqual(len(sm.opt_report["history"]), sm.opt_report["n_eval"])

    def test_safe_call_on_failure(self):
        """A failing likelihood is penalised instead of crashing the training."""
        sm = build_untrained_model(n_levels=2, dim=1)

        def broken(_):
            raise np.linalg.LinAlgError("not positive definite")

        self.assertEqual(sm._safe_call(broken, np.zeros(3)), sm._FAILED_NLL)
        self.assertEqual(sm._safe_call(lambda _: np.nan, np.zeros(3)), sm._FAILED_NLL)
        value, grad = sm._safe_call_grad(
            lambda _x, with_grad=False: (np.inf, np.zeros(3)), np.zeros(3)
        )
        self.assertEqual(value, sm._FAILED_NLL)
        self.assertEqual(grad.size, 3)

    def test_noise0_non_positive_warns(self):
        """noise0=0 would give log10(0) = -inf; it is replaced with a warning."""
        sm = build_untrained_model(n_levels=2, dim=1)
        sm.options["noise0"] = [0.0]
        with self.assertWarns(UserWarning):
            theta, lower, _ = sm._build_hyperparameter_vectors()
        self.assertTrue(np.all(np.isfinite(theta)))
        self.assertTrue(np.all(theta >= lower))

    def test_unknown_optimizer_raises(self):
        sm = build_untrained_model(n_levels=2, dim=1)
        sm.options._declared_entries["hyper_opt"]["values"] = None
        sm.options["hyper_opt"] = "not-an-optimizer"
        _, lower, upper = sm._build_hyperparameter_vectors()
        with self.assertRaises(ValueError):
            sm._single_run(lambda x: 0.0, lower, lower, upper, tag="test")

    # ------------------------------------------------------------------
    # predictions
    # ------------------------------------------------------------------
    def test_prediction_api(self):
        """Shapes and consistency of the prediction methods."""
        sm = train_forrester(eval_noise=True, noise0=[1e-3], opt_max_eval=60)
        x = np.linspace(0.0, 1.0, 17).reshape(-1, 1)

        mean = sm.predict_values(x)
        self.assertEqual(mean.shape, (17, 1))

        variance = np.asarray(sm.predict_variances(x)).ravel()
        all_levels = sm.predict_variances_all_levels(x)
        self.assertEqual(all_levels.shape, (17, sm.lvl))
        # predict_variances must use the same scaling as the per-level version
        np.testing.assert_allclose(variance, all_levels[:, -1], rtol=1e-10)
        self.assertTrue(np.all(all_levels > 0.0))

        means, covariances = sm.predict_all_levels(x)
        self.assertEqual(len(means), sm.lvl)
        self.assertEqual(len(covariances), sm.lvl)
        np.testing.assert_allclose(
            np.asarray(means[-1]).ravel(), mean.ravel(), rtol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(covariances[-1]).ravel(), all_levels[:, -1], rtol=1e-10
        )

    def test_single_fidelity_consistency(self):
        """One level: training and prediction use the same normalisation."""
        rng = np.random.default_rng(2)
        x = np.sort(rng.uniform(0.0, 1.0, (20, 1)), axis=0)
        y = forrester_hf(x)
        sm = MFCK(
            theta0=[1.0],
            n_start=1,
            eval_noise=False,
            print_global=False,
            opt_max_eval=80,
        )
        sm.set_training_values(x, y)
        with Silence():
            sm.train()
        prediction = sm.predict_values(x)
        error = np.linalg.norm(prediction.ravel() - y.ravel()) / np.linalg.norm(y)
        self.assertLess(error, 1e-2)

    def test_kernel_exponent(self):
        """The correlation exponent follows the kernel, not pow_exp_power."""
        expected = {"squar_exp": 2.0, "abs_exp": 1.0, "matern32": 1.0, "matern52": 1.0}
        for corr, power in expected.items():
            sm = build_untrained_model(n_levels=2, dim=1, corr=corr)
            self.assertEqual(sm._pow_exp_power, power)
        # matern kernels used to produce non-SPD matrices with power = 1.9
        sm = train_forrester(
            corr="matern32", eval_noise=True, noise0=[1e-3], opt_max_eval=40
        )
        self.assertTrue(
            np.all(np.isfinite(sm.predict_values(np.linspace(0, 1, 5).reshape(-1, 1))))
        )

    # ------------------------------------------------------------------
    # heteroscedastic noise
    # ------------------------------------------------------------------
    def test_het_noise_vector_scaling(self):
        """noise0 is given in output units and normalised by y_std**2."""
        sm, data = train_het_noise(opt_max_eval=40)
        expected = np.concatenate([data["tau2_lf"], data["tau2_hf"]])
        expected = expected / float(np.asarray(sm.y_std).ravel()[0]) ** 2
        np.testing.assert_allclose(sm._het_noise_vector(), expected, rtol=1e-12)
        # with heteroscedastic noise there are no noise hyper-parameters
        self.assertFalse(sm._has_noise_params())
        self.assertEqual(sm.optimal_theta.size, sm._n_kernel_params())

    def test_het_noise_auxiliary_model(self):
        """The observed variances train a second model of the same class."""
        sm, data = train_het_noise(predict_with_noise=True, opt_max_eval=40)
        auxiliary = sm.noise_model
        self.assertIsInstance(auxiliary, MFCK)
        self.assertEqual(auxiliary.lvl, sm.lvl)
        for x_aux, x_main in zip(auxiliary.X, sm.X):
            np.testing.assert_allclose(x_aux, x_main)
        # the auxiliary model must not fit a noise model of its own
        self.assertFalse(auxiliary.options["predict_with_noise"])
        self.assertIsNone(auxiliary.noise_model)

    def test_predict_noise(self):
        """The noise field is available anywhere and for every level."""
        sm, data = train_het_noise(predict_with_noise=True, opt_max_eval=40)
        x = np.linspace(-0.5, 1.5, 23).reshape(-1, 1)

        fields = sm.predict_noise_all_levels(x)
        self.assertEqual(len(fields), sm.lvl)
        for field in fields:
            self.assertEqual(field.shape, (23, 1))
            self.assertTrue(np.all(np.isfinite(field)))
            self.assertTrue(np.all(field > 0.0))

        hf = sm.predict_noise(x)
        self.assertEqual(hf.shape, (23,))
        np.testing.assert_allclose(hf, fields[-1].ravel(), rtol=1e-12)
        np.testing.assert_allclose(
            sm.predict_noise(x, level=0), fields[0].ravel(), rtol=1e-12
        )

    def test_predict_with_noise_adds_the_field(self):
        """predict_with_noise adds the predicted noise to every variance."""
        x = np.linspace(0.0, 1.0, 15).reshape(-1, 1)
        with_noise, _ = train_het_noise(predict_with_noise=True, opt_max_eval=40)
        without, _ = train_het_noise(predict_with_noise=False, opt_max_eval=40)

        _, var_with = with_noise.predict_all_levels(x)
        _, var_without = without.predict_all_levels(x)
        fields = with_noise.predict_noise_all_levels(x)
        for level in range(with_noise.lvl):
            delta = (
                np.asarray(var_with[level]).ravel()
                - np.asarray(var_without[level]).ravel()
            )
            np.testing.assert_allclose(
                delta, np.asarray(fields[level]).ravel(), rtol=1e-8, atol=1e-12
            )
        np.testing.assert_allclose(
            np.asarray(with_noise.predict_variances(x)).ravel(),
            np.asarray(var_with[-1]).ravel(),
            rtol=1e-10,
        )
        self.assertIsNone(without.noise_model)

    def test_noise_target_transform_log(self):
        """The log transform keeps the predicted variance strictly positive."""
        sm, _ = train_het_noise(
            predict_with_noise=True, noise_target_transform="log", opt_max_eval=40
        )
        x = np.linspace(-1.0, 2.0, 41).reshape(-1, 1)
        field = sm.predict_noise(x)
        self.assertTrue(np.all(field > 0.0))
        self.assertTrue(np.all(np.isfinite(field)))
        # the auxiliary model of a log fit cannot reuse noise0 as its own noise
        self.assertFalse(sm.noise_model.options["use_het_noise"])

    def test_het_noise_without_eval_noise(self):
        """use_het_noise no longer conflicts with eval_noise=False."""
        sm, _ = train_het_noise(
            eval_noise=False, predict_with_noise=True, opt_max_eval=40
        )
        self.assertFalse(sm._has_noise_params())
        x = np.linspace(0.0, 1.0, 7).reshape(-1, 1)
        self.assertTrue(np.all(np.isfinite(sm.predict_values(x))))
        self.assertTrue(np.all(sm.predict_noise(x) > 0.0))

    def test_noise_model_error_branches(self):
        """Validation of the heteroscedastic inputs."""
        sm = build_untrained_model(n_levels=2, dim=1)
        with self.assertRaises(ValueError):  # use_het_noise is False
            sm._fit_noise_model()

        sm.options["use_het_noise"] = True
        sm.options["noise0"] = [np.ones(3)]  # one array instead of two
        with self.assertRaises(ValueError):
            sm._check_noise_targets()

        sm.options["noise0"] = [np.ones(3), np.ones(sm.X[1].shape[0])]
        with self.assertRaises(ValueError):  # wrong number of points at level 0
            sm._check_noise_targets()

    def test_het_noise_enters_the_likelihood(self):
        """The heteroscedastic diagonal is used even when eval_noise=False."""
        sm, data = train_het_noise(eval_noise=False, opt_max_eval=30)
        _, chol, noises = sm._compute_K_and_cholesky()
        self.assertIsNone(noises)  # no noise hyper-parameter in that mode
        n_total = sum(x.shape[0] for x in sm.X)
        self.assertEqual(chol.shape, (n_total, n_total))

        # K + diag(tau^2 / y_std^2) must differ from the noise-free assembly
        kernel = sm._get_kernel_params()
        bare = sm.compute_blockwise_K(sm.X_norma_all, sm.X_norma_all, kernel)
        implied = chol @ chol.T - bare
        np.testing.assert_allclose(
            np.diag(implied),
            sm._het_noise_vector() + sm.options["nugget"],
            rtol=1e-8,
            atol=1e-12,
        )

        # the gradient path uses the same diagonal
        value, grad = sm.neg_log_likelihood_grad(sm.optimal_theta)
        self.assertAlmostEqual(value, sm.neg_log_likelihood(sm.optimal_theta), places=6)
        self.assertEqual(grad.size, sm.optimal_theta.size)

    def test_het_noise_variances_all_levels(self):
        """predict_variances_all_levels also carries the predicted noise."""
        sm, _ = train_het_noise(predict_with_noise=True, opt_max_eval=30)
        x = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
        matrix = sm.predict_variances_all_levels(x)
        self.assertEqual(matrix.shape, (11, sm.lvl))
        _, covariances = sm.predict_all_levels(x)
        for level in range(sm.lvl):
            np.testing.assert_allclose(
                matrix[:, level], np.asarray(covariances[level]).ravel(), rtol=1e-10
            )

    def test_noise_model_fitted_lazily(self):
        """predict_noise fits the auxiliary model on demand."""
        sm, _ = train_het_noise(predict_with_noise=False, opt_max_eval=30)
        self.assertIsNone(sm.noise_model)
        field = sm.predict_noise(np.linspace(0, 1, 5).reshape(-1, 1))
        self.assertIsNotNone(sm.noise_model)
        self.assertTrue(np.all(field > 0.0))

    def test_sequential_with_het_noise(self):
        """The level restriction also truncates the heteroscedastic noise."""
        sm, data = train_het_noise(
            sequential_opt=True, predict_with_noise=True, opt_max_eval=30
        )
        self.assertEqual(sm.opt_report["mode"], "sequential")
        with sm._restricted_levels(1):
            self.assertEqual(sm.lvl, 1)
            self.assertEqual(len(sm.options["noise0"]), 1)
            self.assertEqual(np.size(sm.options["noise0"][0]), data["x_lf"].shape[0])
        self.assertEqual(len(sm.options["noise0"]), 2)
        self.assertEqual(sm.lvl, 2)

    def test_sequential_without_noise_parameters(self):
        """Sequential stages also work when no noise is estimated."""
        sm = train_forrester(sequential_opt=True, eval_noise=False, opt_max_eval=40)
        self.assertEqual(sm.optimal_theta.size, sm._n_kernel_params())
        stages = sm.opt_report["stages"]
        self.assertEqual(stages[0]["n_free_params"], 2)  # sigma_0, l_0
        self.assertEqual(stages[1]["n_free_params"], 3)  # sigma_1, l_1, rho_1

    @unittest.skipIf(NO_NLOPT, "nlopt not installed")
    def test_nlopt_optimizers(self):
        """The nlopt back-ends train and fill the report."""
        for hyper_opt in ("Cobyla-nlopt", "Lbfgs-nlopt"):
            sm = train_forrester(
                hyper_opt=hyper_opt, eval_noise=True, noise0=[1e-3], opt_max_eval=60
            )
            self.assertTrue(np.all(np.isfinite(sm.optimal_theta)))
            self.assertGreater(sm.opt_report["n_eval"], 0)
            self.assertTrue(np.isfinite(sm.opt_report["joint_nll"]))

    def test_nlopt_objective_wrappers(self):
        """The nlopt-style wrappers fill the gradient array in place."""
        rng = np.random.default_rng(9)
        sm = build_untrained_model(n_levels=2, dim=1)
        x = well_conditioned_point(sm, rng)
        param = sm._transform_optimizer_param(x)

        grad = np.zeros(x.size)
        value = sm.neg_log_likelihood_nlopt_grad(x, grad)
        expected_value, expected_grad = sm.neg_log_likelihood_scipy_grad(x)
        self.assertAlmostEqual(value, expected_value, places=10)
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-12)
        self.assertAlmostEqual(
            sm.neg_log_likelihood_nlopt(x), sm.neg_log_likelihood(param), places=10
        )

    def test_safe_call_grad_on_exception(self):
        """A raising gradient objective is penalised, not propagated."""
        sm = build_untrained_model(n_levels=2, dim=1)

        def broken(_x, with_grad=False):
            raise np.linalg.LinAlgError("not positive definite")

        value, grad = sm._safe_call_grad(broken, np.zeros(4))
        self.assertEqual(value, sm._FAILED_NLL)
        np.testing.assert_allclose(grad, np.zeros(4))

    def test_sample_starts_without_generator(self):
        """_sample_starts falls back to a fresh generator when rng is absent."""
        sm = build_untrained_model(n_levels=2, dim=1)
        sm.rng = None
        _, lower, upper = sm._build_hyperparameter_vectors()
        starts = sm._sample_starts(1, lower, upper)
        self.assertEqual(len(starts), 1)
        self.assertTrue(np.all(starts[0] >= lower))

    # ------------------------------------------------------------------
    # documentation example
    # ------------------------------------------------------------------
    @staticmethod
    def run_mfck_example():
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.applications.mfck import MFCK
        from smt.sampling_methods import LHS

        # low fidelity model
        def lf_function(x):
            import numpy as np

            return (
                0.5 * ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)
                + (x - 0.5) * 10.0
                - 5
            )

            # high fidelity model

        def hf_function(x):
            import numpy as np

            return ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)

        # Problem set up
        xlimits = np.array([[0.0, 1.0]])

        # Example with non-nested input data
        Obs_HF = 7  # Number of observations of HF
        Obs_LF = 14  # Number of observations of LF

        # Creation of LHS for non-nested LF data
        sampling = LHS(
            xlimits=xlimits,
            criterion="ese",
        )

        xt_e_non = sampling(Obs_HF)
        xt_c_non = sampling(Obs_LF)

        # Evaluate the LF function
        yt_e_non = hf_function(xt_e_non)
        yt_c_non = lf_function(xt_c_non)

        sm_non_nested = MFCK(
            theta0=xt_e_non.shape[1] * [0.5],
            theta_bounds=[1e-2, 100],
            corr="squar_exp",
            eval_noise=False,
        )
        sm_non_nested.options["lambda"] = 0.0  # Without regularization

        # low-fidelity dataset names being integers from 0 to level-1
        sm_non_nested.set_training_values(xt_c_non, yt_c_non, name=0)
        # high-fidelity dataset without name
        sm_non_nested.set_training_values(xt_e_non, yt_e_non)

        # train the model
        sm_non_nested.train()

        x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

        m_non_nested, c_non_nested = sm_non_nested.predict_all_levels(x)

        plt.figure()
        plt.title("Example with non-nested input data")
        plt.plot(x, hf_function(x), label="reference HF")
        plt.plot(x, lf_function(x), label="reference LF")
        plt.plot(x, m_non_nested[1], linestyle="-.", label="mean_gp_non_nested")
        plt.scatter(xt_e_non, yt_e_non, marker="o", color="k", label="HF doe")
        plt.scatter(
            xt_c_non, yt_c_non, marker="*", color="c", label="LF non-nested doe"
        )

        plt.legend(loc=0)
        plt.ylim(-10, 17)
        plt.xlim(-0.1, 1.1)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

        plt.show()

    @staticmethod
    def run_mfck_sequential_example():
        """Sequential vs joint hyper-parameter estimation, with gradients."""
        import numpy as np

        from smt.applications.mfck import MFCK
        from smt.sampling_methods import LHS

        def lf_function(x):
            return 0.5 * hf_function(x) + (x - 0.5) * 10.0 - 5

        def hf_function(x):
            return ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)

        xlimits = np.array([[0.0, 1.0]])
        sampling = LHS(xlimits=xlimits, criterion="ese", seed=0)
        xt_hf, xt_lf = sampling(8), sampling(16)

        for sequential in (False, True):
            sm = MFCK(
                theta0=[1.0],
                eval_noise=True,
                noise0=[1e-3],
                hyper_opt="Cobyla",
                n_start=1,
                sequential_opt=sequential,
                opt_max_eval=100,
                print_global=False,
            )
            sm.set_training_values(xt_lf, lf_function(xt_lf), name=0)
            sm.set_training_values(xt_hf, hf_function(xt_hf))
            sm.train()
            report = sm.opt_report
            print(
                f"{report['mode']:<11} {report['n_eval']:>4} evaluations, "
                f"joint NLL = {report['joint_nll']:.3f}"
            )
            for stage in report["stages"]:
                print(
                    f"    {stage['stage']:<8} {stage['n_free_params']} free "
                    f"params, nll = {stage['nll']:.3f}"
                )

    # run scripts are used in documentation as documentation is not always rebuild
    # make a test run by pytest to test the run scripts
    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_run_mfck_example(self):
        self.run_mfck_example()

    def test_run_mfck_sequential_example(self):
        with Silence():
            self.run_mfck_sequential_example()


if __name__ == "__main__":
    unittest.main()
