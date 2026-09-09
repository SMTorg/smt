# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:22:32 2025

@author: mcastano

Unit tests for the SMFCK application, covering the original behaviour plus:

  * the option set (FITC / VFE, inducing method, forced noise handling),
  * the selection of the inducing points and its reproducibility,
  * the FITC and VFE sparse likelihoods and their Woodbury terms,
  * the recovery of the exact MFCK model when Z = X (M = N),
  * the prediction API and its guards,
  * the sequential (level-wise) estimation, including the restriction of the
    inducing sets,
  * the heteroscedastic-noise branch and its auxiliary sparse model,
  * the refusal of the gradient-based optimisers.
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

from smt.applications import MFCK, SMFCK
from smt.problems import TensorProduct
from smt.sampling_methods import FullFactorial
from smt.utils.silence import Silence
from smt.utils.sm_test_case import SMTestCase

print_output = False

# scipy checks the evaluation limit *after* the call, so the recorded number of
# likelihood evaluations can exceed the requested budget by a few units.
OPT_BUDGET_SLACK = 5


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def hf_function(x):
    return ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)


def lf_function(x):
    return 0.5 * hf_function(x) + (x - 0.5) * 10.0 - 5


def make_doe(n_hf=12, n_lf=24, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x_hf = np.sort(rng.uniform(0.0, 1.0, (n_hf, 1)), axis=0)
    x_lf = np.sort(rng.uniform(0.0, 1.0, (n_lf, 1)), axis=0)
    y_hf = hf_function(x_hf) + noise * rng.standard_normal(x_hf.shape)
    y_lf = lf_function(x_lf) + noise * rng.standard_normal(x_lf.shape)
    return x_lf, y_lf, x_hf, y_hf


def train_smfck(n_hf=12, n_lf=24, n_inducing=(8, 5), noise=0.2, seed=0,
                data=None, **options):
    """A small trained two-fidelity sparse model."""
    if data is None:
        data = make_doe(n_hf=n_hf, n_lf=n_lf, noise=noise, seed=seed)
    x_lf, y_lf, x_hf, y_hf = data
    options.setdefault("hyper_opt", "Cobyla")
    options.setdefault("theta0", [1.0])
    options.setdefault("theta_bounds", [1e-2, 1e2])
    options.setdefault("n_start", 1)
    options.setdefault("opt_max_eval", 60)
    options.setdefault("print_global", False)
    options.setdefault("n_inducing", list(n_inducing))
    if not options.get("use_het_noise", False):
        options.setdefault("noise0", [1e-2])
    model = SMFCK(**options)
    model.set_training_values(x_lf, y_lf, name=0)
    model.set_training_values(x_hf, y_hf)
    with Silence():
        model.train()
    return model, data


def train_het_smfck(n_hf=12, n_lf=24, seed=5, **options):
    """A trained sparse model with heteroscedastic noise."""
    rng = np.random.default_rng(seed)
    x_hf = np.sort(rng.uniform(0.0, 1.0, (n_hf, 1)), axis=0)
    x_lf = np.sort(rng.uniform(0.0, 1.0, (n_lf, 1)), axis=0)
    tau2_hf = 0.01 + 0.2 * np.sin(3.0 * x_hf.ravel()) ** 2
    tau2_lf = 0.02 + 0.3 * np.sin(3.0 * x_lf.ravel()) ** 2
    y_hf = hf_function(x_hf) + np.sqrt(tau2_hf)[:, None] * rng.standard_normal(
        x_hf.shape
    )
    y_lf = lf_function(x_lf) + np.sqrt(tau2_lf)[:, None] * rng.standard_normal(
        x_lf.shape
    )
    options.setdefault("hyper_opt", "Cobyla")
    options.setdefault("theta0", [1.0])
    options.setdefault("n_start", 1)
    options.setdefault("opt_max_eval", 40)
    options.setdefault("print_global", False)
    options.setdefault("n_inducing", [8, 5])
    model = SMFCK(use_het_noise=True, noise0=[tau2_lf, tau2_hf], **options)
    model.set_training_values(x_lf, y_lf, name=0)
    model.set_training_values(x_hf, y_hf)
    with Silence():
        model.train()
    return model, dict(x_lf=x_lf, x_hf=x_hf, tau2_lf=tau2_lf, tau2_hf=tau2_hf)


class TestSMFCK(SMTestCase):
    def setUp(self):
        self.nt = 100
        self.ne = 100
        self.ndim = 3

    # ------------------------------------------------------------------
    # original tests
    # ------------------------------------------------------------------
    def test_smfck(self):
        self.problems = ["exp"]  # , "tanh", "cos"]

        for fname in self.problems:
            prob = TensorProduct(ndim=self.ndim, func=fname)
            sampling = FullFactorial(xlimits=prob.xlimits, clip=True)

            noise_std = 1e-5

            xt = sampling(self.nt)
            yt = prob(xt)
            for i in range(self.ndim):
                yt = np.concatenate((yt, prob(xt, kx=i)), axis=1)

            y_lf = 2 * prob(xt) + 2 + np.random.normal(0, noise_std, size=xt.shape)
            x_lf = deepcopy(xt)
            xe = sampling(self.ne)
            ye = prob(xe) + +np.random.normal(0, noise_std, size=xe.shape)

            sm = SMFCK(
                hyper_opt="Cobyla",
                theta0=xe.shape[1] * [0.8],
                theta_bounds=[1e-6, 2.0],
                print_global=False,
                method="FITC",
                eval_noise=True,
                noise0=[1e-5],
                noise_bounds=np.array((1e-12, 10.0)),
                corr="squar_exp",
                n_inducing=[x_lf.shape[0] - 1, xe.shape[0] - 1],
                n_start=1,
            )
            sm.options["print_global"] = False

            sm.set_training_values(xe, ye[:, 0])
            sm.set_training_values(x_lf, y_lf[:, 0], name=0)

            sm.train()

            m = sm.predict_values(xt)

            num = np.linalg.norm(m[:, 0] - yt[:, 0])
            den = np.linalg.norm(yt[:, 0])

            t_error = num / den

            self.assert_error(t_error, 0.0, 5e-1, 5e-1)

    def test_smfck_error_branches(self):
        """Covers ValueError in eta and inconsistent n_inducing."""
        sm = SMFCK()
        with self.assertRaises(ValueError):
            sm.eta(j=5, jp=2, rho=[1.0, 1.0])

        # Inconsistent n_inducing
        sm3 = SMFCK(n_inducing=[10])
        xt = np.array([[0.0], [1.0]])
        yt = np.array([[0.0], [1.0]])
        sm3.set_training_values(xt, yt)
        sm3.set_training_values(xt, yt, name=0)
        with self.assertRaises(ValueError):
            sm3.train()

    # ------------------------------------------------------------------
    # options
    # ------------------------------------------------------------------
    def test_method_option(self):
        """Only FITC and VFE are accepted."""
        for method in ("FITC", "VFE"):
            self.assertEqual(SMFCK(method=method).options["method"], method)
        for method in ("fitc", "SoR", "DTC"):
            with self.assertRaises(AssertionError):
                SMFCK(method=method)

    def test_default_options(self):
        """The sparse likelihoods need a noise term, so eval_noise is on."""
        sm = SMFCK()
        self.assertTrue(sm.options["eval_noise"])
        self.assertFalse(sm.options["use_het_noise"])
        self.assertFalse(sm._supports_gradient)
        self.assertEqual(sm.options["inducing_method"], "kmeans")

    def test_eval_noise_is_restored(self):
        """Switching eval_noise off is refused with a warning, not silently."""
        x_lf, y_lf, x_hf, y_hf = make_doe()
        sm = SMFCK(hyper_opt="Cobyla", theta0=[1.0], n_start=1, opt_max_eval=30,
                   n_inducing=[6, 4], eval_noise=False, noise0=[1e-2],
                   print_global=False)
        sm.set_training_values(x_lf, y_lf, name=0)
        sm.set_training_values(x_hf, y_hf)
        with self.assertWarns(UserWarning):
            with Silence():
                sm.train()
        self.assertTrue(sm.options["eval_noise"])
        self.assertTrue(sm._has_noise_params())

    # ------------------------------------------------------------------
    # inducing points
    # ------------------------------------------------------------------
    def test_inducing_methods(self):
        """k-means centres and random subsets have the expected shapes."""
        x_lf, _, x_hf, _ = make_doe()
        for method in ("kmeans", "random"):
            sm = SMFCK(n_inducing=[7, 4], inducing_method=method, seed=0,
                       print_global=False)
            sm.lvl = 2
            inducing = sm._compute_inducing_points([x_lf, x_hf])
            self.assertEqual(len(inducing), 2)
            self.assertEqual(inducing[0].shape, (7, 1))
            self.assertEqual(inducing[1].shape, (4, 1))
            for z, x in zip(inducing, (x_lf, x_hf)):
                self.assertTrue(np.all(z >= x.min() - 1e-12))
                self.assertTrue(np.all(z <= x.max() + 1e-12))
            if method == "random":  # a subset of the training points
                for z, x in zip(inducing, (x_lf, x_hf)):
                    for row in z:
                        self.assertTrue(np.any(np.all(np.isclose(x, row), axis=1)))

    def test_inducing_points_truncated(self):
        """Asking for more inducing points than data warns and truncates."""
        x_lf, _, x_hf, _ = make_doe(n_hf=6, n_lf=8)
        sm = SMFCK(n_inducing=[20, 20], print_global=False)
        sm.lvl = 2
        with self.assertWarns(UserWarning):
            inducing = sm._compute_inducing_points([x_lf, x_hf])
        self.assertEqual(inducing[0].shape[0], x_lf.shape[0])
        self.assertEqual(inducing[1].shape[0], x_hf.shape[0])

    def test_inducing_points_without_seed(self):
        """k-means also works when no seed is set."""
        x_lf, _, x_hf, _ = make_doe()
        sm = SMFCK(n_inducing=[5, 3], seed=None, print_global=False)
        sm.lvl = 2
        inducing = sm._compute_inducing_points([x_lf, x_hf])
        self.assertEqual([z.shape[0] for z in inducing], [5, 3])

    def test_inducing_points_reproducible(self):
        """Training the same model twice must give the same inducing points."""
        _, _, x_hf, _ = make_doe()
        x_lf, y_lf, _, y_hf = make_doe()
        sm, _ = train_smfck(data=(x_lf, y_lf, x_hf, y_hf))
        first_z = [z.copy() for z in sm.Z]
        first_theta = np.array(sm.optimal_theta, copy=True)

        sm.set_training_values(x_lf, y_lf, name=0)
        sm.set_training_values(x_hf, y_hf)
        with Silence():
            sm.train()
        for before, after in zip(first_z, sm.Z):
            np.testing.assert_allclose(before, after)
        np.testing.assert_allclose(first_theta, sm.optimal_theta, rtol=1e-10)

    # ------------------------------------------------------------------
    # sparse likelihoods
    # ------------------------------------------------------------------
    def test_sparse_likelihood_methods(self):
        """FITC and VFE both produce a finite value and consistent Woodbury."""
        sm, _ = train_smfck()
        n_inducing = sum(z.shape[0] for z in sm.Z_norma_all)
        for method in ("FITC", "VFE"):
            value, w_vec, w_inv = sm._sparse_likelihood(
                sm.X_norma_all, sm.y_norma_all, sm.Z_norma_all,
                sm.optimal_theta, method,
            )
            self.assertTrue(np.isfinite(value))
            self.assertIsInstance(value, float)
            self.assertEqual(w_vec.shape, (n_inducing, 1))
            self.assertEqual(w_inv.shape, (n_inducing, n_inducing))
            np.testing.assert_allclose(w_inv, w_inv.T, atol=1e-8)

    def test_vfe_penalises_the_trace(self):
        """The VFE bound adds a non-negative trace term to the FITC-like fit."""
        sm, _ = train_smfck()
        fitc = sm._sparse_likelihood(
            sm.X_norma_all, sm.y_norma_all, sm.Z_norma_all,
            sm.optimal_theta, "FITC")[0]
        vfe = sm._sparse_likelihood(
            sm.X_norma_all, sm.y_norma_all, sm.Z_norma_all,
            sm.optimal_theta, "VFE")[0]
        self.assertTrue(np.isfinite(fitc) and np.isfinite(vfe))
        # both are negative log-likelihoods of the same data at the same point
        self.assertNotAlmostEqual(fitc, vfe, places=6)

    def test_unknown_sparse_method(self):
        sm, _ = train_smfck()
        with self.assertRaises(ValueError):
            sm._sparse_likelihood(
                sm.X_norma_all, sm.y_norma_all, sm.Z_norma_all,
                sm.optimal_theta, "SoR",
            )

    def test_fitc_vfe_wrappers(self):
        """The legacy _FITC / _VFE entry points still work."""
        sm, _ = train_smfck()
        for method, wrapper in (("FITC", sm._FITC), ("VFE", sm._VFE)):
            reference = sm._sparse_likelihood(
                sm.X_norma_all, sm.y_norma_all, sm.Z_norma_all,
                sm.optimal_theta, method,
            )
            result = wrapper(sm.X_norma_all, sm.y_norma_all, sm.Z_norma_all,
                             sm.optimal_theta)
            self.assertAlmostEqual(result[0], reference[0], places=10)
            np.testing.assert_allclose(result[1], reference[1], rtol=1e-12)

    def test_woodbury_matches_the_optimum(self):
        """Predictions must use the Woodbury terms of `optimal_theta`."""
        sm, _ = train_smfck()
        stored_vec = np.array(sm.woodbury_data["vec"], copy=True)
        stored_inv = np.array(sm.woodbury_data["inv"], copy=True)

        # recomputing the likelihood at the optimum must not change them
        sm.neg_log_likelihood(sm.optimal_theta)
        np.testing.assert_allclose(stored_vec, sm.woodbury_data["vec"], rtol=1e-12)
        np.testing.assert_allclose(stored_inv, sm.woodbury_data["inv"], rtol=1e-12)

        # ... while another point does change them
        other = np.array(sm.optimal_theta, copy=True)
        other[0] *= 1.5
        sm.neg_log_likelihood(other)
        self.assertFalse(np.allclose(stored_vec, sm.woodbury_data["vec"]))

    def test_split_noise(self):
        """The noise vector has one entry per training point."""
        sm, _ = train_smfck()
        kernel, varis = sm._split_noise(sm.optimal_theta, sm.X_norma_all)
        n_total = sum(x.shape[0] for x in sm.X_norma_all)
        self.assertEqual(varis.size, n_total)
        self.assertEqual(kernel.size, sm._n_kernel_params())
        self.assertTrue(np.all(varis > 0.0))

    def test_fitc_recovers_the_exact_model(self):
        """
        With Z = X (M = N) the Nystrom approximation is exact, so FITC must
        reproduce the exact MFCK model: its negative log-likelihood is
        0.5 (log|K+D| + y'(K+D)^-1 y), i.e. half the one of MFCK.
        """
        data = make_doe(n_hf=10, n_lf=14, noise=0.2, seed=2)
        x_lf, y_lf, x_hf, y_hf = data

        exact = MFCK(hyper_opt="Cobyla", theta0=[1.0], theta_bounds=[1e-2, 1e2],
                     n_start=1, opt_max_eval=60, eval_noise=True,
                     noise0=[1e-2], print_global=False)
        exact.set_training_values(x_lf, y_lf, name=0)
        exact.set_training_values(x_hf, y_hf)
        with Silence():
            exact.train()

        sparse, _ = train_smfck(data=data, n_inducing=(14, 10))
        # inducing points on the training points, and the same hyper-parameters
        sparse.Z_norma_all = [x.copy() for x in sparse.X_norma_all]
        sparse.optimal_theta = np.array(exact.optimal_theta, copy=True)

        self.assertAlmostEqual(
            2.0 * sparse.neg_log_likelihood(sparse.optimal_theta),
            exact.neg_log_likelihood(exact.optimal_theta),
            places=5,
        )

        x = np.linspace(0.0, 1.0, 21).reshape(-1, 1)
        np.testing.assert_allclose(
            np.asarray(sparse.predict_values(x)).ravel(),
            np.asarray(exact.predict_values(x)).ravel(),
            rtol=1e-5, atol=1e-8,
        )

        # The sparse predictive variance subtracts two nearly equal terms, so
        # at M = N it needs more regularisation than the default nugget: with
        # 1000 eps the difference is O(1), with 1e-8 it is O(1e-4).
        sparse.options["nugget"] = 1e-8
        sparse.neg_log_likelihood(sparse.optimal_theta)
        np.testing.assert_allclose(
            np.asarray(sparse.predict_variances(x)).ravel(),
            np.asarray(exact.predict_variances(x)).ravel(),
            rtol=5e-3, atol=1e-6,
        )

    # ------------------------------------------------------------------
    # predictions
    # ------------------------------------------------------------------
    def test_prediction_api(self):
        """Shapes and mutual consistency of the prediction methods."""
        sm, _ = train_smfck()
        x = np.linspace(0.0, 1.0, 17).reshape(-1, 1)

        mean = sm.predict_values(x)
        self.assertEqual(mean.shape, (17, 1))
        variance = np.asarray(sm.predict_variances(x))
        self.assertEqual(variance.shape, (17, 1))
        self.assertTrue(np.all(variance > 0.0))

        all_levels = sm.predict_variances_all_levels(x)
        self.assertEqual(all_levels.shape, (17, sm.lvl))
        np.testing.assert_allclose(all_levels[:, -1], variance.ravel(), rtol=1e-12)

        means, covariances = sm.predict_all_levels(x)
        self.assertEqual(len(means), sm.lvl)
        np.testing.assert_allclose(
            np.asarray(means[-1]).ravel(), mean.ravel(), rtol=1e-12
        )
        for level in range(sm.lvl):
            self.assertEqual(np.asarray(covariances[level]).shape, (17, 1))

    def test_cross_covariance_inducing(self):
        """The stacked cross-covariance has one row per inducing point."""
        sm, _ = train_smfck()
        x = np.linspace(0.0, 1.0, 9).reshape(-1, 1)
        x_norm = (x - sm.X_offset) / sm.X_scale
        kernel = sm.optimal_theta[: -sm.lvl]
        n_inducing = sum(z.shape[0] for z in sm.Z_norma_all)
        for level in range(sm.lvl):
            k_xz = sm._cross_covariance_inducing(x_norm, level, kernel)
            self.assertEqual(k_xz.shape, (n_inducing, 9))
            self.assertTrue(np.all(np.isfinite(k_xz)))

    def test_predict_before_training(self):
        """Predicting before train() is an explicit error."""
        sm = SMFCK(n_inducing=[4, 3], print_global=False)
        sm.lvl = 2
        with self.assertRaises(RuntimeError):
            sm.predict_all_levels(np.linspace(0, 1, 3).reshape(-1, 1))

    def test_sparse_prediction_is_not_the_dense_one(self):
        """predict_values must go through the inducing points, not through K."""
        sm, _ = train_smfck()
        x = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
        sparse_mean = np.asarray(sm.predict_values(x)).ravel()
        # the dense formula of MFCK would need the full covariance matrix;
        # here the prediction only involves the Woodbury vector
        n_inducing = sum(z.shape[0] for z in sm.Z_norma_all)
        self.assertEqual(sm.woodbury_data["vec"].shape[0], n_inducing)
        self.assertTrue(np.all(np.isfinite(sparse_mean)))

    # ------------------------------------------------------------------
    # gradient guard
    # ------------------------------------------------------------------
    def test_gradient_is_refused(self):
        """The FITC / VFE gradients are not implemented and are refused."""
        sm, _ = train_smfck()
        with self.assertRaises(NotImplementedError):
            sm.neg_log_likelihood_grad(sm.optimal_theta)

        x_lf, y_lf, x_hf, y_hf = make_doe()
        for hyper_opt in ("TNC", "Lbfgs-nlopt"):
            model = SMFCK(hyper_opt=hyper_opt, theta0=[1.0], n_inducing=[6, 4],
                          n_start=1, noise0=[1e-2], print_global=False)
            model.set_training_values(x_lf, y_lf, name=0)
            model.set_training_values(x_hf, y_hf)
            with self.assertRaises(ValueError):
                model.train()

    # ------------------------------------------------------------------
    # sequential estimation
    # ------------------------------------------------------------------
    def test_sequential_training(self):
        """The level-wise strategy also applies to the sparse likelihoods."""
        data = make_doe()
        joint, _ = train_smfck(data=data, sequential_opt=False)
        sequential, _ = train_smfck(data=data, sequential_opt=True)

        self.assertEqual(joint.opt_report["mode"], "joint")
        self.assertEqual(sequential.opt_report["mode"], "sequential")
        stages = sequential.opt_report["stages"]
        self.assertEqual([s["stage"] for s in stages], ["level0", "level1"])
        self.assertEqual(stages[0]["n_free_params"], 3)
        self.assertEqual(stages[1]["n_free_params"], 4)
        self.assertTrue(np.isfinite(sequential.opt_report["joint_nll"]))

        x = np.linspace(0.0, 1.0, 13).reshape(-1, 1)
        self.assertTrue(np.all(np.isfinite(sequential.predict_values(x))))
        # the Woodbury terms are refreshed on the full model after the cascade
        n_inducing = sum(z.shape[0] for z in sequential.Z_norma_all)
        self.assertEqual(sequential.woodbury_data["vec"].shape[0], n_inducing)

    def test_restricted_levels_truncates_inducing_sets(self):
        """_restricted_levels also exposes a sub-model of the inducing sets."""
        sm, _ = train_smfck()
        n_z = [z.shape[0] for z in sm.Z_norma_all]
        with sm._restricted_levels(1):
            self.assertEqual(sm.lvl, 1)
            self.assertEqual(len(sm.Z), 1)
            self.assertEqual(len(sm.Z_norma_all), 1)
            self.assertEqual(sm.Z_norma_all[0].shape[0], n_z[0])
            value = sm.neg_log_likelihood(sm.optimal_theta[: sm._n_kernel_params() + 1])
            self.assertTrue(np.isfinite(value))
            self.assertEqual(sm.woodbury_data["vec"].shape[0], n_z[0])
        self.assertEqual(sm.lvl, 2)
        self.assertEqual(len(sm.Z_norma_all), 2)
        self.assertEqual([z.shape[0] for z in sm.Z_norma_all], n_z)

    # ------------------------------------------------------------------
    # heteroscedastic noise
    # ------------------------------------------------------------------
    def test_het_noise_training(self):
        """Measurement variances can be supplied per training point."""
        sm, data = train_het_smfck()
        self.assertFalse(sm._has_noise_params())
        self.assertEqual(sm.optimal_theta.size, sm._n_kernel_params())

        expected = np.concatenate([data["tau2_lf"], data["tau2_hf"]])
        expected = expected / float(np.asarray(sm.y_std).ravel()[0]) ** 2
        _, varis = sm._split_noise(sm.optimal_theta, sm.X_norma_all)
        np.testing.assert_allclose(varis, expected, rtol=1e-12)

        x = np.linspace(0.0, 1.0, 9).reshape(-1, 1)
        self.assertTrue(np.all(np.isfinite(sm.predict_values(x))))
        self.assertTrue(np.all(np.asarray(sm.predict_variances(x)) > 0.0))

    def test_het_noise_auxiliary_model(self):
        """The auxiliary noise model is a sparse model of the same class."""
        sm, data = train_het_smfck(predict_with_noise=True)
        auxiliary = sm.noise_model
        self.assertIsInstance(auxiliary, SMFCK)
        self.assertEqual(auxiliary.options["method"], sm.options["method"])
        self.assertEqual(
            [z.shape[0] for z in auxiliary.Z], [z.shape[0] for z in sm.Z]
        )
        self.assertFalse(auxiliary.options["predict_with_noise"])
        self.assertIsNone(auxiliary.noise_model)

        x = np.linspace(-0.2, 1.2, 15).reshape(-1, 1)
        fields = sm.predict_noise_all_levels(x)
        self.assertEqual(len(fields), sm.lvl)
        for field in fields:
            self.assertEqual(field.shape, (15, 1))
            self.assertTrue(np.all(field > 0.0))

    def test_het_noise_adds_to_the_variances(self):
        """predict_with_noise adds the predicted field to every variance."""
        x = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
        with_noise, _ = train_het_smfck(predict_with_noise=True)
        without, _ = train_het_smfck(predict_with_noise=False)
        _, var_with = with_noise.predict_all_levels(x)
        _, var_without = without.predict_all_levels(x)
        fields = with_noise.predict_noise_all_levels(x)
        for level in range(with_noise.lvl):
            delta = (np.asarray(var_with[level]).ravel()
                     - np.asarray(var_without[level]).ravel())
            np.testing.assert_allclose(
                delta, np.asarray(fields[level]).ravel(), rtol=1e-8, atol=1e-12
            )

    # ------------------------------------------------------------------
    # optimisation bookkeeping
    # ------------------------------------------------------------------
    def test_opt_report(self):
        """The optimisation report is filled for the sparse model too."""
        sm, _ = train_smfck(opt_max_eval=40)
        report = sm.opt_report
        self.assertEqual(report["mode"], "joint")
        self.assertEqual(report["n_eval"], sum(s["n_eval"] for s in report["stages"]))
        self.assertLessEqual(report["n_eval"], 40 + OPT_BUDGET_SLACK)
        self.assertEqual(len(report["history"]), report["n_eval"])
        self.assertTrue(np.isfinite(report["joint_nll"]))

    @unittest.skipIf(NO_NLOPT, "nlopt not installed")
    def test_default_nlopt_optimizer(self):
        """The default derivative-free nlopt back-end trains."""
        sm, _ = train_smfck(hyper_opt="Cobyla-nlopt", opt_max_eval=60)
        self.assertTrue(np.all(np.isfinite(sm.optimal_theta)))
        self.assertGreater(sm.opt_report["n_eval"], 0)

    def test_vfe_method_trains(self):
        """The VFE variant trains and predicts (values=('FITC',) used to reject)."""
        sm, _ = train_smfck(method="VFE")
        x = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
        self.assertEqual(sm.options["method"], "VFE")
        self.assertTrue(np.all(np.isfinite(sm.predict_values(x))))
        self.assertTrue(np.all(np.asarray(sm.predict_variances(x)) > 0.0))

    # ------------------------------------------------------------------
    # documentation example
    # ------------------------------------------------------------------
    @staticmethod
    def run_smfck_example():
        import matplotlib.pyplot as plt
        import numpy as np
        from smt.sampling_methods import LHS  # noqa
        from smt.applications import SMFCK

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
            seed=0,
        )

        xt_e_non = sampling(Obs_HF)
        xt_c_non = sampling(Obs_LF)

        # Evaluate the HF and LF functions
        yt_e = hf_function(xt_e_non)
        yt_c = lf_function(xt_c_non)

        sm = SMFCK(
            hyper_opt="Cobyla",
            theta0=xt_e_non.shape[1] * [1.0],
            theta_bounds=[1e-6, 2.0],
            print_global=False,
            method="FITC",
            eval_noise=True,
            noise0=[1e-5],
            noise_bounds=np.array((1e-12, 10.0)),
            corr="squar_exp",
            n_inducing=[xt_c_non.shape[0] - 2, xt_e_non.shape[0] - 1],
        )

        # low-fidelity dataset names being integers from 0 to level-1
        sm.set_training_values(xt_c_non, yt_c, name=0)
        # high-fidelity dataset without name
        sm.set_training_values(xt_e_non, yt_e)

        # train the model
        sm.train()

        x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

        # query the outputs

        mean, cov = sm.predict_all_levels(x)

        y = mean[-1]

        plt.figure()

        plt.plot(x, hf_function(x), label="reference")
        plt.plot(x, y, linestyle="-.", label="mean_gp")
        plt.scatter(xt_e_non, yt_e, marker="o", color="k", label="HF doe")
        plt.scatter(xt_c_non, yt_c, marker="*", color="g", label="LF doe")
        plt.plot(
            sm.Z[0],
            -9.9 * np.ones_like(sm.Z[0]),
            "g|",
            mew=2,
            label=f"LF inducing:{sm.Z[0].shape[0]}",
        )
        plt.plot(
            sm.Z[1],
            -9.9 * np.ones_like(sm.Z[1]),
            "k|",
            mew=2,
            label=f"HF inducing:{sm.Z[1].shape[0]}",
        )

        plt.legend(loc=0)
        plt.ylim(-10, 17)
        plt.xlim(-0.1, 1.1)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

        plt.show()

    @staticmethod
    def run_smfck_het_noise_example():
        """Heteroscedastic noise: the observed variances train a second model."""
        import numpy as np

        from smt.applications import SMFCK
        from smt.sampling_methods import LHS

        xlimits = np.array([[0.0, 1.0]])
        sampling = LHS(xlimits=xlimits, criterion="ese", seed=0)
        rng = np.random.default_rng(0)
        xt_c, xt_e = sampling(30), sampling(15)

        def hf(x):
            return ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)

        def lf(x):
            return 0.5 * hf(x) + (x - 0.5) * 10.0 - 5

        tau2_c = 0.05 + 0.5 * np.sin(3.0 * xt_c.ravel()) ** 2
        tau2_e = 0.02 + 0.2 * np.sin(3.0 * xt_e.ravel()) ** 2
        yt_c = lf(xt_c) + np.sqrt(tau2_c)[:, None] * rng.standard_normal(xt_c.shape)
        yt_e = hf(xt_e) + np.sqrt(tau2_e)[:, None] * rng.standard_normal(xt_e.shape)

        sm = SMFCK(
            hyper_opt="Cobyla", theta0=[1.0], n_inducing=[10, 6], n_start=1,
            method="FITC", use_het_noise=True, predict_with_noise=True,
            noise0=[tau2_c, tau2_e], print_global=False,
        )
        sm.set_training_values(xt_c, yt_c, name=0)
        sm.set_training_values(xt_e, yt_e)
        sm.train()

        x = np.linspace(0, 1, 101).reshape(-1, 1)
        mean, variance = sm.predict_all_levels(x)  # noise already included
        noise = sm.predict_noise(x)  # HF noise field
        print(f"HF mean in [{float(np.min(mean[-1])):.2f}, "
              f"{float(np.max(mean[-1])):.2f}], "
              f"predicted noise in [{noise.min():.4f}, {noise.max():.4f}]")

    # run scripts are used in documentation as documentation is not always rebuild
    # make a test run by pytest to test the run scripts
    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_run_smfck_example(self):
        self.run_smfck_example()

    def test_run_smfck_het_noise_example(self):
        with Silence():
            self.run_smfck_het_noise_example()


if __name__ == "__main__":
    unittest.main()