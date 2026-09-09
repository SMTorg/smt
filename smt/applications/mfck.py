# -*- coding: utf-8 -*-
"""
Created on Sat May 04 10:10:12 2024

@author: Mauricio Castano Aguirre <mauricio.castano_aguirre@onera.fr>
Multi-Fidelity co-Kriging model construction for non-nested experimental
design sets.
-------
[1] Loic Le Gratiet (2013). Multi-fidelity Gaussian process modelling
[Doctoral Thesis, Universite Paris-Sud].
[2] Edwin V. Bonilla, Kian Ming A. Chai, and Christopher K. I. Williams
(2007). Multi-task Gaussian Process prediction. In International
Conference on Neural Information Processing Systems.

Hyper-parameter estimation
--------------------------
The MFCK parameter vector is

    theta = ( (sigma_k, l_k)_{0<=k<=L}, (rho_k)_{1<=k<=L}, (tau_k^2)_{0<=k<=L} )

stored, in optimiser space (log10 for all positive parameters, linear scale
for the rho's), with the following block layout::

  [ sigma_0 l_0 | sigma_1 l_1 rho_1 | ... | sigma_L l_L rho_L | tau_0^2 ... tau_L^2 ]
    <- nx+1 ->    <----- nx+2 ----->                             <--- L+1 --->

Two estimation strategies are available (option ``sequential_opt``):

* ``sequential_opt=False`` (joint MLE, default).  The full vector is
  optimised at once by maximising the marginal log-likelihood of
  y = [y_0, ..., y_L].  The search space has (nx+1) + L(nx+2) + (L+1)
  dimensions, which quickly becomes hard for a derivative-free optimiser
  such as COBYLA.

* ``sequential_opt=True`` (sequential / level-wise MLE).  Block-coordinate
  ascent on the same likelihood: at stage k only the block
  (sigma_k, l_k, rho_k, tau_k^2) is free, the blocks of levels 0..k-1 being
  frozen at their previously estimated values.  The objective used at stage
  k is the *exact marginal likelihood of the sub-model made of levels 0..k*,
  which is legitimate because, by the autoregressive construction,

      cov(y_kappa(x), y_kappa'(x')) with kappa, kappa' <= k

  only involves gamma_0..gamma_k and rho_1..rho_k: the joint law of
  (y_0, ..., y_k) does not depend on the parameters of the higher levels.
  Stage k therefore only factorises a (N_0+...+N_k) covariance matrix and
  explores a (nx+2)+1 dimensional space.

  Under *nested* DoEs and the Markov assumption, the likelihood factorises
  as p(y_0) * prod_k p(y_k | y_{k-1}), each factor depending on a single
  block, so the sequential estimator coincides with the joint MLE (this is
  what the recursive formulation of Le Gratiet exploits).  With *non-nested*
  DoEs the factorisation is only approximate: the sequential estimator is a
  fixed point of a block-coordinate ascent, not the joint maximiser.  Use
  ``sequential_refine=True`` to polish the sequential solution with a final
  joint optimisation.
"""

import time
import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np
from scipy import optimize
from scipy.linalg import cho_solve, solve_triangular

from smt.sampling_methods import LHS
from smt.surrogate_models.krg_based import KrgBased
from smt.surrogate_models.krg_based.distances import componentwise_distance, differences
from smt.utils.misc import standardization

try:
    import nlopt as _nlopt  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - depends on the installation
    _nlopt = None


class MFCK(KrgBased):
    # Value returned to the optimiser when the likelihood cannot be evaluated
    # (non-SPD covariance matrix, overflow, ...).  A large *finite* value keeps
    # COBYLA alive instead of crashing the training.
    _FAILED_NLL = 1e10
    # Default evaluation budget of a single optimiser run
    _DEFAULT_MAX_EVAL = {
        "Cobyla": 100,
        "Cobyla-nlopt": 1000,
        "TNC": 100,
        "Lbfgs-nlopt": 100,
    }
    # Optimisers that require the analytical gradient of the likelihood
    _GRADIENT_OPT = ("TNC", "Lbfgs-nlopt")
    # Correlation kernels for which dR/dtheta is available through SMT
    _GRADIENT_KERNELS = ("pow_exp", "abs_exp", "squar_exp", "matern32", "matern52")
    # Set to False in subclasses whose likelihood has no gradient yet (SMFCK)
    _supports_gradient = True
    # Attributes describing the fidelity levels (see _restricted_levels)
    _LEVEL_ATTRS = ("lvl", "X", "y", "X_norma_all", "y_norma_all", "Z", "Z_norma_all")

    @staticmethod
    def _get_nlopt() -> Any:
        if _nlopt is None:  # pragma: no cover - depends on the installation
            raise ImportError("nlopt is required when hyper_opt='Cobyla-nlopt'")
        return _nlopt

    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        self.name = "MFCK"

        declare(
            "rho0",
            2.0,
            types=(float),
            desc="Initial rho for the autoregressive model , \
                  (scalar factor between two consecutive fidelities, \
                    e.g., Y_HF = (Rho) * Y_LF + Gamma",
        )
        declare(
            "rho_bounds",
            [-5, 5],
            types=(list, np.ndarray),
            desc="Bounds for the rho parameter used in the autoregressive model",
        )
        declare(
            "sigma0",
            1.0,
            types=(float),
            desc="Initial variance parameter",
        )
        declare(
            "sigma_bounds",
            [1e-1, 100],
            types=(list, np.ndarray),
            desc="Bounds for the variance parameter",
        )
        declare(
            "lambda",
            0.0,
            types=(float),
            desc="Regularization parameter",
        )
        declare(
            "hyper_opt",
            "Cobyla-nlopt",
            values=("Cobyla", "Cobyla-nlopt", "TNC", "Lbfgs-nlopt"),
            desc="Optimiser for hyperparameters optimisation. 'Cobyla' and \
                  'Cobyla-nlopt' are derivative free, 'TNC' (scipy) and \
                  'Lbfgs-nlopt' use the analytical likelihood gradient",
        )

        self.options["nugget"] = (
            1e-9  # Incresing the nugget for numerical stability reasons
        )
        self.options["hyper_opt"] = (
            "Cobyla"  # MFCK doesn't support gradient-based optimizers
        )
        declare(
            "sequential_opt",
            False,
            types=(bool),
            desc="For sequential optimization of hyperparameters, if True, \
                  the optimization is performed sequentially for each fidelity level",
        )
        declare(
            "sequential_refine",
            False,
            types=(bool),
            desc="Only used when sequential_opt=True. If True, a final joint \
                  optimization of all the hyperparameters is run, initialized \
                  at the sequential solution",
        )
        declare(
            "predict_with_noise",
            False,
            types=bool,
            values=(True, False),
            desc="If use_het_noise is True, an auxiliary multi-fidelity model is \
                  fitted on the observed noise variances so that the noise field \
                  can be predicted at any input, and is added to the predictive \
                  variances",
        )
        declare(
            "noise_target_transform",
            "none",
            values=("none", "log"),
            desc="Transformation applied to the noise variances before fitting \
                  the auxiliary noise model. 'log' guarantees positive \
                  predictions and is usually more accurate",
        )
        declare(
            "opt_max_eval",
            None,
            types=(int, type(None)),
            desc="Maximum number of likelihood evaluations per optimizer run. \
                  None means 100 for 'Cobyla' and 1000 for 'Cobyla-nlopt'",
        )

        # optimisation bookkeeping (filled by train())
        self._nll_history = []
        self.opt_report = {}
        # auxiliary model fitted on the observed noise variances
        self.noise_model = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self):
        """
        Overrides MFK implementation
        Trains the Multi-Fidelity co-Kriging model
        Returns
        -------
        None.
        """
        xt = []
        yt = []
        i = 0
        while self.training_points.get(i, None) is not None:
            xt.append(self.training_points[i][0][0])
            yt.append(self.training_points[i][0][1])
            i = i + 1
        xt.append(self.training_points[None][0][0])
        yt.append(self.training_points[None][0][1])
        self.lvl = i + 1
        self.X = xt
        self.y = np.vstack(yt)
        self._check_param_het_safe()

        (
            _,
            _,
            self.X_offset,
            self.y_mean,
            self.X_scale,
            self.y_std,
        ) = standardization(np.concatenate(xt, axis=0), np.concatenate(yt, axis=0))

        self.X_norma_all = [(x - self.X_offset) / self.X_scale for x in xt]
        self.y_norma_all = np.vstack([(f - self.y_mean) / self.y_std for f in yt])

        self._fit_hyperparameters()

    # ------------------------------------------------------------------
    # Hyper-parameter vector bookkeeping
    # ------------------------------------------------------------------
    def _n_kernel_params(self, n_levels=None):
        """Length of the kernel part of the parameter vector for `n_levels`."""
        n_levels = self.lvl if n_levels is None else n_levels
        return (self.nx + 1) + (n_levels - 1) * (self.nx + 2)

    def _level_block_slice(self, k):
        """
        Slice of the kernel parameter vector holding the block of level k:
        (sigma_0, l_0) for k = 0 and (sigma_k, l_k, rho_k) for k > 0.
        """
        if k == 0:
            return slice(0, self.nx + 1)
        start = (self.nx + 1) + (k - 1) * (self.nx + 2)
        return slice(start, start + self.nx + 2)

    def _check_param_het_safe(self):
        """
        Runs KrgBased._check_param with a heteroscedastic-compatible noise0.

        With use_het_noise=True, MFCK expects one array of noise variances per
        fidelity level, while KrgBased expects a flat array of length nt: the
        raw option would raise an inhomogeneous-array error.  The per-level
        structure is therefore restored right after the check.
        """
        # KrgBased switches hyper_opt from TNC to Cobyla when a noise is
        # estimated, because its own TNC path has no noise gradient.  MFCK does
        # (see neg_log_likelihood_grad), so the user choice is restored.
        hyper_opt = self.options["hyper_opt"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*TNC not available.*")
            if not self.options["use_het_noise"]:
                self._check_param()
                self.options["hyper_opt"] = hyper_opt
                return
            het_noise0 = self.options["noise0"]
            self.options["noise0"] = [
                max(float(self.options["noise_bounds"][0]), 1e-6)
            ]
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*TNC not available.*")
                self._check_param()
        finally:
            self.options["noise0"] = het_noise0
            self.options["hyper_opt"] = hyper_opt

    def _het_noise_vector(self):
        """
        Heteroscedastic noise variances on the *normalised* output scale.

        `noise0` is given by the user in the original output units (a physical
        variance, e.g. CXC_std**2), while K is assembled for the standardised
        outputs: the variances must therefore be divided by y_std**2.  Without
        this the noise is silently interpreted as already normalised, which
        rescales it by var(y) -- a factor of several hundreds on aerodynamic
        coefficients.
        """
        noise = np.concatenate(
            [np.asarray(v, dtype=float).ravel() for v in self.options["noise0"]]
        )
        return noise / float(np.asarray(self.y_std).ravel()[0]) ** 2

    def _has_noise_params(self):
        """True when the noise variances belong to the parameter vector."""
        return bool(self.options["eval_noise"]) and not bool(
            self.options["use_het_noise"]
        )

    def _build_hyperparameter_vectors(self, n_levels=None):
        """
        Builds (theta_ini, lower_bounds, upper_bounds) in optimiser space, i.e.
        log10 for the variances / length-scales / noise variances and linear
        scale for the rho parameters.
        """
        n_levels = self.lvl if n_levels is None else n_levels
        theta0 = np.asarray(self._theta0, dtype=float).ravel()
        sigma_lb, sigma_ub = self.options["sigma_bounds"]
        theta_lb, theta_ub = self.options["theta_bounds"]

        theta_ini, lower, upper = [], [], []
        for k in range(n_levels):
            theta_ini.append(np.log10(np.hstack((self.options["sigma0"], theta0))))
            lower.append(np.log10(np.hstack((sigma_lb, np.full(self.nx, theta_lb)))))
            upper.append(np.log10(np.hstack((sigma_ub, np.full(self.nx, theta_ub)))))
            if k > 0:  # rho_k is NOT log-transformed (it may be negative)
                theta_ini.append(np.atleast_1d(float(self.options["rho0"])))
                lower.append(np.atleast_1d(float(self.options["rho_bounds"][0])))
                upper.append(np.atleast_1d(float(self.options["rho_bounds"][1])))

        theta_ini = np.hstack(theta_ini)
        lower = np.hstack(lower)
        upper = np.hstack(upper)

        if self._has_noise_params():
            noise_lb, noise_ub = self.options["noise_bounds"]
            noise_lb = max(float(noise_lb), np.finfo(float).tiny)
            noise0 = np.asarray(self.options["noise0"], dtype=float).ravel()
            if noise0.size != n_levels:
                noise0 = np.full(n_levels, noise0[0])
            if np.any(noise0 <= 0.0):
                # log10(0) = -inf would poison the whole optimisation
                warnings.warn(
                    "MFCK: 'noise0' contains non-positive values while "
                    "eval_noise=True; they are replaced by "
                    "max(noise_bounds[0], 1e-6) as a starting point.",
                    stacklevel=2,
                )
                noise0 = np.where(noise0 <= 0.0, max(noise_lb, 1e-6), noise0)
            noise0 = np.clip(noise0, noise_lb, noise_ub)

            theta_ini = np.hstack([theta_ini, np.log10(noise0)])
            lower = np.hstack([lower, np.full(n_levels, np.log10(noise_lb))])
            upper = np.hstack([upper, np.full(n_levels, np.log10(float(noise_ub)))])

        return theta_ini, lower, upper

    @contextmanager
    def _restricted_levels(self, n_levels):
        """
        Temporarily exposes only the `n_levels` lowest fidelity levels, so that
        the likelihood machinery (which reads self.lvl / self.X_norma_all /
        self.y_norma_all / self.Z_norma_all) evaluates the marginal likelihood
        of the sub-model made of levels 0..n_levels-1.
        """
        if n_levels >= self.lvl:
            yield
            return

        saved = {a: getattr(self, a) for a in self._LEVEL_ATTRS if hasattr(self, a)}
        n_rows = int(sum(x.shape[0] for x in self.X_norma_all[:n_levels]))
        het = bool(self.options["use_het_noise"])
        saved_noise0 = self.options["noise0"] if het else None
        try:
            self.lvl = n_levels
            self.X = self.X[:n_levels]
            self.X_norma_all = self.X_norma_all[:n_levels]
            self.y_norma_all = self.y_norma_all[:n_rows]
            if hasattr(self, "Z"):
                self.Z = self.Z[:n_levels]
            if hasattr(self, "Z_norma_all"):
                self.Z_norma_all = self.Z_norma_all[:n_levels]
            if het and len(saved_noise0) == len(saved["X"]):
                self.options["noise0"] = list(saved_noise0)[:n_levels]
            yield
        finally:
            for name, value in saved.items():
                setattr(self, name, value)
            if het:
                self.options["noise0"] = saved_noise0

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    def _max_eval(self):
        if self.options["opt_max_eval"] is not None:
            return int(self.options["opt_max_eval"])
        return self._DEFAULT_MAX_EVAL.get(self.options["hyper_opt"], 100)

    def _safe_call(self, objective, x):
        """Evaluates `objective`, never letting a linear-algebra error escape."""
        try:
            value = float(objective(np.asarray(x, dtype=float)))
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return self._FAILED_NLL
        if not np.isfinite(value):
            return self._FAILED_NLL
        return value

    def _safe_call_grad(self, objective, x):
        """Same as _safe_call for objectives returning (value, gradient)."""
        x = np.asarray(x, dtype=float)
        try:
            value, grad = objective(x, with_grad=True)
            value = float(value)
            grad = np.asarray(grad, dtype=float).ravel()
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return self._FAILED_NLL, np.zeros(x.size)
        if not np.isfinite(value) or not np.all(np.isfinite(grad)):
            return self._FAILED_NLL, np.zeros(x.size)
        return value, grad

    def _sample_starts(self, n_samples, lower_bounds, upper_bounds):
        """Additional starting points for the multi-start strategy."""
        if n_samples <= 0:
            return []
        xlimits = np.stack((lower_bounds, upper_bounds), axis=1)
        if n_samples >= 2:
            # NB: LHS(criterion='ese') never returns when asked for one point.
            sampling = LHS(xlimits=xlimits, criterion="ese", seed=self.options["seed"])
            return list(sampling(n_samples))
        rng = getattr(self, "rng", None)
        if rng is None:
            rng = np.random.default_rng(self.options["seed"])
        return [rng.uniform(lower_bounds, upper_bounds)]

    def _single_run(self, objective, x0, lower_bounds, upper_bounds, tag):
        """One COBYLA run (scipy or nlopt) on an arbitrary parameter block."""
        x0 = np.clip(np.asarray(x0, dtype=float), lower_bounds, upper_bounds)
        best = {"x": np.array(x0, copy=True), "f": np.inf, "n": 0}
        max_eval = self._max_eval()

        def record(x, value):
            best["n"] += 1
            if value < best["f"]:
                best["f"] = value
                best["x"] = np.array(x, dtype=float, copy=True)
            self._nll_history.append(
                {"tag": tag, "eval": best["n"], "nll": value, "best": best["f"]}
            )

        def wrapped(x, grad=None):
            value = self._safe_call(objective, x)
            record(x, value)
            return value

        def wrapped_grad_scipy(x):
            value, gradient = self._safe_call_grad(objective, x)
            record(x, value)
            return value, gradient

        def wrapped_grad_nlopt(x, grad):
            value, gradient = self._safe_call_grad(objective, x)
            record(x, value)
            if grad.size > 0:
                grad[:] = gradient
            return value

        if self.options["hyper_opt"] == "Cobyla":
            constraints = []
            for i in range(x0.shape[0]):
                constraints.append(
                    {"type": "ineq", "fun": lambda t, i=i: t[i] - lower_bounds[i]}
                )
                constraints.append(
                    {"type": "ineq", "fun": lambda t, i=i: upper_bounds[i] - t[i]}
                )
            optimize.minimize(
                wrapped,
                x0,
                method="COBYLA",
                constraints=constraints,
                options={"rhobeg": 0.5, "tol": 1e-6, "maxiter": max_eval},
            )
        elif self.options["hyper_opt"] == "Cobyla-nlopt":
            nlopt = self._get_nlopt()
            opt = nlopt.opt(nlopt.LN_COBYLA, x0.shape[0])
            opt.set_lower_bounds(np.asarray(lower_bounds, dtype=float))
            opt.set_upper_bounds(np.asarray(upper_bounds, dtype=float))
            opt.set_min_objective(wrapped)
            opt.set_maxeval(max_eval)
            opt.set_xtol_rel(1e-6)
            try:
                opt.optimize(np.copy(x0))
            except Exception as err:  # nlopt.RoundoffLimited & friends
                warnings.warn(
                    f"MFCK: nlopt stopped early ({type(err).__name__}); "
                    "the best point found so far is kept.",
                    stacklevel=2,
                )
        elif self.options["hyper_opt"] == "TNC":
            optimize.minimize(
                wrapped_grad_scipy,
                x0,
                method="TNC",
                jac=True,
                bounds=list(zip(lower_bounds, upper_bounds)),
                options={"maxfun": max_eval, "gtol": 1e-8, "ftol": 1e-10},
            )
        elif self.options["hyper_opt"] == "Lbfgs-nlopt":
            nlopt = self._get_nlopt()
            opt = nlopt.opt(nlopt.LD_LBFGS, x0.shape[0])
            opt.set_lower_bounds(np.asarray(lower_bounds, dtype=float))
            opt.set_upper_bounds(np.asarray(upper_bounds, dtype=float))
            opt.set_min_objective(wrapped_grad_nlopt)
            opt.set_maxeval(max_eval)
            opt.set_xtol_rel(1e-8)
            try:
                opt.optimize(np.copy(x0))
            except Exception as err:
                warnings.warn(
                    f"MFCK: nlopt stopped early ({type(err).__name__}); "
                    "the best point found so far is kept.",
                    stacklevel=2,
                )
        else:
            raise ValueError(
                f"The optimizer {self.options['hyper_opt']} is not available"
            )

        # The returned point is the best one *seen*: this protects against
        # optimisers returning their last (not necessarily best) iterate.
        return np.clip(best["x"], lower_bounds, upper_bounds), best["f"], best["n"]

    def _optimize_block(self, objective, x0, lower_bounds, upper_bounds, tag):
        """Multi-start optimisation of an arbitrary parameter block."""
        starts = [np.asarray(x0, dtype=float)]
        n_start = max(1, int(self.options["n_start"]))
        starts += self._sample_starts(n_start - 1, lower_bounds, upper_bounds)

        best_x, best_f, n_eval = starts[0], np.inf, 0
        for j, x_start in enumerate(starts):
            x_j, f_j, n_j = self._single_run(
                objective, x_start, lower_bounds, upper_bounds, f"{tag}/start{j}"
            )
            n_eval += n_j
            if f_j < best_f:
                best_x, best_f = x_j, f_j
        return best_x, best_f, n_eval

    def _make_stage_objective(self, x_full, k):
        """
        Objective of the k-th stage of the sequential strategy: marginal
        likelihood of the sub-model (levels 0..k), seen as a function of the
        free block (sigma_k, l_k[, rho_k][, tau_k^2]) only.
        """
        noisy = self._has_noise_params()
        n_ker_full = self._n_kernel_params(self.lvl)
        n_ker_sub = self._n_kernel_params(k + 1)
        blk = self._level_block_slice(k)
        n_free_kernel = blk.stop - blk.start
        frozen_kernel = np.array(x_full[:n_ker_sub], dtype=float, copy=True)
        frozen_noise = (
            np.array(x_full[n_ker_full : n_ker_full + k], dtype=float, copy=True)
            if noisy
            else None
        )

        # position of the free entries inside the *sub-model* vector
        free_in_sub = list(range(blk.start, blk.stop))
        if noisy:
            free_in_sub.append(n_ker_sub + k)
        free_in_sub = np.asarray(free_in_sub, dtype=int)

        def objective(x_free, with_grad=False):
            kernel = frozen_kernel.copy()
            kernel[blk] = x_free[:n_free_kernel]
            if noisy:
                param = np.hstack([kernel, frozen_noise, x_free[n_free_kernel]])
            else:
                param = kernel
            if not with_grad:
                return self.neg_log_likelihood_scipy(param)
            # the gradient of the sub-model restricted to the free block
            value, gradient = self.neg_log_likelihood_scipy_grad(param)
            return value, gradient[free_in_sub]

        return objective

    def _joint_objective(self, param, with_grad=False):
        """Objective of the joint strategy: full-model likelihood."""
        if not with_grad:
            return self.neg_log_likelihood_scipy(param)
        return self.neg_log_likelihood_scipy_grad(param)

    def _sequential_optimization(self, theta_ini, lower_bounds, upper_bounds):
        """Level-wise (block-coordinate) maximum likelihood estimation."""
        x_full = np.array(theta_ini, dtype=float, copy=True)
        n_ker_full = self._n_kernel_params(self.lvl)
        noisy = self._has_noise_params()
        stages = []

        for k in range(self.lvl):
            blk = self._level_block_slice(k)
            free_idx = list(range(blk.start, blk.stop))
            if noisy:
                free_idx.append(n_ker_full + k)
            free_idx = np.asarray(free_idx, dtype=int)

            t0 = time.time()
            with self._restricted_levels(k + 1):
                objective = self._make_stage_objective(x_full, k)
                x_blk, f_k, n_eval = self._optimize_block(
                    objective,
                    x_full[free_idx],
                    lower_bounds[free_idx],
                    upper_bounds[free_idx],
                    tag=f"level{k}",
                )
            x_full[free_idx] = x_blk
            stages.append(
                {
                    "stage": f"level{k}",
                    "n_free_params": int(free_idx.size),
                    "n_eval": int(n_eval),
                    "nll": float(f_k),
                    "time": time.time() - t0,
                }
            )

        if self.options["sequential_refine"]:
            t0 = time.time()
            x_full, f_ref, n_eval = self._optimize_block(
                self._joint_objective,
                x_full,
                lower_bounds,
                upper_bounds,
                tag="refine",
            )
            stages.append(
                {
                    "stage": "refine",
                    "n_free_params": int(x_full.size),
                    "n_eval": int(n_eval),
                    "nll": float(f_ref),
                    "time": time.time() - t0,
                }
            )
        return x_full, stages

    def _fit_hyperparameters(self):
        """Estimates the hyper-parameters and stores them in optimal_theta."""
        self._nll_history = []
        self._check_gradient_support()
        theta_ini, lower_bounds, upper_bounds = self._build_hyperparameter_vectors()
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        t0 = time.time()
        if self.options["sequential_opt"] and self.lvl > 1:
            mode = "sequential"
            x_opt, stages = self._sequential_optimization(
                theta_ini, lower_bounds, upper_bounds
            )
        else:
            mode = "joint"
            x_opt, f_joint, n_eval = self._optimize_block(
                self._joint_objective,
                theta_ini,
                lower_bounds,
                upper_bounds,
                tag="joint",
            )
            stages = [
                {
                    "stage": "joint",
                    "n_free_params": int(theta_ini.size),
                    "n_eval": int(n_eval),
                    "nll": float(f_joint),
                    "time": time.time() - t0,
                }
            ]
        training_time = time.time() - t0

        self.optimal_x = x_opt  # solution in optimiser (log10) space
        self.optimal_theta = self._transform_optimizer_param(x_opt)

        self.opt_report = {
            "mode": mode,
            "stages": stages,
            "n_eval": int(sum(s["n_eval"] for s in stages)),
            "training_time": training_time,
            # likelihood of the *complete* model at the returned point: the only
            # quantity that can be compared between both strategies
            "joint_nll": self._safe_call(self.neg_log_likelihood_scipy, x_opt),
            "history": self._nll_history,
        }
        self._post_training()

    def _post_training(self):
        """Hook called once the hyper-parameters have been estimated."""
        self.noise_model = None
        if self.options["use_het_noise"] and self.options["predict_with_noise"]:
            self._fit_noise_model()

    # ------------------------------------------------------------------
    # Heteroscedastic noise: auxiliary multi-fidelity model
    # ------------------------------------------------------------------
    # With use_het_noise=True the noise variances tau_k^2(x_{k,i}) are known at
    # the training inputs only.  To obtain the noise at an arbitrary x, they are
    # treated as the observations of a *second* multi-fidelity model of the same
    # class, trained on the datasets (X_k, tau_k^2), with its own
    # hyper-parameters, its own standardisation and -- for SMFCK -- its own
    # inducing points.  Predictions then follow the usual MFCK conditioning, so
    # the noise field is available at any input and for every fidelity level,
    # and it reverts to the mean observed noise (not to zero) far from the data.

    def _noise_model_options(self):
        """
        Options of the auxiliary model fitted on the observed noise.

        Every option of the parent is inherited, including the heteroscedastic
        noise settings: the observed variances are used both as the targets and
        as the observation noise of the auxiliary model, which reproduces the
        behaviour of re-training the model by hand on (X_k, tau_k^2).  Only
        `predict_with_noise` is switched off, which stops the recursion.

        Override this method to fit a different auxiliary model, e.g. a
        homoscedastic one with an estimated noise:

            def _noise_model_options(self):
                options = super()._noise_model_options()
                options.update(use_het_noise=False, eval_noise=True,
                               noise0=[1e-6])
                return options
        """
        options = {}
        for name in self.options._declared_entries:
            if name == "predict_with_noise":
                continue
            value = self.options[name]
            if value is None:  # e.g. xlimits: let the default apply
                continue
            options[name] = value
        options["predict_with_noise"] = False
        options["print_global"] = False
        if options.get("noise_target_transform") == "log":
            # noise0 is expressed in tau^2 units: it cannot be used as the
            # observation noise of a model whose targets are log(tau^2).
            options["use_het_noise"] = False
            options["eval_noise"] = True
            options["noise0"] = [max(float(self.options["noise_bounds"][0]), 1e-6)]
        return options

    def _check_noise_targets(self):
        """Validates options['noise0'] against the training sets."""
        noise0 = self.options["noise0"]
        if len(noise0) != self.lvl:
            raise ValueError(
                "for the heteroscedastic case, noise0 must contain one array of "
                f"noise variances per fidelity level ({self.lvl} expected, "
                f"{len(noise0)} given)"
            )
        targets = []
        for k, values in enumerate(noise0):
            values = np.asarray(values, dtype=float).reshape(-1, 1)
            if values.shape[0] != self.X[k].shape[0]:
                raise ValueError(
                    f"noise0[{k}] has {values.shape[0]} entries but level {k} "
                    f"has {self.X[k].shape[0]} training points"
                )
            targets.append(values)
        return targets

    def _fit_noise_model(self):
        """
        Trains the auxiliary model on the observed noise variances.  Called
        once, at the end of train(); the fitted model is stored in
        `self.noise_model`.
        """
        if not self.options["use_het_noise"]:
            raise ValueError(
                "the auxiliary noise model requires use_het_noise=True"
            )
        targets = self._check_noise_targets()
        if self.options["noise_target_transform"] == "log":
            floor = np.finfo(float).tiny
            targets = [np.log(np.maximum(t, floor)) for t in targets]

        model = type(self)()
        for name, value in self._noise_model_options().items():
            try:
                model.options[name] = value
            except Exception:  # pragma: no cover - option not transferable
                continue
        for k in range(self.lvl - 1):
            model.set_training_values(self.X[k], targets[k], name=k)
        model.set_training_values(self.X[-1], targets[-1])
        model.train()
        self.noise_model = model
        return model

    def predict_noise_all_levels(self, x):
        """
        Predicts the heteroscedastic noise variance of every fidelity level at
        the inputs x, using the auxiliary model fitted on the observed noise.

        Returns
        -------
        list of np.ndarray [n_evals, 1]
        """
        if self.noise_model is None:
            self._fit_noise_model()
        means, _ = self.noise_model.predict_all_levels(x)
        floor = max(float(self.options["noise_bounds"][0]), np.finfo(float).tiny)
        log_targets = self.options["noise_target_transform"] == "log"
        predictions = []
        for mean in means:
            mean = np.asarray(mean, dtype=float).reshape(-1, 1)
            if log_targets:
                # exp(mu) is the median of the log-normal posterior.  The mean,
                # exp(mu + s^2/2), is also defensible but inflates by orders of
                # magnitude wherever the auxiliary posterior is wide (i.e. far
                # from the data), which makes it useless as a plug-in noise.
                predictions.append(np.exp(mean))
            else:
                # a GP fitted on raw variances can undershoot below zero
                predictions.append(np.clip(mean, floor, np.inf))
        return predictions

    def predict_noise(self, x, level=None):
        """
        Predicted noise variance at the inputs x for one fidelity level
        (the highest one by default).

        Returns
        -------
        np.ndarray [n_evals]
        """
        index = self.lvl - 1 if level is None else int(level)
        return self.predict_noise_all_levels(x)[index].ravel()

    def _predicted_noise_or_none(self, x):
        """Noise field to be added to the predictive variances, if requested."""
        if self.options["use_het_noise"] and self.options["predict_with_noise"]:
            return self.predict_noise_all_levels(x)
        return None

    # ------------------------------------------------------------------
    # Covariance assembly
    # ------------------------------------------------------------------
    def _get_kernel_params(self):
        """
        Returns the appropriate kernel parameters based on noise options.
        Returns
        -------
        np.ndarray
            Kernel parameters (excludes noise parameters if eval_noise and not
            use_het_noise)
        """
        if self._has_noise_params():
            return self.optimal_theta[: -self.lvl]
        else:
            return self.optimal_theta

    def _compute_K_and_cholesky(self):
        """
        Computes the blockwise covariance matrix K and its Cholesky decomposition
        with noise handling based on options.
        Returns
        -------
        tuple
            (K matrix, Cholesky decomposition L, noise parameters if applicable)
        """
        kernel_params = self._get_kernel_params()

        self.K = self.compute_blockwise_K(
            self.X_norma_all, self.X_norma_all, kernel_params
        )

        if self.options["eval_noise"] or self.options["use_het_noise"]:
            if self.options["use_het_noise"]:
                noise_matrix = self._het_noise_vector() * np.eye(self.K.shape[0])
                noises = None
            else:
                noises = self.optimal_theta[-self.lvl : :]
                varis = []
                for i, v in enumerate(noises):
                    varis = np.hstack([varis, np.full(self.X[i].shape[0], noises[i])])
                noise_matrix = varis * np.eye(self.K.shape[0])
            L = np.linalg.cholesky(
                self.K + noise_matrix + self.options["nugget"] * np.eye(self.K.shape[0])
            )
            return self.K, L, noises
        else:
            L = np.linalg.cholesky(
                self.K + self.options["nugget"] * np.eye(self.K.shape[0])
            )
            return self.K, L, None

    def _compute_cross_covariance_list(self, x, ind):
        """
        Computes the list of cross-covariance matrices between training data
        and evaluation point(s), with noise parameter handling.
        Parameters
        ----------
        x : np.ndarray
            Evaluation point(s)
        ind : int
            Level index for the evaluation point(s)
        Returns
        -------
        list
            List of cross-covariance matrices
        """
        kernel_params = self._get_kernel_params()
        k_xX = []

        for j in range(self.lvl):
            if ind >= j:
                k_xX.append(
                    self.compute_cross_K(self.X_norma_all[j], x, ind, j, kernel_params)
                )
            else:
                k_xX.append(
                    self.compute_cross_K(self.X_norma_all[j], x, j, ind, kernel_params)
                )

        return k_xX

    def eta(self, j, jp, rho):
        """Compute eta_{j,l} based on the given rho values."""
        if j < jp:
            return np.prod(rho[j:jp])  # Product of rho[j+1] to rho[l]
        elif j == jp:
            return 1
        else:
            raise ValueError(
                f"The iterative variable j={j} cannot be greater than j'={jp}"
            )

    # Covariance between y_l(x) and y_l'(x')
    def compute_cross_K(self, x, xp, L, Lp, param):
        """
        Calculation Cov(y_l(x), y_{l'}(x')) using the autoregressive formulation.
        Parmeters:
        - x: First input for the covariannce (np.ndarray)
        - xp: Second input for the covariannce (np.ndarray)
        - L: Level index of the first output (scalar)
        - Lp: Level index of the second output (scalar)
        - param: Set of Hyper-parameters (vector)
        Returns:
        - Covariance matrix cov(y_l(x), y_{l'}(x')) (np.ndarray)
        """
        cov_value = 0.0

        sigma_0 = param[0]
        l_0 = param[1 : self.nx + 1]
        sigmas_gamma = param[self.nx + 1 :: self.nx + 2]
        l_s = [
            param[i : i + self.nx].tolist()
            for i in np.arange(self.nx + 2, param.shape[0] - 1, self.nx + 2)
        ]
        rho_values = param[2 + 2 * self.nx :: self.nx + 2]

        # Sum of j=0 until l_^prime
        for j in range(Lp + 1):
            eta_j_l = self.eta(j, L, rho_values)
            eta_j_lp = self.eta(j, Lp, rho_values)

            if j == 0:
                # Cov(gamma_j(x), gamma_j(x')) using the kernel for K_00
                cov_gamma_j = self._compute_K(x, xp, [sigma_0, l_0])
            else:
                # Cov(gamma_j(x), gamma_j(x')) using the kernel
                cov_gamma_j = self._compute_K(x, xp, [sigmas_gamma[j - 1], l_s[j - 1]])
            # Add to the value of the covariance
            cov_value += eta_j_l * eta_j_lp * cov_gamma_j

        return cov_value

    def compute_diag_K(self, x, xp, L, Lp, param):
        """
        Calculation of the diagonal of K using the autoregressive formulation.
        Parmeters:
        - x: First input for the covariannce (np.ndarray)
        - xp: Second input for the covariannce (np.ndarray)
        - L: Level index of the first output (scalar)
        - Lp: Level index of the second output (scalar)
        - param: Set of Hyper-parameters (vector)
        Returns:
        - Diagonal variance for cov(y_l(x), y_{l'}(x')) (np.ndarray)
        """
        v_value = 0.0

        sigma_0 = param[0]
        sigmas_gamma = param[self.nx + 1 :: self.nx + 2]
        rho_values = param[2 + 2 * self.nx :: self.nx + 2]

        # Sum of j=0 until l_^prime
        for j in range(Lp + 1):
            eta_j_l = self.eta(j, L, rho_values)
            eta_j_lp = self.eta(j, Lp, rho_values)

            if j == 0:
                variance = np.full(x.shape[0], sigma_0)
            else:
                variance = np.full(x.shape[0], sigmas_gamma[j - 1])
            v_value += eta_j_l * eta_j_lp * variance

        return v_value

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    def predict_all_levels(self, x):
        """
        Generalized prediction function for the multi-fidelity co-Kriging
        Parameters
        ----------
        x : np.ndarray
            Array with the inputs for make the prediction.
        Returns
        -------
        means : (list, np.array)
            Returns the conditional means per level.
        covariances: (list, np.array)
            Returns the conditional variances per level (original output scale).
        """
        means = []
        covariances = []
        noise_pred = self._predicted_noise_or_none(x)
        x = (x - self.X_offset) / self.X_scale

        _, L, noises = self._compute_K_and_cholesky()
        alpha1 = solve_triangular(L, self.y_norma_all, lower=True)
        kernel_params = self._get_kernel_params()

        for ind in range(self.lvl):
            k_xx = self.compute_diag_K(x, x, ind, ind, kernel_params)
            k_xX = self._compute_cross_covariance_list(x, ind)

            beta1 = solve_triangular(L, np.vstack(k_xX), lower=True)
            means.append(self.y_std * np.dot(beta1.T, alpha1) + self.y_mean)

            var = k_xx - np.sum(beta1**2, axis=0)
            if noises is not None:
                var = var + noises[ind]
            var = var * self.y_std**2
            if noise_pred is not None:
                var = var.reshape(-1, 1) + noise_pred[ind]
            covariances.append(var)

        return means, covariances

    def predict_values(self, x, is_acting=None):
        """
        Prediction function for the highest fidelity level
        Parameters
        ----------
        x : array
            Array with the inputs for make the prediction.
        Returns
        -------
        mean : np.array
            Conditional mean of the highest fidelity level.
        """
        x = (x - self.X_offset) / self.X_scale
        _, L, _ = self._compute_K_and_cholesky()

        ind = self.lvl - 1
        k_xX = self._compute_cross_covariance_list(x, ind)

        beta1 = solve_triangular(L, np.vstack(k_xX), lower=True)
        alpha1 = solve_triangular(L, self.y_norma_all, lower=True)

        return self.y_std * np.dot(beta1.T, alpha1) + self.y_mean

    def predict_variances(
        self, X: np.ndarray, is_acting=None, is_ri=False
    ) -> np.ndarray:
        """
        Evaluates the variance of the highest fidelity level at a set of points.

        Arguments
        ---------
        X : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        variance : np.ndarray
            Prediction variance, in the original (unnormalized) output scale.
        """
        noise_pred = self._predicted_noise_or_none(X)
        X = (X - self.X_offset) / self.X_scale
        _, L, noises = self._compute_K_and_cholesky()

        kernel_params = self._get_kernel_params()
        ind = self.lvl - 1
        k_xx = self.compute_diag_K(X, X, ind, ind, kernel_params)
        k_xX = self._compute_cross_covariance_list(X, ind)

        beta1 = solve_triangular(L, np.vstack(k_xX), lower=True)

        variance = k_xx - np.sum(beta1**2, axis=0)
        if noises is not None:
            variance = variance + noises[ind]

        # same scaling convention as predict_variances_all_levels
        variance = variance * self.y_std**2
        if noise_pred is not None:
            variance = variance.reshape(-1, 1) + noise_pred[ind]
        return variance

    def predict_variances_all_levels(self, x):
        """
        Evaluates the variance of every fidelity level at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        MSE : np.ndarray [n_evals, n_levels]
        """
        noise_pred = self._predicted_noise_or_none(x)
        x = (x - self.X_offset) / self.X_scale
        _, L, noises = self._compute_K_and_cholesky()

        kernel_params = self._get_kernel_params()
        MSE = np.zeros((x.shape[0], self.lvl))

        for ind in range(self.lvl):
            k_xx = self.compute_diag_K(x, x, ind, ind, kernel_params)
            k_xX = self._compute_cross_covariance_list(x, ind)

            beta1 = solve_triangular(L, np.vstack(k_xX), lower=True)

            var = k_xx - np.sum(beta1**2, axis=0)
            if noises is not None:
                var = var + noises[ind]
            MSE[:, ind] = var

        MSE *= self.y_std**2
        if noise_pred is not None:
            MSE += np.hstack([n.reshape(-1, 1) for n in noise_pred])
        return MSE

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------
    def neg_log_likelihood(self, param, grad=None):
        """
        Negative marginal log-likelihood of the (possibly restricted) MFCK
        model, for a parameter vector expressed in model space.
        """
        param = np.asarray(param, dtype=float)
        reg_term = self.options["lambda"] * np.sum(np.power(param, 2))
        nugget = self.options["nugget"]

        if self.options["eval_noise"] or self.options["use_het_noise"]:
            if self.options["use_het_noise"]:
                self.K = self.compute_blockwise_K(
                    self.X_norma_all, self.X_norma_all, param
                )
                noise_matrix = self._het_noise_vector() * np.eye(self.K.shape[0])
            else:
                self.K = self.compute_blockwise_K(
                    self.X_norma_all, self.X_norma_all, param[: -self.lvl]
                )
                noises = param[-self.lvl : :]
                varis = []
                for i, v in enumerate(noises):
                    varis = np.hstack(
                        [varis, np.full(self.X_norma_all[i].shape[0], noises[i])]
                    )
                noise_matrix = varis * np.eye(self.K.shape[0])
            L = np.linalg.cholesky(
                self.K + noise_matrix + nugget * np.eye(self.K.shape[0])
            )
        else:
            self.K = self.compute_blockwise_K(self.X_norma_all, self.X_norma_all, param)
            L = np.linalg.cholesky(self.K + nugget * np.eye(self.K.shape[0]))

        beta = solve_triangular(L, self.y_norma_all, lower=True)
        NMLL = 2 * np.sum(np.log(np.diag(L))) + np.dot(beta.T, beta) + reg_term
        return float(np.asarray(NMLL).ravel()[0])

    def _transform_optimizer_param(self, param):
        """Map optimizer parameters from log10-space to model-space."""
        param = np.array(param, dtype=float, copy=True)

        if self._has_noise_params():
            kernel_param = np.array(param[: -self.lvl], copy=True)
        else:
            kernel_param = np.array(param, copy=True)

        kernel_param[0] = 10 ** kernel_param[0]
        kernel_param[1 : self.nx + 1] = 10 ** kernel_param[1 : self.nx + 1]
        kernel_param[self.nx + 1 :: self.nx + 2] = (
            10 ** kernel_param[self.nx + 1 :: self.nx + 2]
        )

        for i in np.arange(self.nx + 2, kernel_param.shape[0] - 1, self.nx + 2):
            kernel_param[i : i + self.nx] = 10 ** kernel_param[i : i + self.nx]

        if self._has_noise_params():
            param[-self.lvl :] = 10 ** param[-self.lvl :]
            param[: -self.lvl] = kernel_param
            return param

        return kernel_param

    # ------------------------------------------------------------------
    # Analytical gradients of the marginal log-likelihood
    # ------------------------------------------------------------------
    # The block-wise covariance can be written, over the concatenated design
    # X = [X_0; ...; X_L] with level index kappa(a) of the a-th row, as
    #
    #     K = sum_j  sigma_j^2 (e_j e_j^T) o R_j ,      e_j[a] = eta_{j,kappa(a)}
    #
    # (Hadamard product, eta_{j,kappa} := 0 for j > kappa), which is exactly
    # Eq. (10) written as a sum over the latent processes gamma_j.  Hence
    #
    #   dK/d sigma_j^2  = (e_j e_j^T) o R_j
    #   dK/d l_{j,d}    = sigma_j^2 (e_j e_j^T) o (dR_j / d l_{j,d})
    #   dK/d rho_m      = sum_j sigma_j^2 (g_{j,m} e_j^T + e_j g_{j,m}^T) o R_j
    #   dK/d tau_k^2    = diag(1{kappa(a) = k})
    #
    # with g_{j,m}[a] = d eta_{j,kappa(a)} / d rho_m as given in the appendix.
    # With L = log|K| + y^T K^-1 y and W = K^-1 - alpha alpha^T (alpha = K^-1 y),
    # dL/dtheta_i = sum(W o dK/dtheta_i), so every entry reduces to a quadratic
    # form and no n x n derivative matrix is ever stored.

    def _check_gradient_support(self):
        """Validates the optimiser / kernel combination before training."""
        if self.options["hyper_opt"] not in self._GRADIENT_OPT:
            return
        if not self._supports_gradient:
            raise ValueError(
                f"{self.name} has no analytical likelihood gradient; use "
                "hyper_opt='Cobyla' or 'Cobyla-nlopt'."
            )
        if self.options["corr"] not in self._GRADIENT_KERNELS:
            raise ValueError(
                f"The gradient of the '{self.options['corr']}' kernel is not "
                f"available; supported kernels are {self._GRADIENT_KERNELS}."
            )

    def _param_indices(self, k):
        """(index of sigma_k, slice of l_k, index of rho_k) in the parameters."""
        nx = self.nx
        if k == 0:
            return 0, slice(1, nx + 1), None
        start = (nx + 1) + (k - 1) * (nx + 2)
        return start, slice(start + 1, start + 1 + nx), start + nx + 1

    def _unpack_kernel_param(self, kernel_param):
        """Splits the kernel parameters into (sigmas, length-scales, rhos)."""
        sigmas, thetas, rhos = [], [], []
        for k in range(self.lvl):
            i_sigma, sl_theta, i_rho = self._param_indices(k)
            sigmas.append(float(kernel_param[i_sigma]))
            thetas.append(np.asarray(kernel_param[sl_theta], dtype=float))
            if i_rho is not None:
                rhos.append(float(kernel_param[i_rho]))
        return sigmas, thetas, np.asarray(rhos, dtype=float)

    def _level_index_vector(self):
        """Fidelity level of every row of the concatenated training set."""
        return np.concatenate(
            [np.full(x.shape[0], k, dtype=int) for k, x in enumerate(self.X_norma_all)]
        )

    def _eta_matrix(self, rho):
        """eta[j, kappa] of Eq. (5), with the convention eta = 0 for j > kappa."""
        eta = np.zeros((self.lvl, self.lvl))
        for j in range(self.lvl):
            for kappa in range(j, self.lvl):
                eta[j, kappa] = 1.0 if kappa == j else float(np.prod(rho[j:kappa]))
        return eta

    def _eta_grad_matrix(self, rho, m):
        """d eta[j, kappa] / d rho_{m+1} (m is the 0-based index in `rho`)."""
        deta = np.zeros((self.lvl, self.lvl))
        for j in range(self.lvl):
            for kappa in range(j + 1, self.lvl):
                if j <= m < kappa:
                    # product over {j, ..., kappa-1} \ {m}: never divide by rho_m
                    keep = [i for i in range(j, kappa) if i != m]
                    deta[j, kappa] = float(np.prod(rho[keep])) if keep else 1.0
        return deta

    def _componentwise_distance(self, A, B):
        """Componentwise correlation distances; independent of the kernel params."""
        dx = differences(A, B)
        d = componentwise_distance(
            dx,
            self.options["corr"],
            self.nx,
            power=self._pow_exp_power,
        )
        del dx
        return d

    def _corr_from_distance(self, d, theta, shape, grad_ind=None):
        """R (or dR/dtheta_{grad_ind}) from precomputed distances."""
        self.corr.theta = np.asarray(theta, dtype=float)
        return self.corr(d, grad_ind=grad_ind).reshape(shape)

    def neg_log_likelihood_grad(self, param):
        """
        Negative marginal log-likelihood and its analytical gradient with
        respect to the model-space parameters
        theta = ((sigma_k^2, l_k)_k, (rho_k)_k, (tau_k^2)_k).

        Returns
        -------
        (float, np.ndarray)
        """
        if not self._supports_gradient:
            raise NotImplementedError(
                f"{self.name} does not provide the likelihood gradient"
            )
        param = np.asarray(param, dtype=float)
        lvl, nx = self.lvl, self.nx
        noisy = self._has_noise_params()
        kernel_param = param[: -lvl] if noisy else param
        sigmas, thetas, rho = self._unpack_kernel_param(kernel_param)

        Xall = np.vstack(self.X_norma_all)
        n = Xall.shape[0]
        lev = self._level_index_vector()
        eta = self._eta_matrix(rho)
        e = [eta[j, lev] for j in range(lvl)]

        # --- covariance matrix, latent process by latent process
        d = self._componentwise_distance(Xall, Xall)
        R_list = []
        K = np.zeros((n, n))
        for j in range(lvl):
            R = self._corr_from_distance(d, thetas[j], (n, n))
            R_list.append(R)
            K += sigmas[j] * np.outer(e[j], e[j]) * R

        # --- noise / nugget
        if noisy:
            varis = np.asarray(param[-lvl:], dtype=float)[lev]
        elif self.options["use_het_noise"]:
            varis = self._het_noise_vector()
        else:
            varis = np.zeros(n)
        Kt = K + np.diag(varis + self.options["nugget"])

        chol = np.linalg.cholesky(Kt)
        y = self.y_norma_all
        alpha = cho_solve((chol, True), y)
        reg_term = self.options["lambda"] * np.sum(np.power(param, 2))
        nll = float(
            2.0 * np.sum(np.log(np.diag(chol))) + (y.T @ alpha).item() + reg_term
        )

        # --- W = K^-1 - alpha alpha^T
        W = cho_solve((chol, True), np.eye(n)) - alpha @ alpha.T

        grad = np.zeros_like(param)
        deta_list = [self._eta_grad_matrix(rho, m) for m in range(lvl - 1)]
        for j in range(lvl):
            i_sigma, sl_theta, _ = self._param_indices(j)
            M = W * R_list[j]
            Me = M @ e[j]

            # sigma_j^2
            grad[i_sigma] = float(e[j] @ Me)

            # length-scales of gamma_j
            for i_dim in range(nx):
                dR = self._corr_from_distance(
                    d, thetas[j], (n, n), grad_ind=i_dim
                )
                grad[sl_theta.start + i_dim] = sigmas[j] * float(
                    e[j] @ ((W * dR) @ e[j])
                )

            # rho_m: (g e^T + e g^T) o R contributes 2 g^T (W o R) e by symmetry
            for m in range(lvl - 1):
                g = deta_list[m][j, lev]
                if np.any(g):
                    i_rho = self._param_indices(m + 1)[2]
                    grad[i_rho] += 2.0 * sigmas[j] * float(g @ Me)

        # --- noise variances
        if noisy:
            w_diag = np.diag(W)
            n_kernel = kernel_param.shape[0]
            for k in range(lvl):
                grad[n_kernel + k] = float(np.sum(w_diag[lev == k]))

        grad += 2.0 * self.options["lambda"] * param
        return nll, grad

    def _log_scale_mask(self):
        """True for the parameters stored in log10 scale (everything but rho)."""
        size = self._n_kernel_params() + (self.lvl if self._has_noise_params() else 0)
        mask = np.ones(size, dtype=bool)
        for k in range(1, self.lvl):
            mask[self._param_indices(k)[2]] = False
        return mask

    def neg_log_likelihood_scipy_grad(self, param):
        """
        Likelihood and its gradient in optimiser space.  The chain rule for the
        log10 parametrisation p = 10^u gives dL/du = ln(10) p dL/dp; the rho's
        are optimised in linear scale and are left untouched.
        """
        model_param = self._transform_optimizer_param(param)
        nll, grad = self.neg_log_likelihood_grad(model_param)
        scale = np.ones_like(model_param)
        mask = self._log_scale_mask()
        scale[mask] = model_param[mask] * np.log(10.0)
        return nll, grad * scale

    def neg_log_likelihood_nlopt_grad(self, param, grad=None):
        """Likelihood and gradient for the nlopt gradient-based optimisers."""
        value, gradient = self.neg_log_likelihood_scipy_grad(param)
        if grad is not None and np.size(grad) > 0:
            grad[:] = gradient
        return value

    def neg_log_likelihood_scipy(self, param):
        """
        Likelihood for Cobyla-scipy (SMT) optimizer
        """
        return self.neg_log_likelihood(self._transform_optimizer_param(param))

    def neg_log_likelihood_nlopt(self, param, grad=None):
        """
        Likelihood for nlopt optimizers
        """
        return self.neg_log_likelihood(self._transform_optimizer_param(param), grad)

    def compute_blockwise_K(self, X, Xprime, param):
        K_block = {}
        n = 0
        nprime = 0
        for i in X:
            n = n + i.shape[0]

        for i in Xprime:
            nprime = nprime + i.shape[0]
        for jp in range(self.lvl):
            for j in range(self.lvl):
                if jp >= j:
                    K_block[(jp, j)] = self.compute_cross_K(
                        X[j], Xprime[jp], jp, j, param
                    )
                else:
                    K_block[(jp, j)] = self.compute_cross_K(
                        X[j], Xprime[jp], j, jp, param
                    )
        K = np.zeros((n, nprime))
        row_init, col_init = 0, 0
        for j in range(self.lvl):
            col_init = 0
            for jp in range(self.lvl):
                r, c = K_block[(jp, j)].shape
                K[row_init : row_init + r, col_init : col_init + c] = K_block[(jp, j)]
                col_init += c
            row_init += r

        return K

    def _compute_K(self, A: np.ndarray, B: np.ndarray, param):
        """
        Compute the covariance matrix K between A and B
            Modified for MFCK
        """
        # Compute pairwise componentwise L1-distances between A and B
        dx = differences(A, B)
        d = componentwise_distance(
            dx,
            self.options["corr"],
            self.X[0].shape[1],
            # _pow_exp_power is the exponent associated with the selected
            # kernel (2 for squar_exp, 1 for abs_exp/matern), while
            # options["pow_exp_power"] is only the pow_exp default (1.9)
            power=self._pow_exp_power,
        )
        self.corr.theta = np.asarray(param[1])
        r = self.corr(d)
        R = r.reshape(A.shape[0], B.shape[0])
        K = param[0] * R
        return K