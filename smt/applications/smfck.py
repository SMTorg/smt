# -*- coding: utf-8 -*-
"""
Created on Sat May 04 10:10:12 2024

@author: Mauricio Castano Aguirre <mauricio.castano_aguirre@onera.fr>
Sparse Multi-Fidelity co-Kriging (SMFCK) model construction for non-nested
experimental design sets.
-------
[1] Loic Le Gratiet (2013). Multi-fidelity Gaussian process modelling
[Doctoral Thesis, Universite Paris-Sud].
[2] Edwin V. Bonilla, Kian Ming A. Chai, and Christopher K. I. Williams
(2007). Multi-task Gaussian Process prediction. In International
Conference on Neural Information Processing Systems.
[3] Snelson & Ghahramani (2005). Sparse Gaussian processes using
pseudo-inputs (FITC).
[4] Titsias (2009). Variational learning of inducing variables in sparse
Gaussian processes (VFE).

Hyper-parameter estimation
--------------------------
SMFCK inherits the parameter layout and the two estimation strategies of
MFCK (see smt/applications/mfck.py):

* ``sequential_opt=False``: joint optimisation of the FITC / VFE objective
  over all the fidelity levels at once.
* ``sequential_opt=True``: block-coordinate ascent, stage k optimising
  (sigma_k, l_k, rho_k, tau_k^2) on the *sparse* marginal likelihood of the
  sub-model made of levels 0..k, i.e. with the inducing sets Z_0..Z_k only.
  Stage k therefore manipulates matrices of size (M_0+...+M_k), so the early
  stages are extremely cheap.
"""

import warnings

import numpy as np
from scipy.cluster.vq import kmeans
from scipy.linalg import solve_triangular

from smt.applications.mfck import MFCK
from smt.utils.misc import standardization


class SMFCK(MFCK):
    # a single COBYLA run is cheaper here than for the exact model
    _DEFAULT_MAX_EVAL = {"Cobyla": 50, "Cobyla-nlopt": 80}
    # the gradients of the FITC / VFE bounds are not derived yet: the
    # gradient-based optimisers of MFCK are refused with an explicit message
    _supports_gradient = False

    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        self.name = "SMFCK"
        declare(
            "n_inducing",
            [6, 5],
            types=(list, np.ndarray),
            desc="Number of inducing points per fidelity level",
        )
        declare(
            "method",
            "FITC",
            values=("FITC", "VFE"),
            desc="Methods available for Sparse Multi-fidelity",
        )
        declare(
            "inducing_method",
            "kmeans",
            types=str,
            values=["random", "kmeans"],
            desc="The chosen method to induce points",
        )

        self.options["rho0"] = 1.0
        self.options["rho_bounds"] = [-5.0, 5.0]
        self.options["sigma0"] = 1.0
        self.options["sigma_bounds"] = [1e-6, 100]
        self.options["lambda"] = 0.0
        # The FITC/VFE objectives are built on a Gaussian likelihood: the noise
        # variances are structural parameters of the sparse model and must be
        # part of the optimisation (or provided through use_het_noise).
        self.options["eval_noise"] = True
        self.options["use_het_noise"] = False
        self.options["seed"] = 0
        self.options["hyper_opt"] = (
            "Cobyla-nlopt"  # MFCK doesn't support gradient-based optimizers
        )
        self.options["nugget"] = 1000.0 * np.finfo(np.double).eps
        self.woodbury_data = {"vec": None, "inv": None}
        self._seed = self.options["seed"]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _compute_inducing_points(self, xt):
        """Inducing point locations, one set per fidelity level."""
        n_inducing = np.asarray(self.options["n_inducing"]).ravel()
        # Reset the k-means seed at every call: self._seed used to keep
        # incrementing across calls, so training the *same* model twice on the
        # same data produced different inducing points (and therefore
        # different hyper-parameters).
        self._seed = self.options["seed"]
        zt = []
        for i, x in enumerate(xt):
            n_ind = int(n_inducing[i])
            if n_ind > x.shape[0]:
                warnings.warn(
                    f"SMFCK: n_inducing[{i}]={n_ind} is larger than the number "
                    f"of training points ({x.shape[0]}) at level {i}; it is "
                    "truncated.",
                    stacklevel=2,
                )
                n_ind = x.shape[0]
            if self.options["inducing_method"] == "random":
                idx = self.rng.permutation(x.shape[0])[:n_ind]
                zt.append(np.atleast_2d(x[idx]))
            else:  # kmeans
                if self._seed is not None:
                    self._seed += 1
                zt.append(np.atleast_2d(kmeans(x, n_ind, rng=self._seed)[0]))
        return zt

    def train(self):
        """
        Overrides MFK implementation
        Trains the Sparse Multi-Fidelity co-Kriging model
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
        if np.shape(self.options["n_inducing"])[0] != self.lvl:
            raise ValueError(
                f"n_inducing {self.options['n_inducing']} don't correspond to "
                "the fidelities"
            )

        self.y = np.vstack(yt)
        self._check_param_het_safe()

        if not self.options["use_het_noise"] and not self.options["eval_noise"]:
            warnings.warn(
                "SMFCK: the FITC/VFE marginal likelihoods require a noise term; "
                "'eval_noise' has been switched back on.",
                stacklevel=2,
            )
            self.options["eval_noise"] = True

        (
            _,
            _,
            self.X_offset,
            self.y_mean,
            self.X_scale,
            self.y_std,
        ) = standardization(np.concatenate(xt, axis=0), np.concatenate(yt, axis=0))

        self.Z = self._compute_inducing_points(xt)
        self.X_norma_all = [(x - self.X_offset) / self.X_scale for x in xt]
        self.Z_norma_all = [(z - self.X_offset) / self.X_scale for z in self.Z]
        self.y_norma_all = np.vstack([(f - self.y_mean) / self.y_std for f in yt])

        self._fit_hyperparameters()

    def _post_training(self):
        """
        Re-evaluates the sparse likelihood at the optimum so that the Woodbury
        terms used for prediction match `optimal_theta`.  Without this, they
        would correspond to the *last* likelihood evaluation performed by the
        optimiser (and, in sequential mode, to a sub-model).  The auxiliary
        noise model (heteroscedastic case) is then fitted by MFCK.
        """
        self.neg_log_likelihood(self.optimal_theta)
        super()._post_training()

    # ------------------------------------------------------------------
    # Sparse likelihoods
    # ------------------------------------------------------------------
    def _split_noise(self, param, X):
        """Returns (kernel_param, noise vector over all the training points)."""
        if self.options["use_het_noise"]:
            return param, self._het_noise_vector()
        noises = param[-self.lvl : :]
        varis = np.hstack(
            [np.full(X[i].shape[0], noises[i]) for i in range(self.lvl)]
        )
        return param[: -self.lvl], varis

    def _sparse_likelihood(self, X, Y, Z, param, method):
        """
        Negative FITC / VFE marginal log-likelihood together with the Woodbury
        terms used for prediction.

        FITC : p(y) = N(0, Q + diag(Kff - Q) + D)
        VFE  : L    = log N(0, Q + D) - 1/2 tr(D^{-1} [Kff - Q])
        """
        kernel_param, varis = self._split_noise(param, X)
        Y = np.asarray(Y, dtype=float).reshape(-1, 1)

        Kdiag = np.concatenate(
            [self.compute_diag_K(X[i], X[i], i, i, kernel_param) for i in range(self.lvl)]
        )
        Kmm = self.compute_blockwise_K(Z, Z, kernel_param)
        Knm = self.compute_blockwise_K(X, Z, kernel_param)

        nugget = self.options["nugget"]
        U = np.linalg.cholesky(Kmm + np.eye(Kmm.shape[0]) * nugget)
        # V = U^{-1} Kmn (solve instead of an explicit inverse)
        V = solve_triangular(U, Knm.T, lower=True)
        Qdiag = np.sum(np.square(V), 0)

        if method == "FITC":
            nu = Kdiag - Qdiag + varis
            trace_term = 0.0
        elif method == "VFE":
            nu = np.array(varis, dtype=float, copy=True)
            trace_term = np.sum((Kdiag - Qdiag) / varis)
        else:
            raise ValueError(f"Unknown sparse method '{method}' (FITC or VFE)")

        # numerical guard: Kdiag - Qdiag is >= 0 in exact arithmetic only
        nu = np.maximum(nu, 1e-12)
        beta = 1.0 / nu

        A = np.eye(Kmm.shape[0]) + (V * beta) @ V.T
        L = np.linalg.cholesky(A + np.eye(A.shape[0]) * nugget)
        a = Y * beta[:, None]
        b = solve_triangular(L, V @ a, lower=True)

        likelihood = 0.5 * (
            np.sum(np.log(nu))
            + 2.0 * np.sum(np.log(np.diag(L)))
            + (a.T @ Y).item()
            - float(np.einsum("ij,ij->", b, b))
            + trace_term
        )

        eye_m = np.eye(Kmm.shape[0])
        Ui = solve_triangular(U, eye_m, lower=True)
        Li = solve_triangular(L, eye_m, lower=True)
        LiUi = Li @ Ui
        woodbury_vec = LiUi.T @ b
        woodbury_inv = Ui.T @ Ui - LiUi.T @ LiUi

        return float(likelihood), woodbury_vec, woodbury_inv

    def neg_log_likelihood(self, param, grad=None):
        likelihood, w_vec, w_inv = self._sparse_likelihood(
            self.X_norma_all,
            self.y_norma_all,
            self.Z_norma_all,
            np.asarray(param, dtype=float),
            self.options["method"],
        )
        self.woodbury_data["vec"] = w_vec
        self.woodbury_data["inv"] = w_inv
        return likelihood

    def neg_log_likelihood_grad(self, param):
        raise NotImplementedError(
            "SMFCK: the analytical gradients of the FITC/VFE marginal "
            "likelihoods are not implemented; train with hyper_opt='Cobyla' "
            "or 'Cobyla-nlopt'."
        )

    # kept for backward compatibility
    def _FITC(self, X, Y, Z, param):
        return self._sparse_likelihood(X, Y, Z, param, "FITC")

    def _VFE(self, X, Y, Z, param):
        return self._sparse_likelihood(X, Y, Z, param, "VFE")

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    def _cross_covariance_inducing(self, x, ind, kernel_param):
        """[k_{ind,0}(x, Z_0), ..., k_{ind,L}(x, Z_L)] stacked row-wise."""
        k_xZ = []
        for j in range(self.lvl):
            if ind >= j:
                k_xZ.append(
                    self.compute_cross_K(self.Z_norma_all[j], x, ind, j, kernel_param)
                )
            else:
                k_xZ.append(
                    self.compute_cross_K(self.Z_norma_all[j], x, j, ind, kernel_param)
                )
        return np.vstack(k_xZ)

    def predict_all_levels(self, x):
        """
        Generalized prediction function for the sparse multi-fidelity co-Kriging
        Parameters
        ----------
        x : np.ndarray
            Array with the inputs for make the prediction.
        Returns
        -------
        means : (list, np.array)
            Conditional means per level.
        covariances: (list, np.array)
            Conditional variances per level (original output scale).
        """
        if self.woodbury_data["vec"] is None:
            raise RuntimeError("SMFCK: the model must be trained before predicting")

        means = []
        covariances = []
        noise_pred = self._predicted_noise_or_none(x)
        x = (x - self.X_offset) / self.X_scale

        if self.options["use_het_noise"]:
            kernel_param = self.optimal_theta
            noises = None
        else:
            kernel_param = self.optimal_theta[: -self.lvl]
            noises = self.optimal_theta[-self.lvl : :]

        for ind in range(self.lvl):
            k_xx = self.compute_diag_K(x, x, ind, ind, kernel_param)
            k_xZ = self._cross_covariance_inducing(x, ind, kernel_param)

            means.append(
                self.y_std * (k_xZ.T @ self.woodbury_data["vec"]) + self.y_mean
            )

            val = np.sum(np.dot(self.woodbury_data["inv"].T, k_xZ) * k_xZ, 0)
            if noises is None:
                var = (k_xx - val)[:, None]
            else:
                var = (k_xx + noises[ind] - val)[:, None]

            var = np.clip(var, 1e-15, np.inf) * self.y_std**2
            if noise_pred is not None:
                var = var + noise_pred[ind]
            covariances.append(var)

        return means, covariances

    def predict_values(self, x, is_acting=None):
        """Conditional mean of the highest fidelity level (sparse model)."""
        means, _ = self.predict_all_levels(x)
        return means[-1]

    def predict_variances(
        self, X: np.ndarray, is_acting=None, is_ri=False
    ) -> np.ndarray:
        """Conditional variance of the highest fidelity level (sparse model)."""
        _, covariances = self.predict_all_levels(X)
        return covariances[-1]

    def predict_variances_all_levels(self, x):
        """Conditional variances of all the fidelity levels (sparse model)."""
        _, covariances = self.predict_all_levels(x)
        return np.hstack([np.asarray(c).reshape(-1, 1) for c in covariances])