"""
Author: Paul Saves <paul.saves@ird.fr>

This package is distributed under New BSD license.

Polynomial Chaos Expansion (PCE) and Sparse PCE surrogate models.
"""

from copy import deepcopy
from numbers import Integral

import numpy as np
from scipy.stats import qmc

from smt.design_space.design_space import BaseDesignSpace
from smt.surrogate_models.pce_backend import (
    SUPPORTED_METHODS,
    _is_single_dist_spec,
    assemble_basis_deriv_matrix,
    assemble_basis_matrix,
    build_pce_jax_predictor,
    eval_1d_tables,
    parse_distributions,
    select_adaptive_pce,
)
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils import persistence
from smt.utils.caching import cached_operation


def _normalize_method_name(method: str) -> str:
    m = str(method).strip().lower()
    if m not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported PCE method {method!r}. Expected one of {SUPPORTED_METHODS}."
        )
    return m


def _normalize_degree_sequence(degree_opt):
    if isinstance(degree_opt, Integral) and not isinstance(degree_opt, bool):
        vals = [int(degree_opt)]
    elif isinstance(degree_opt, (range, tuple, list, np.ndarray)):
        vals = []
        for item in degree_opt:
            if not isinstance(item, Integral) or isinstance(item, bool):
                raise ValueError(f"Degree entries must be integers >= 0, got {item!r}.")
            vals.append(int(item))
    else:
        raise ValueError(f"Invalid degree specification: {degree_opt!r}.")

    if len(vals) == 0 or any(v < 0 for v in vals):
        raise ValueError(f"Degrees must be non-empty integers >= 0, got {vals}.")
    return sorted(set(vals))


def _normalize_qnorm_sequence(q_opt):
    if isinstance(q_opt, Integral) and not isinstance(q_opt, bool):
        vals = [float(q_opt)]
    elif isinstance(q_opt, float):
        vals = [float(q_opt)]
    elif isinstance(q_opt, (tuple, list, np.ndarray)):
        vals = [float(v) for v in np.asarray(q_opt, dtype=float).ravel()]
    else:
        raise ValueError(f"Invalid q_norm specification: {q_opt!r}.")

    if len(vals) == 0 or any(v <= 0.0 or v > 1.0 for v in vals):
        raise ValueError(f"q_norm values must lie in (0, 1], got {vals}.")
    # Deduplicate preserving order
    seen = []
    for v in vals:
        if all(abs(v - s) > 1e-12 for s in seen):
            seen.append(v)
    return seen


def _infer_nx_for_design(
    marginals=None, xlimits=None, design_space=None, distributions=None, nx=None
):
    if nx is not None:
        return int(nx)
    if marginals is not None:
        return len(marginals)
    if xlimits is not None:
        return int(np.asarray(xlimits).shape[0])
    if design_space is not None:
        return int(len(design_space.design_variables))
    if isinstance(distributions, (list, tuple)) and len(distributions) > 0:
        if not _is_single_dist_spec(distributions):
            return int(len(distributions))
    raise ValueError(
        "Cannot infer input dimension nx; provide nx, xlimits, design_space, or per-dimension distributions."
    )


class PCE(SurrogateModel):
    """
    Polynomial Chaos Expansion (PCE) and Sparse Adaptive PCE surrogate model.

    Supports orthonormal Askey polynomial families (Legendre, Hermite, Laguerre,
    Jacobi), discretized Stieltjes and empirical/arbitrary PCE (aPCE) marginals,
    CDF-Hermite SciPy marginals, hyperbolic q-norm basis truncation, analytical
    Blatman-Sudret LOO cross-validation, sparse solvers (LARS, OMP, LASSO,
    Subspace Pursuit, Bayesian Compressive Sensing, OLS, weighted regression, and
    Gaussian quadrature projection), higher-order moments, Sobol indices,
    conditional expectation sub-models, exact derivatives, and JAX predictors.
    """

    name = "PCE"

    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        supports = self.supports

        declare(
            "degree",
            2,
            types=(Integral, tuple, list, range, np.ndarray),
            desc="Total polynomial degree (int or sequence of candidate degrees for LOO adaptivity)",
        )
        declare(
            "q_norm",
            1.0,
            types=(Integral, float, tuple, list, np.ndarray),
            desc="Hyperbolic truncation quasi-norm q in (0, 1] (or sequence of candidates)",
        )
        declare(
            "max_interaction",
            None,
            types=(type(None), Integral),
            desc="Maximum interaction order allowed in multi-indices (None means no cap)",
        )
        declare(
            "method",
            "ols",
            values=[
                "ols",
                "lars",
                "omp",
                "lasso",
                "sp",
                "bcs",
                "quadrature",
                "OLS",
                "LARS",
                "OMP",
                "LASSO",
                "SP",
                "BCS",
                "QUADRATURE",
            ],
            types=str,
            desc="Coefficient estimation strategy: 'ols', 'lars', 'omp', 'lasso', 'sp', 'bcs', or 'quadrature'",
        )
        declare(
            "distributions",
            None,
            desc="Marginal input distributions (Askey specs, dicts, tuples, or SciPy frozen RVs)",
        )
        declare(
            "xlimits",
            None,
            types=(type(None), list, tuple, np.ndarray),
            desc="Lower/upper bounds in each dimension - ndarray [nx, 2]",
        )
        declare(
            "design_space",
            None,
            types=(type(None), BaseDesignSpace),
            desc="Continuous SMT DesignSpace definition",
        )
        declare(
            "weights",
            None,
            types=(type(None), list, tuple, np.ndarray, str),
            desc="Optional sample weights [nt] or 'christoffel' for coherence-optimal weighted regression",
        )
        declare(
            "anisotropy_weights",
            None,
            types=(type(None), list, tuple, np.ndarray),
            desc="Positive per-dimension anisotropy weights for hyperbolic truncation",
        )
        declare(
            "early_stop",
            True,
            types=bool,
            desc="Whether to apply Blatman-Sudret early stopping during sparse/degree search",
        )
        declare(
            "loo_target",
            1e-6,
            types=(Integral, float),
            desc="Target normalized modified LOO error stopping threshold across candidate degrees",
        )
        declare(
            "max_basis",
            10000,
            types=Integral,
            desc="Safety cap on candidate basis cardinality during degree sweeps",
        )
        declare(
            "reg",
            1e-12,
            types=(Integral, float),
            desc="Small regularization fallback when OLS system is rank-deficient or underdetermined",
        )
        declare(
            "jax_dtype",
            "float64",
            values=["float32", "float64"],
            types=str,
            desc="Default floating-point precision for PCE.get_jax_predictor()",
        )
        declare(
            "data_dir",
            values=None,
            types=(type(None), str),
            desc="Directory for loading / saving cached data; None means do not save or load",
        )

        supports["derivatives"] = True

    def _check_is_trained(self):
        if not getattr(self, "_is_trained", False):
            raise RuntimeError(
                f"{self.__class__.__name__} model has not been trained yet. Call train() first."
            )

    def _validate_options(self):
        max_inter = self.options["max_interaction"]
        if max_inter is not None:
            if isinstance(max_inter, bool) or max_inter < 0:
                raise ValueError(
                    f"max_interaction must be a non-negative integer or None, got {max_inter!r}."
                )
        max_basis = self.options["max_basis"]
        if isinstance(max_basis, bool) or max_basis < 1:
            raise ValueError(f"max_basis must be an integer >= 1, got {max_basis!r}.")
        reg = self.options["reg"]
        if isinstance(reg, bool) or float(reg) < 0.0:
            raise ValueError(f"reg must be a non-negative number, got {reg!r}.")
        loo_target = self.options["loo_target"]
        if isinstance(loo_target, bool) or float(loo_target) < 0.0:
            raise ValueError(
                f"loo_target must be a non-negative number, got {loo_target!r}."
            )
        jax_dtype = str(self.options["jax_dtype"]).strip().lower()
        if jax_dtype not in ("float32", "float64"):
            raise ValueError(
                f"jax_dtype must be 'float32' or 'float64', got {self.options['jax_dtype']!r}."
            )
        weights = self.options["weights"]
        if isinstance(weights, str):
            w_mode = weights.strip().lower()
            if w_mode != "christoffel":
                raise ValueError(
                    f"Unsupported string weights option {weights!r}; expected 'christoffel'."
                )
            if _normalize_method_name(self.options["method"]) == "quadrature":
                raise ValueError(
                    "weights='christoffel' is only supported for regression methods, not 'quadrature'."
                )

    def _new_train(self):
        xt = self.training_points[None][0][0]
        yt = self.training_points[None][0][1]

        self._validate_options()
        degrees = _normalize_degree_sequence(self.options["degree"])
        q_norms = _normalize_qnorm_sequence(self.options["q_norm"])
        method = _normalize_method_name(self.options["method"])

        max_inter = self.options["max_interaction"]
        if max_inter is not None:
            max_inter = int(max_inter)

        marginals = parse_distributions(
            nx=self.nx,
            distributions=self.options["distributions"],
            xlimits=self.options["xlimits"],
            design_space=self.options["design_space"],
            xt=xt,
        )

        result = select_adaptive_pce(
            marginals=marginals,
            xt=xt,
            yt=yt,
            degrees=degrees,
            q_norms=q_norms,
            method=method,
            max_interaction=max_inter,
            anisotropy_weights=self.options["anisotropy_weights"],
            weights=self.options["weights"],
            early_stop=bool(self.options["early_stop"]),
            loo_target=float(self.options["loo_target"]),
            max_basis=int(self.options["max_basis"]),
            reg=float(self.options["reg"]),
        )

        self._marginals = marginals
        self.multi_indices = result["multi_indices"]
        self.coefficients = result["coefficients"]
        self.degree_ = result["degree"]
        self.q_norm_ = result["q_norm"]
        self.n_basis_candidate_ = result["n_basis_candidate"]
        self.loo_error_ = result["loo"]
        self.mloo_error_ = result["mloo"]
        self.q2_loo_ = 1.0 - result["loo"]
        self._is_trained = True

    def _train(self):
        if self.options["data_dir"] is not None:
            inputs = {"self": self}
            with cached_operation(inputs, self.options["data_dir"]) as outputs:
                if outputs:
                    self._marginals = outputs["marginals"]
                    self.multi_indices = outputs["multi_indices"]
                    self.coefficients = outputs["coefficients"]
                    self.degree_ = outputs["degree_"]
                    self.q_norm_ = outputs["q_norm_"]
                    self.n_basis_candidate_ = outputs["n_basis_candidate_"]
                    self.loo_error_ = outputs["loo_error_"]
                    self.mloo_error_ = outputs["mloo_error_"]
                    self.q2_loo_ = outputs["q2_loo_"]
                    self._is_trained = True
                else:
                    self._new_train()
                    outputs["marginals"] = self._marginals
                    outputs["multi_indices"] = self.multi_indices
                    outputs["coefficients"] = self.coefficients
                    outputs["degree_"] = self.degree_
                    outputs["q_norm_"] = self.q_norm_
                    outputs["n_basis_candidate_"] = self.n_basis_candidate_
                    outputs["loo_error_"] = self.loo_error_
                    outputs["mloo_error_"] = self.mloo_error_
                    outputs["q2_loo_"] = self.q2_loo_
        else:
            self._new_train()

    def _predict_values(self, x: np.ndarray) -> np.ndarray:
        self._check_is_trained()
        V_tables, _ = eval_1d_tables(self._marginals, self.multi_indices, x)
        Phi = assemble_basis_matrix(self.multi_indices, V_tables, x.shape[0])
        return Phi @ self.coefficients

    def _predict_derivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        self._check_is_trained()
        V_tables, dV_kx = eval_1d_tables(
            self._marginals, self.multi_indices, x, deriv_kx=kx
        )
        dPhi = assemble_basis_deriv_matrix(
            self.multi_indices, V_tables, dV_kx, kx=kx, n_samples=x.shape[0]
        )
        return dPhi @ self.coefficients

    def christoffel_weights(self, x: np.ndarray) -> np.ndarray:
        """
        Compute coherence-optimal inverse Christoffel weights ``P / sum_{alpha} Psi_alpha(x)^2``
        on the trained PCE basis for points ``x`` of shape ``(n, nx)``.
        """
        self._check_is_trained()
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, self.nx)
        if x_arr.ndim != 2 or x_arr.shape[1] != self.nx:
            raise ValueError(
                f"Expected 2D input with {self.nx} columns, got {x_arr.shape}."
            )
        V_tables, _ = eval_1d_tables(self._marginals, self.multi_indices, x_arr)
        Phi = assemble_basis_matrix(self.multi_indices, V_tables, x_arr.shape[0])
        k_diag = np.sum(Phi * Phi, axis=1)
        return float(self.multi_indices.shape[0]) / np.maximum(k_diag, 1e-30)

    def get_jax_predictor(self, dtype=None):
        """
        Return a pure JAX function ``predict(x)`` that is end-to-end differentiable.

        Parameters
        ----------
        dtype : str or jnp.dtype, optional
            Target JAX floating-point dtype (``'float64'`` or ``'float32'``).
        """
        self._check_is_trained()
        default_dtype = str(self.options["jax_dtype"]).strip().lower()
        if default_dtype not in ("float32", "float64"):
            raise ValueError(
                f"jax_dtype must be 'float32' or 'float64', got {self.options['jax_dtype']!r}."
            )
        return build_pce_jax_predictor(
            marginals=self._marginals,
            multi_indices=self.multi_indices,
            coefficients=self.coefficients,
            ny=self.ny,
            dtype=dtype,
            default_dtype=default_dtype,
        )

    # ------------------------------------------------------------------
    # Analytical UQ Moments, Global Sensitivity, and Conditional Expectation
    # ------------------------------------------------------------------

    def _const_row_mask(self) -> np.ndarray:
        return np.all(self.multi_indices == 0, axis=1)

    def mean(self) -> np.ndarray:
        """
        Analytical PCE output mean ``E[Y] = c_0`` of shape ``(ny,)``.
        """
        self._check_is_trained()
        const_mask = self._const_row_mask()
        if not np.any(const_mask):
            return np.zeros(self.ny, dtype=float)
        return np.sum(self.coefficients[const_mask], axis=0).astype(float)

    def variance(self) -> np.ndarray:
        """
        Analytical PCE output variance ``Var[Y] = sum_{alpha != 0} c_alpha^2`` of shape ``(ny,)``.
        """
        self._check_is_trained()
        nonconst = ~self._const_row_mask()
        if not np.any(nonconst):
            return np.zeros(self.ny, dtype=float)
        return np.sum(self.coefficients[nonconst] ** 2, axis=0).astype(float)

    def std(self) -> np.ndarray:
        """Analytical PCE output standard deviation of shape ``(ny,)``."""
        return np.sqrt(np.maximum(self.variance(), 0.0))

    def _central_moments_3_4(self):
        """
        Compute exact 3rd and 4th central moments ``(mu3, mu4)`` of shape ``(ny,)``.

        Uses exact 1D Gauss-quadrature tensor contraction for up to 60 active
        non-constant basis terms, falling back to high-order Sobol QMC above 60.
        """
        self._check_is_trained()
        nonconst = ~self._const_row_mask()
        if not np.any(nonconst):
            zeros = np.zeros(self.ny, dtype=float)
            return zeros, zeros.copy()

        active_indices = self.multi_indices[nonconst]
        C = self.coefficients[nonconst]  # (M, ny)
        n_active = active_indices.shape[0]

        if n_active <= 60:
            T3 = np.ones((n_active, n_active, n_active), dtype=float)
            T4 = np.ones((n_active * n_active, n_active * n_active), dtype=float)
            for j in range(self.nx):
                max_deg_j = int(np.max(active_indices[:, j]))
                if max_deg_j == 0:
                    continue
                marg_j = self._marginals[j]
                if marg_j.stieltjes_nodes is not None:
                    z_q, w_q = marg_j.stieltjes_nodes
                    x_q = marg_j.offset + marg_j.scale * z_q
                else:
                    n_q = max(2 * max_deg_j + 2, 4)
                    x_q, w_q = marg_j.gauss_quadrature_1d(n_q)
                V_j = marg_j.eval_1d(x_q, max_deg_j, with_deriv=False)
                Phi_j = V_j[:, active_indices[:, j]]  # (n_q, M)
                T3 *= np.einsum("q,qa,qb,qc->abc", w_q, Phi_j, Phi_j, Phi_j)
                P_j = (Phi_j[:, :, None] * Phi_j[:, None, :]).reshape(
                    x_q.shape[0], n_active * n_active
                )
                T4 *= P_j.T @ (w_q[:, None] * P_j)

            mu3 = np.einsum("abc,am,bm,cm->m", T3, C, C, C)
            C2 = (C[:, None, :] * C[None, :, :]).reshape(n_active * n_active, self.ny)
            mu4 = np.einsum("ij,im,jm->m", T4, C2, C2)
            return mu3.astype(float), mu4.astype(float)

        x_qmc = self.sample_design(16384, sampling="sobol", seed=0)
        y_centered = self._predict_values(x_qmc) - self.mean()[None, :]
        mu3 = np.mean(y_centered**3, axis=0)
        mu4 = np.mean(y_centered**4, axis=0)
        return mu3.astype(float), mu4.astype(float)

    def skewness(self) -> np.ndarray:
        """PCE output skewness ``E[(Y - E[Y])^3] / std^3`` of shape ``(ny,)``."""
        sd = self.std()
        mu3, _ = self._central_moments_3_4()
        out = np.zeros(self.ny, dtype=float)
        pos = sd > 0.0
        out[pos] = mu3[pos] / (sd[pos] ** 3)
        return out

    def kurtosis(self, fisher: bool = True) -> np.ndarray:
        """
        PCE output kurtosis of shape ``(ny,)``.

        If ``fisher=True`` (default), returns excess kurtosis ``mu4 / var^2 - 3``;
        otherwise returns Pearson kurtosis ``mu4 / var^2``. Returns 0 for constants.
        """
        var = self.variance()
        _, mu4 = self._central_moments_3_4()
        out = np.zeros(self.ny, dtype=float)
        pos = var > 0.0
        out[pos] = mu4[pos] / (var[pos] ** 2) - (3.0 if fisher else 0.0)
        return out

    def moments(self, fisher: bool = True) -> dict:
        """
        Return dictionary of analytical PCE output moments:
        ``'mean'``, ``'variance'``, ``'std'``, ``'skewness'``, and ``'kurtosis'``.
        """
        mean = self.mean()
        var = self.variance()
        sd = np.sqrt(np.maximum(var, 0.0))
        mu3, mu4 = self._central_moments_3_4()
        skew = np.zeros(self.ny, dtype=float)
        kurt = np.zeros(self.ny, dtype=float)
        pos = var > 0.0
        skew[pos] = mu3[pos] / (sd[pos] ** 3)
        kurt[pos] = mu4[pos] / (var[pos] ** 2) - (3.0 if fisher else 0.0)
        return {
            "mean": mean,
            "variance": var,
            "std": sd,
            "skewness": skew,
            "kurtosis": kurt,
        }

    def _format_output_array(self, arr, output):
        if output is not None:
            if not isinstance(output, Integral) or isinstance(output, bool):
                raise ValueError(f"output index must be an integer, got {output!r}.")
            out_idx = int(output)
            if out_idx < 0 or out_idx >= self.ny:
                raise ValueError(f"output={out_idx} out of bounds for ny={self.ny}.")
            return arr[..., out_idx]
        if self.ny == 1:
            return arr[..., 0]
        return arr

    def sobol_indices(self, output=None, second_order: bool = False) -> dict:
        """
        Compute exact analytical Sobol sensitivity indices from orthonormal PCE coefficients.

        Parameters
        ----------
        output : int, optional
            Specific 0-based output index when ``ny > 1``. If omitted, returns shape
            ``(nx,)`` when ``ny == 1`` or ``(nx, ny)`` when ``ny > 1``.
        second_order : bool, default=False
            Whether to also return symmetric pairwise interaction Sobol indices
            ``S_ij`` under key ``'second_order'``.
        """
        self._check_is_trained()
        var_total = self.variance()
        nonconst = ~self._const_row_mask()
        active_idx = self.multi_indices[nonconst] > 0  # (P_nc, nx)
        sq_coef = self.coefficients[nonconst] ** 2  # (P_nc, ny)
        inter_order = np.sum(active_idx, axis=1)

        S1 = np.zeros((self.nx, self.ny), dtype=float)
        ST = np.zeros((self.nx, self.ny), dtype=float)
        S2 = (
            np.zeros((self.nx, self.nx, self.ny), dtype=float) if second_order else None
        )

        pos_var = var_total > 0.0
        if np.any(pos_var) and active_idx.shape[0] > 0:
            order1_mask = inter_order == 1
            for j in range(self.nx):
                in_j = active_idx[:, j]
                if np.any(in_j):
                    ST[j, pos_var] = (
                        np.sum(sq_coef[in_j][:, pos_var], axis=0) / var_total[pos_var]
                    )
                    strict_j = order1_mask & in_j
                    if np.any(strict_j):
                        S1[j, pos_var] = (
                            np.sum(sq_coef[strict_j][:, pos_var], axis=0)
                            / var_total[pos_var]
                        )

            if second_order:
                order2_mask = inter_order == 2
                for i in range(self.nx - 1):
                    for j in range(i + 1, self.nx):
                        pair_ij = order2_mask & active_idx[:, i] & active_idx[:, j]
                        if np.any(pair_ij):
                            val = (
                                np.sum(sq_coef[pair_ij][:, pos_var], axis=0)
                                / var_total[pos_var]
                            )
                            S2[i, j, pos_var] = val
                            S2[j, i, pos_var] = val

        res = {
            "first_order": self._format_output_array(S1, output),
            "total": self._format_output_array(ST, output),
        }
        if second_order:
            res["second_order"] = self._format_output_array(S2, output)
        return res

    def subset_sobol(self, indices, closed: bool = True, output=None):
        """
        Compute exact analytical closed or total Sobol index for a variable subset.

        Parameters
        ----------
        indices : sequence of int
            0-based variable indices defining subset ``u``.
        closed : bool, default=True
            If True, sums variance contributions depending *only* on variables in
            ``u`` (matching ``Var(E[Y | X_u]) / Var(Y)``). If False, computes total
            subset Sobol ``S_u^T``.
        """
        self._check_is_trained()
        idx_tuple = _validate_index_subset(indices, self.nx)
        var_total = self.variance()
        nonconst = ~self._const_row_mask()
        active_idx = self.multi_indices[nonconst] > 0
        sq_coef = self.coefficients[nonconst] ** 2

        if active_idx.shape[0] == 0:
            raw = np.zeros(self.ny, dtype=float)
        else:
            in_u = np.zeros(self.nx, dtype=bool)
            in_u[list(idx_tuple)] = True
            if closed:
                mask = ~np.any(active_idx[:, ~in_u], axis=1)
            else:
                mask = np.any(active_idx[:, in_u], axis=1)

            raw = np.zeros(self.ny, dtype=float)
            pos_var = var_total > 0.0
            if np.any(pos_var) and np.any(mask):
                raw[pos_var] = (
                    np.sum(sq_coef[mask][:, pos_var], axis=0) / var_total[pos_var]
                )

        if output is not None:
            return float(self._format_output_array(raw, output))
        if self.ny == 1:
            return float(raw[0])
        return raw

    def conditional_expectation(self, conditioning_indices) -> "PCE":
        """
        Build the analytical conditional expectation metamodel ``E[Y | X_u]``.

        Under independent input marginals, integrating out complementary inputs
        ``X_{-u}`` keeps all and only the PCE terms whose multi-indices vanish
        outside ``conditioning_indices``, returning a ready-to-evaluate reduced ``PCE``
        model in ``len(conditioning_indices)`` variables.
        """
        self._check_is_trained()
        cond_idx = _validate_index_subset(conditioning_indices, self.nx)

        in_u = np.zeros(self.nx, dtype=bool)
        in_u[list(cond_idx)] = True
        keep_terms = ~np.any(self.multi_indices[:, ~in_u] > 0, axis=1)

        if not np.any(keep_terms):
            reduced_indices = np.zeros((1, len(cond_idx)), dtype=int)
            reduced_coef = np.zeros((1, self.ny), dtype=float)
        else:
            reduced_indices = self.multi_indices[keep_terms][:, cond_idx].copy()
            reduced_coef = self.coefficients[keep_terms, :].copy()

        # Construct trained reduced PCE preserving options & JAX settings
        sub_xlimits = None
        if self.options["xlimits"] is not None:
            sub_xlimits = np.asarray(self.options["xlimits"], dtype=float)[
                list(cond_idx)
            ]
        elif all(
            np.isfinite(self._marginals[i].bounds[0])
            and np.isfinite(self._marginals[i].bounds[1])
            for i in cond_idx
        ):
            sub_xlimits = np.array(
                [self._marginals[i].bounds for i in cond_idx], dtype=float
            )

        cond_degree = (
            int(np.max(np.sum(reduced_indices, axis=1))) if reduced_indices.size else 0
        )
        sub_marginals = [deepcopy(self._marginals[i]) for i in cond_idx]
        cond_model = PCE(
            degree=cond_degree,
            q_norm=self.q_norm_,
            method="ols",
            distributions=sub_marginals,
            xlimits=sub_xlimits,
            print_global=bool(self.options["print_global"]),
            print_training=bool(self.options["print_training"]),
            print_prediction=bool(self.options["print_prediction"]),
            print_problem=bool(self.options["print_problem"]),
            print_solver=bool(self.options["print_solver"]),
            jax_dtype=self.options["jax_dtype"],
        )
        cond_model.nx = len(cond_idx)
        cond_model.ny = self.ny
        cond_model.nt = 0
        cond_model._marginals = sub_marginals
        cond_model.multi_indices = reduced_indices
        cond_model.coefficients = reduced_coef
        cond_model.degree_ = cond_degree
        cond_model.q_norm_ = self.q_norm_
        cond_model.n_basis_candidate_ = int(reduced_indices.shape[0])
        cond_model.loo_error_ = np.full(self.ny, np.nan, dtype=float)
        cond_model.mloo_error_ = np.full(self.ny, np.nan, dtype=float)
        cond_model.q2_loo_ = np.full(self.ny, np.nan, dtype=float)
        cond_model._is_trained = True
        return cond_model

    # ------------------------------------------------------------------
    # Quadrature & Probabilistic Experimental Design Helpers
    # ------------------------------------------------------------------

    def _resolve_marginals_before_train(self, nx=None):
        if getattr(self, "_marginals", None) is not None and (
            nx is None or len(self._marginals) == nx
        ):
            return self._marginals
        inferred_nx = _infer_nx_for_design(
            marginals=getattr(self, "_marginals", None),
            xlimits=self.options["xlimits"],
            design_space=self.options["design_space"],
            distributions=self.options["distributions"],
            nx=nx,
        )
        return parse_distributions(
            nx=inferred_nx,
            distributions=self.options["distributions"],
            xlimits=self.options["xlimits"],
            design_space=self.options["design_space"],
        )

    def quadrature_points(self, order=None, nx=None):
        """
        Generate tensor-product Gaussian quadrature nodes and probability weights
        matched to each variable's marginal distribution.

        Returns
        -------
        x_quad : np.ndarray of shape ``(n_quad, nx)``
        weights : np.ndarray of shape ``(n_quad,)``
            Normalized probability weights summing to 1.0.
        """
        marginals = self._resolve_marginals_before_train(nx=nx)
        dim = len(marginals)

        if order is None:
            max_deg = max(_normalize_degree_sequence(self.options["degree"]))
            aniso = self.options["anisotropy_weights"]
            if aniso is not None:
                w = np.asarray(aniso, dtype=float).ravel()
                if w.shape != (dim,) or np.any(w <= 0.0):
                    raise ValueError(
                        f"anisotropy_weights must contain {dim} positive floats."
                    )
                orders = [
                    max(1, int(np.floor(float(max_deg) / float(w[j]) + 1e-12)) + 1)
                    for j in range(dim)
                ]
            else:
                orders = [max_deg + 1] * dim
        elif isinstance(order, Integral) and not isinstance(order, bool):
            orders = [int(order)] * dim
        elif isinstance(order, (tuple, list, np.ndarray)):
            orders = [int(o) for o in order]
            if len(orders) != dim:
                raise ValueError(
                    f"Expected {dim} quadrature orders, got {len(orders)}."
                )
        else:
            raise ValueError(f"Invalid quadrature order specification: {order!r}.")

        total_pts = 1
        for o in orders:
            if o < 1:
                raise ValueError(f"Quadrature order must be >= 1, got {o}.")
            total_pts *= o
        if total_pts > 200000:
            raise ValueError(
                f"Tensor-product quadrature grid requires {total_pts} points (> 200000). "
                "Use regression with sparse PCE or specify smaller per-axis orders."
            )

        nodes_1d = []
        weights_1d = []
        for m, o in zip(marginals, orders):
            x_j, w_j = m.gauss_quadrature_1d(o)
            nodes_1d.append(x_j)
            weights_1d.append(w_j)

        grids = np.meshgrid(*nodes_1d, indexing="ij")
        x_quad = np.column_stack([g.ravel() for g in grids])
        w_grids = np.meshgrid(*weights_1d, indexing="ij")
        weights = np.ones(x_quad.shape[0], dtype=float)
        for wg in w_grids:
            weights *= wg.ravel()
        weights /= np.sum(weights)
        return x_quad, weights

    def sample_design(
        self, n_samples: int, sampling: str = "lhs", seed=None, nx=None
    ) -> np.ndarray:
        """
        Generate an experimental design in physical space distributed according to
        the configured marginal probability distributions.

        Parameters
        ----------
        n_samples : int
            Number of design samples.
        sampling : {'lhs', 'mc', 'random', 'sobol', 'halton'}
            Space-filling or Monte Carlo sampling method on ``[0, 1]^nx``.
        seed : int or np.random.Generator, optional
            Random seed for reproducibility.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}.")

        marginals = self._resolve_marginals_before_train(nx=nx)
        dim = len(marginals)
        unit_limits = np.tile(np.array([[0.0, 1.0]]), (dim, 1))
        method = str(sampling).strip().lower()

        if method == "lhs":
            from smt.sampling_methods import LHS

            u = LHS(xlimits=unit_limits, criterion="ese", seed=seed)(int(n_samples))
        elif method in ("mc", "random"):
            from smt.sampling_methods import Random

            u = Random(xlimits=unit_limits, seed=seed)(int(n_samples))
        elif method == "sobol":
            from smt.sampling_methods.qmc import _build_qmc_sampler

            rng = np.random.default_rng(seed)
            u = _build_qmc_sampler(qmc.Sobol, dim, scramble=True, rng=rng).random(
                int(n_samples)
            )
        elif method == "halton":
            from smt.sampling_methods.qmc import _build_qmc_sampler

            rng = np.random.default_rng(seed)
            u = _build_qmc_sampler(qmc.Halton, dim, scramble=True, rng=rng).random(
                int(n_samples)
            )
        else:
            raise ValueError(
                f"Unsupported sampling method {sampling!r}. Expected 'lhs', 'mc', 'random', 'sobol', or 'halton'."
            )

        x = np.empty_like(u)
        for j, marg in enumerate(marginals):
            x[:, j] = marg.ppf(u[:, j])
        return x

    def save(self, filename):
        """Save trained PCE model to disk."""
        persistence.save(self, filename)

    @staticmethod
    def load(filename):
        """Load saved PCE model from disk."""
        return persistence.load(filename)


class SparsePCE(PCE):
    """
    Sparse adaptive Polynomial Chaos Expansion surrogate model.

    Convenience preset configuring ``PCE`` with Hybrid-LARS selection
    and Blatman-Sudret degree adaptivity over ``degree=range(1, 6)``.
    Set ``q_norm < 1.0`` (e.g. ``0.75``) to enable hyperbolic truncation in high dimensions.
    """

    name = "SparsePCE"

    def _initialize(self):
        super()._initialize()
        self.options["method"] = "lars"
        self.options["degree"] = range(1, 6)
        self.options["q_norm"] = 1.0


def _validate_index_subset(indices, nx: int) -> tuple:
    """Validate and preserve ordering of a non-empty unique 0-based index subset."""
    try:
        seq = list(indices)
    except TypeError as exc:
        raise ValueError(
            f"Expected a sequence of integer indices, got {indices!r}."
        ) from exc

    if len(seq) == 0:
        raise ValueError("conditioning_indices / subset indices cannot be empty.")

    out = []
    seen = set()
    for item in seq:
        if not isinstance(item, Integral) or isinstance(item, bool):
            raise ValueError(f"Variable index must be an integer, got {item!r}.")
        idx = int(item)
        if idx < 0 or idx >= nx:
            raise ValueError(f"Index {idx} out of bounds for nx={nx}.")
        if idx in seen:
            raise ValueError(f"Duplicate index {idx} in subset.")
        seen.add(idx)
        out.append(idx)
    return tuple(out)
