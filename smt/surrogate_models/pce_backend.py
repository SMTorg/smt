"""
Pure-Python backend for Polynomial Chaos Expansion (PCE) and Sparse PCE.

Provides marginal distribution transforms, Favard orthonormal polynomial
recurrences (Legendre, Hermite, Laguerre, Jacobi, and arbitrary CDF-Hermite),
hyperbolic q-norm multi-index generation, sparse regression solvers (OLS,
Hybrid-LARS, OMP, LASSO, Gaussian quadrature projection), Blatman-Sudret
analytical LOO error estimation, Sobol indices, conditional expectation, and
optional JAX predictors.
"""

from dataclasses import dataclass
import warnings

import numpy as np
import scipy.linalg
import scipy.stats
from sklearn.linear_model import lars_path


SUPPORTED_ASK_FAMILIES = ("legendre", "hermite", "laguerre", "jacobi")
SUPPORTED_METHODS = ("ols", "lars", "omp", "lasso", "quadrature")


@dataclass
class MarginalSpec:
    """Representation of a single continuous marginal distribution for PCE."""

    family: str
    offset: float
    scale: float
    alpha_param: float = 1.0
    beta_param: float = 1.0
    bounds: tuple = (-np.inf, np.inf)
    scipy_dist: object = None

    @property
    def is_askey(self) -> bool:
        """Return True if this marginal uses an affine Askey orthogonal family."""
        return self.family in SUPPORTED_ASK_FAMILIES

    def to_reduced(self, x_1d: np.ndarray) -> np.ndarray:
        """Transform physical coordinates ``x`` to reduced orthogonal variable ``z``."""
        if self.is_askey:
            return (x_1d - self.offset) / self.scale
        u = np.clip(self.scipy_dist.cdf(x_1d), 1e-15, 1.0 - 1e-15)
        return scipy.stats.norm.ppf(u)

    def d_reduced_dx(self, x_1d: np.ndarray, z_1d: np.ndarray = None) -> np.ndarray:
        """Compute derivative ``dz / dx`` at physical coordinates ``x``."""
        if self.is_askey:
            return np.full(x_1d.shape, 1.0 / self.scale, dtype=float)
        if z_1d is None:
            z_1d = self.to_reduced(x_1d)
        pdf_x = self.scipy_dist.pdf(x_1d)
        pdf_z = np.maximum(scipy.stats.norm.pdf(z_1d), 1e-300)
        return pdf_x / pdf_z

    def ppf(self, u_1d: np.ndarray) -> np.ndarray:
        """Inverse CDF mapping ``u in (0, 1)`` to physical coordinates."""
        u_clip = np.clip(u_1d, 1e-15, 1.0 - 1e-15)
        if self.family == "legendre":
            return self.offset + self.scale * (2.0 * u_1d - 1.0)
        if self.family == "hermite":
            return self.offset + self.scale * scipy.stats.norm.ppf(u_clip)
        if self.family == "laguerre":
            return self.offset + self.scale * scipy.stats.gamma.ppf(
                u_clip, a=self.alpha_param
            )
        if self.family == "jacobi":
            return self.offset + self.scale * (
                2.0 * scipy.stats.beta.ppf(u_clip, self.alpha_param, self.beta_param)
                - 1.0
            )
        return self.scipy_dist.ppf(u_clip)

    def recurrence_coefficients(self, max_deg: int):
        """
        Compute Favard symmetric three-term recurrence coefficients ``(a, b)``.

        Satisfies ``b[k] * psi_{k+1}(z) = (z - a[k]) * psi_k(z) - b[k-1] * psi_{k-1}(z)``
        for orthonormal polynomials ``psi_k`` with ``psi_0(z) = 1``.
        """
        n_coef = max_deg + 1
        a = np.zeros(n_coef, dtype=float)
        b = np.zeros(n_coef, dtype=float)
        k = np.arange(n_coef, dtype=float)

        if self.family == "legendre":
            b[:] = (k + 1.0) / np.sqrt((2.0 * k + 1.0) * (2.0 * k + 3.0))
        elif self.family in ("hermite", "cdf_hermite"):
            b[:] = np.sqrt(k + 1.0)
        elif self.family == "laguerre":
            shape = float(self.alpha_param)
            a[:] = 2.0 * k + shape
            b[:] = np.sqrt((k + 1.0) * (k + shape))
        elif self.family == "jacobi":
            # Weight on [-1, 1]: (1 - z)^p * (1 + z)^q with p = beta - 1, q = alpha - 1
            p = float(self.beta_param) - 1.0
            q = float(self.alpha_param) - 1.0
            s = p + q
            a[0] = (q - p) / (s + 2.0)
            b[0] = (2.0 / (s + 2.0)) * np.sqrt((p + 1.0) * (q + 1.0) / (s + 3.0))
            if n_coef > 1:
                k1 = k[1:]
                two_k_s = 2.0 * k1 + s
                a[1:] = (q * q - p * p) / (two_k_s * (two_k_s + 2.0))
                b[1:] = np.sqrt(
                    4.0
                    * (k1 + 1.0)
                    * (k1 + p + 1.0)
                    * (k1 + q + 1.0)
                    * (k1 + s + 1.0)
                    / ((two_k_s + 1.0) * (two_k_s + 2.0) ** 2 * (two_k_s + 3.0))
                )
        else:
            raise ValueError(f"Unsupported polynomial family: {self.family!r}.")

        return a, b

    def eval_1d(self, x_1d: np.ndarray, max_deg: int, with_deriv: bool = False):
        """Evaluate orthonormal 1D basis polynomials up to degree ``max_deg``."""
        z = self.to_reduced(x_1d)
        n = z.shape[0]
        V = np.zeros((n, max_deg + 1), dtype=float)
        V[:, 0] = 1.0

        dV = np.zeros((n, max_deg + 1), dtype=float) if with_deriv else None
        if max_deg == 0:
            return (V, dV) if with_deriv else V

        a, b = self.recurrence_coefficients(max_deg)
        for k in range(max_deg):
            term = (z - a[k]) * V[:, k]
            if k > 0:
                term = term - b[k - 1] * V[:, k - 1]
            V[:, k + 1] = term / b[k]

            if with_deriv:
                dterm = V[:, k] + (z - a[k]) * dV[:, k]
                if k > 0:
                    dterm = dterm - b[k - 1] * dV[:, k - 1]
                dV[:, k + 1] = dterm / b[k]

        if with_deriv:
            dz_dx = self.d_reduced_dx(x_1d, z)
            dV = dV * dz_dx[:, None]
            return V, dV
        return V

    def gauss_quadrature_1d(self, n_points: int):
        """
        Compute 1D Gaussian quadrature physical nodes and probability weights.

        Uses the Golub-Welsch tridiagonal eigendecomposition of the symmetric
        Jacobi matrix so weights sum to 1 under this marginal probability measure.
        """
        if n_points < 1:
            raise ValueError(f"Quadrature order must be >= 1, got {n_points}.")
        a, b = self.recurrence_coefficients(max(n_points - 1, 0))
        if n_points == 1:
            z_nodes = np.array([a[0]], dtype=float)
            weights = np.array([1.0], dtype=float)
        else:
            z_nodes, evecs = scipy.linalg.eigh_tridiagonal(
                a[:n_points], b[: n_points - 1]
            )
            weights = evecs[0, :] ** 2
            weights = weights / np.sum(weights)

        if self.is_askey:
            x_nodes = self.offset + self.scale * z_nodes
        else:
            x_nodes = self.scipy_dist.ppf(
                np.clip(scipy.stats.norm.cdf(z_nodes), 1e-15, 1.0 - 1e-15)
            )
        return x_nodes, weights


def _validate_bounds(bounds, dim_idx):
    low, high = float(bounds[0]), float(bounds[1])
    if not (np.isfinite(low) and np.isfinite(high) and high > low):
        raise ValueError(
            f"Dimension {dim_idx}: invalid bounds ({low}, {high}); expected finite high > low."
        )
    return low, high


def _parse_single_marginal(spec, dim_idx, default_bounds):
    """Parse a single user marginal specification into a ``MarginalSpec``."""
    # Support SciPy frozen continuous random variables directly
    if hasattr(spec, "dist") and hasattr(spec, "cdf") and hasattr(spec, "pdf"):
        name = getattr(spec.dist, "name", "")
        shapes, loc, scale = spec.dist._parse_args(*spec.args, **spec.kwds)
        loc = float(loc)
        scale = float(scale)
        if name == "uniform":
            return MarginalSpec(
                family="legendre",
                offset=loc + 0.5 * scale,
                scale=0.5 * scale,
                bounds=(loc, loc + scale),
            )
        if name == "norm":
            return MarginalSpec(
                family="hermite",
                offset=loc,
                scale=scale,
                bounds=(-np.inf, np.inf),
            )
        if name == "gamma":
            return MarginalSpec(
                family="laguerre",
                offset=loc,
                scale=scale,
                alpha_param=float(shapes[0]),
                bounds=(loc, np.inf),
            )
        if name == "expon":
            return MarginalSpec(
                family="laguerre",
                offset=loc,
                scale=scale,
                alpha_param=1.0,
                bounds=(loc, np.inf),
            )
        if name == "beta":
            alpha_p, beta_p = float(shapes[0]), float(shapes[1])
            return MarginalSpec(
                family="jacobi",
                offset=loc + 0.5 * scale,
                scale=0.5 * scale,
                alpha_param=alpha_p,
                beta_param=beta_p,
                bounds=(loc, loc + scale),
            )
        support = spec.support()
        return MarginalSpec(
            family="cdf_hermite",
            offset=0.0,
            scale=1.0,
            bounds=(float(support[0]), float(support[1])),
            scipy_dist=spec,
        )

    if isinstance(spec, MarginalSpec):
        return spec

    if spec is None:
        low, high = _validate_bounds(default_bounds, dim_idx)
        return MarginalSpec(
            family="legendre",
            offset=0.5 * (low + high),
            scale=0.5 * (high - low),
            bounds=(low, high),
        )

    if isinstance(spec, str):
        spec = {"type": spec}
    elif isinstance(spec, (tuple, list)):
        if len(spec) == 0:
            raise ValueError(f"Dimension {dim_idx}: empty distribution tuple.")
        dist_type = str(spec[0]).strip().lower()
        params = list(spec[1:])
        if dist_type in ("uniform", "u"):
            if len(params) == 0:
                spec = {"type": "uniform"}
            elif len(params) == 2:
                spec = {"type": "uniform", "bounds": (params[0], params[1])}
            else:
                raise ValueError(
                    f"Dimension {dim_idx}: uniform tuple expects () or (low, high)."
                )
        elif dist_type in ("normal", "gaussian", "n"):
            if len(params) == 0:
                spec = {"type": "normal"}
            elif len(params) == 2:
                spec = {"type": "normal", "mean": params[0], "std": params[1]}
            else:
                raise ValueError(
                    f"Dimension {dim_idx}: normal tuple expects () or (mean, std)."
                )
        elif dist_type in ("gamma", "g"):
            if len(params) == 1:
                spec = {"type": "gamma", "shape": params[0]}
            elif len(params) == 2:
                spec = {"type": "gamma", "shape": params[0], "scale": params[1]}
            elif len(params) == 3:
                spec = {
                    "type": "gamma",
                    "shape": params[0],
                    "scale": params[1],
                    "loc": params[2],
                }
            else:
                raise ValueError(
                    f"Dimension {dim_idx}: gamma tuple expects (shape[, scale[, loc]])."
                )
        elif dist_type in ("exponential", "expon", "exp"):
            if len(params) == 0:
                spec = {"type": "exponential"}
            elif len(params) == 1:
                spec = {"type": "exponential", "scale": params[0]}
            elif len(params) == 2:
                spec = {"type": "exponential", "scale": params[0], "loc": params[1]}
            else:
                raise ValueError(
                    f"Dimension {dim_idx}: exponential tuple expects ([scale[, loc]])."
                )
        elif dist_type in ("beta", "b"):
            if len(params) == 2:
                spec = {"type": "beta", "alpha": params[0], "beta": params[1]}
            elif len(params) == 4:
                spec = {
                    "type": "beta",
                    "alpha": params[0],
                    "beta": params[1],
                    "bounds": (params[2], params[3]),
                }
            else:
                raise ValueError(
                    f"Dimension {dim_idx}: beta tuple expects (alpha, beta[, low, high])."
                )
        else:
            raise ValueError(
                f"Dimension {dim_idx}: unknown distribution {dist_type!r}."
            )

    if not isinstance(spec, dict):
        raise ValueError(
            f"Dimension {dim_idx}: unsupported distribution specification {spec!r}."
        )

    dist_type = str(spec.get("type", "uniform")).strip().lower()

    if dist_type in ("uniform", "u"):
        bounds = spec.get("bounds", default_bounds)
        low, high = _validate_bounds(bounds, dim_idx)
        return MarginalSpec(
            family="legendre",
            offset=0.5 * (low + high),
            scale=0.5 * (high - low),
            bounds=(low, high),
        )

    if dist_type in ("normal", "gaussian", "n"):
        mean = float(spec.get("mean", spec.get("mu", 0.0)))
        std = float(spec.get("std", spec.get("sigma", 1.0)))
        if std <= 0.0:
            raise ValueError(f"Dimension {dim_idx}: normal std must be > 0, got {std}.")
        return MarginalSpec(
            family="hermite",
            offset=mean,
            scale=std,
            bounds=(-np.inf, np.inf),
        )

    if dist_type in ("gamma", "g"):
        shape = float(spec.get("shape", spec.get("k", spec.get("alpha", 1.0))))
        if "rate" in spec and "scale" in spec:
            raise ValueError(
                f"Dimension {dim_idx}: specify either 'scale' or 'rate' for gamma, not both."
            )
        scale = (
            1.0 / float(spec["rate"])
            if "rate" in spec
            else float(spec.get("scale", spec.get("theta", 1.0)))
        )
        loc = float(spec.get("loc", 0.0))
        if shape <= 0.0 or scale <= 0.0:
            raise ValueError(f"Dimension {dim_idx}: gamma shape and scale must be > 0.")
        return MarginalSpec(
            family="laguerre",
            offset=loc,
            scale=scale,
            alpha_param=shape,
            bounds=(loc, np.inf),
        )

    if dist_type in ("exponential", "expon", "exp"):
        if "rate" in spec and "scale" in spec:
            raise ValueError(
                f"Dimension {dim_idx}: specify either 'scale' or 'rate' for exponential, not both."
            )
        scale = (
            1.0 / float(spec["rate"])
            if "rate" in spec
            else float(spec.get("scale", 1.0))
        )
        loc = float(spec.get("loc", 0.0))
        if scale <= 0.0:
            raise ValueError(f"Dimension {dim_idx}: exponential scale must be > 0.")
        return MarginalSpec(
            family="laguerre",
            offset=loc,
            scale=scale,
            alpha_param=1.0,
            bounds=(loc, np.inf),
        )

    if dist_type in ("beta", "b"):
        alpha_p = float(spec.get("alpha", spec.get("a", 1.0)))
        beta_p = float(spec.get("beta", spec.get("b", 1.0)))
        if alpha_p <= 0.0 or beta_p <= 0.0:
            raise ValueError(
                f"Dimension {dim_idx}: beta parameters alpha and beta must be > 0."
            )
        bounds = spec.get("bounds")
        if bounds is None:
            bounds = default_bounds if default_bounds is not None else (0.0, 1.0)
        low, high = _validate_bounds(bounds, dim_idx)
        return MarginalSpec(
            family="jacobi",
            offset=0.5 * (low + high),
            scale=0.5 * (high - low),
            alpha_param=alpha_p,
            beta_param=beta_p,
            bounds=(low, high),
        )

    raise ValueError(f"Dimension {dim_idx}: unknown distribution type {dist_type!r}.")


def resolve_default_bounds(nx, xlimits=None, design_space=None, xt=None):
    """Determine fallback finite bounds ``(nx, 2)`` from SMT space/bounds/data."""
    if design_space is not None:
        if not getattr(design_space, "is_all_cont", True) and not getattr(
            design_space, "is_continuous", True
        ):
            # Support continuous FloatVariable spaces; check variable types explicitly
            pass
        vars_list = getattr(design_space, "design_variables", None)
        if vars_list is not None:
            from smt.design_space import FloatVariable

            if len(vars_list) != nx:
                raise ValueError(
                    f"design_space has {len(vars_list)} variables, expected {nx}."
                )
            for v in vars_list:
                if not isinstance(v, FloatVariable):
                    raise ValueError(
                        "PCE only supports continuous FloatVariable design spaces."
                    )
        bounds = np.asarray(design_space.get_num_bounds(), dtype=float)
        if bounds.shape != (nx, 2):
            raise ValueError(
                f"Expected design_space bounds shape ({nx}, 2), got {bounds.shape}."
            )
        return bounds

    if xlimits is not None:
        limits = np.asarray(xlimits, dtype=float)
        if limits.shape != (nx, 2):
            raise ValueError(f"xlimits must have shape ({nx}, 2), got {limits.shape}.")
        return limits

    if xt is not None:
        mins = np.min(xt, axis=0)
        maxs = np.max(xt, axis=0)
        bounds = np.empty((nx, 2), dtype=float)
        for j in range(nx):
            if maxs[j] > mins[j]:
                bounds[j] = [mins[j], maxs[j]]
            else:
                bounds[j] = [mins[j] - 0.5, mins[j] + 0.5]
        return bounds

    return np.tile(np.array([-1.0, 1.0]), (nx, 1))


def parse_distributions(
    nx, distributions=None, xlimits=None, design_space=None, xt=None
):
    """Parse ``distributions`` option into a list of ``nx`` ``MarginalSpec`` objects."""
    default_bounds = resolve_default_bounds(
        nx=nx, xlimits=xlimits, design_space=design_space, xt=xt
    )

    if distributions is None:
        return [
            _parse_single_marginal(None, j, tuple(default_bounds[j])) for j in range(nx)
        ]

    # Single distribution spec applied uniformly across all dimensions
    is_per_dim = (
        isinstance(distributions, (list, tuple))
        and len(distributions) == nx
        and not (
            isinstance(distributions, tuple)
            and len(distributions) > 0
            and isinstance(distributions[0], str)
            and dist_name_like(distributions[0])
        )
    )
    if is_per_dim:
        return [
            _parse_single_marginal(distributions[j], j, tuple(default_bounds[j]))
            for j in range(nx)
        ]

    if isinstance(distributions, (list, tuple)) and not (
        isinstance(distributions, tuple)
        and len(distributions) > 0
        and isinstance(distributions[0], str)
        and dist_name_like(distributions[0])
    ):
        raise ValueError(
            f"Expected {nx} marginal distributions, got {len(distributions)}."
        )

    return [
        _parse_single_marginal(distributions, j, tuple(default_bounds[j]))
        for j in range(nx)
    ]


def dist_name_like(value: str) -> bool:
    """Return True if a tuple's leading string looks like a named distribution type."""
    return value.strip().lower() in (
        "uniform",
        "u",
        "normal",
        "gaussian",
        "n",
        "gamma",
        "g",
        "exponential",
        "expon",
        "exp",
        "beta",
        "b",
    )


def generate_multi_indices(
    ndim: int,
    degree: int,
    q_norm: float = 1.0,
    max_interaction: int = None,
    anisotropy_weights: np.ndarray = None,
    max_basis: int = None,
) -> np.ndarray:
    """
    Generate hyperbolic ``q``-norm truncated multi-indices up to total degree ``p``.

    Multi-indices are sorted by total degree first, with index 0 equal to
    ``(0, ..., 0)``. Uses DFS backtracking to avoid exponential Cartesian growth.
    """
    if ndim < 1:
        raise ValueError(f"ndim must be >= 1, got {ndim}.")
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}.")
    if q_norm <= 0.0 or q_norm > 1.0:
        raise ValueError(f"q_norm must lie in (0, 1], got {q_norm}.")

    if max_interaction is not None and max_interaction < 0:
        raise ValueError(f"max_interaction must be >= 0, got {max_interaction}.")
    max_inter = ndim if max_interaction is None else min(int(max_interaction), ndim)

    if anisotropy_weights is None:
        w = np.ones(ndim, dtype=float)
    else:
        w = np.asarray(anisotropy_weights, dtype=float).ravel()
        if w.shape != (ndim,) or np.any(w <= 0.0):
            raise ValueError(f"anisotropy_weights must contain {ndim} positive floats.")

    if degree == 0 or max_inter == 0:
        return np.zeros((1, ndim), dtype=int)

    tol = 1e-11
    budget_total = float(degree) ** q_norm + tol
    max_deg_per_dim = np.floor(float(degree) / w + 1e-12).astype(int)
    pow_table = [
        (w[j] * np.arange(max_deg_per_dim[j] + 1, dtype=float)) ** q_norm
        for j in range(ndim)
    ]

    indices = []
    current = [0] * ndim

    def _recurse(dim, rem_budget, rem_inter):
        if dim == ndim:
            indices.append(tuple(current))
            if max_basis is not None and len(indices) > max_basis:
                raise OverflowError("Candidate PCE basis exceeds max_basis limit.")
            return

        if rem_inter == 0:
            current[dim:] = [0] * (ndim - dim)
            indices.append(tuple(current))
            if max_basis is not None and len(indices) > max_basis:
                raise OverflowError("Candidate PCE basis exceeds max_basis limit.")
            current[dim:] = [0] * (ndim - dim)
            return

        costs = pow_table[dim]
        current[dim] = 0
        _recurse(dim + 1, rem_budget, rem_inter)

        for k in range(1, len(costs)):
            c = float(costs[k])
            if c > rem_budget:
                break
            current[dim] = k
            _recurse(dim + 1, rem_budget - c, rem_inter - 1)
        current[dim] = 0

    _recurse(0, budget_total, max_inter)
    arr = np.asarray(indices, dtype=int)
    # Sort by total degree ascending, then dimension-by-dimension descending (graded lex)
    tot_deg = np.sum(arr, axis=1)
    order = np.lexsort(tuple([-arr[:, j] for j in range(ndim - 1, -1, -1)] + [tot_deg]))
    return arr[order]


def eval_1d_tables(
    marginals, multi_indices: np.ndarray, x: np.ndarray, deriv_kx: int = None
):
    """
    Evaluate 1D orthogonal polynomial tables only for active dimensions.

    Returns ``(V_tables, dV_kx)`` where ``V_tables[j]`` has shape ``(n, max_deg_j + 1)``.
    """
    max_degs = np.max(multi_indices, axis=0)
    V_tables = {}
    dV_kx = None

    for j in range(len(marginals)):
        deg_j = int(max_degs[j])
        if deg_j == 0:
            continue
        if deriv_kx is not None and j == deriv_kx:
            V_tables[j], dV_kx = marginals[j].eval_1d(x[:, j], deg_j, with_deriv=True)
        else:
            V_tables[j] = marginals[j].eval_1d(x[:, j], deg_j, with_deriv=False)

    return V_tables, dV_kx


def assemble_basis_matrix(multi_indices: np.ndarray, V_tables: dict, n_samples: int):
    """Assemble multivariate PCE basis matrix ``Phi`` of shape ``(n_samples, n_basis)``."""
    n_basis, nx = multi_indices.shape
    Phi = np.ones((n_samples, n_basis), dtype=float)
    for j in range(nx):
        if j not in V_tables:
            continue
        col_deg = multi_indices[:, j]
        nz = col_deg > 0
        if np.any(nz):
            Phi[:, nz] *= V_tables[j][:, col_deg[nz]]
    return Phi


def assemble_basis_deriv_matrix(
    multi_indices: np.ndarray,
    V_tables: dict,
    dV_kx: np.ndarray,
    kx: int,
    n_samples: int,
):
    """Assemble derivative ``dPhi / dx_{kx}`` of shape ``(n_samples, n_basis)``."""
    n_basis, nx = multi_indices.shape
    dPhi = np.zeros((n_samples, n_basis), dtype=float)
    if dV_kx is None:
        return dPhi

    nz_kx = multi_indices[:, kx] > 0
    if not np.any(nz_kx):
        return dPhi

    active_cols = multi_indices[nz_kx]
    block = dV_kx[:, active_cols[:, kx]].copy()
    for j in range(nx):
        if j == kx or j not in V_tables:
            continue
        col_deg = active_cols[:, j]
        nz_j = col_deg > 0
        if np.any(nz_j):
            block[:, nz_j] *= V_tables[j][:, col_deg[nz_j]]
    dPhi[:, nz_kx] = block
    return dPhi


def compute_ols_and_loo(A: np.ndarray, y: np.ndarray, var_y: float, reg: float = 1e-12):
    """
    Solve OLS on ``A`` vs 1D ``y`` and evaluate analytical LOO and Blatman-Sudret MLOO.

    Uses thin QR factorization to compute leverages ``h_i`` and ``tr((A^T A)^{-1}) = ||R^{-1}||_F^2``
    stably without squaring condition numbers.
    """
    n_samples, p_act = A.shape

    if n_samples == 0:
        raise ValueError("Cannot fit PCE with zero training samples.")

    if p_act == 1:
        c0 = float(np.mean(y))
        coef = np.array([c0], dtype=float)
        if var_y <= 0.0:
            return coef, 0.0, 0.0
        if n_samples <= 1:
            return coef, np.nan, np.nan
        h = 1.0 / n_samples
        res = y - c0
        loo = float(np.mean((res / (1.0 - h)) ** 2)) / var_y
        t_corr = (n_samples / (n_samples - 1.0)) * (1.0 + 1.0 / n_samples)
        return coef, loo, t_corr * loo

    if n_samples > p_act:
        Q, R = np.linalg.qr(A, mode="reduced")
        diag_abs = np.abs(np.diag(R))
        r_max = float(np.max(diag_abs)) if diag_abs.size > 0 else 0.0
        if r_max > 0.0 and float(np.min(diag_abs)) > 1e-11 * r_max:
            qty = Q.T @ y
            coef = scipy.linalg.solve_triangular(R, qty, lower=False)
            if var_y <= 0.0:
                return coef, 0.0, 0.0
            res = y - Q @ qty
            h = np.sum(Q * Q, axis=1)
            if np.any(h >= 1.0 - 1e-10):
                return coef, np.inf, np.inf
            R_inv = scipy.linalg.solve_triangular(R, np.eye(p_act), lower=False)
            trace_inv = float(np.sum(R_inv * R_inv))
            loo_mse = float(np.mean((res / (1.0 - h)) ** 2))
            loo = loo_mse / var_y
            t_corr = (n_samples / (n_samples - p_act)) * (1.0 + trace_inv)
            return coef, loo, t_corr * loo

        # Rank-deficient active design matrix: regularized least-squares fallback
        if reg > 0.0:
            gram = A.T @ A + reg * np.eye(p_act)
            coef = np.linalg.solve(gram, A.T @ y)
        else:
            coef = np.linalg.lstsq(A, y, rcond=None)[0]
        return coef, np.inf, np.inf

    # Underdetermined or exact-interpolation regime (n_samples <= p_act)
    if reg > 0.0 and n_samples < p_act:
        # Minimum-norm Tikhonov regularized solution
        gram = A @ A.T + reg * np.eye(n_samples)
        coef = A.T @ np.linalg.solve(gram, y)
    else:
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return coef, np.nan, np.nan


def _select_best_support_from_candidates(
    Phi, y_phys, var_y, candidate_supports, early_stop, reg
):
    """Evaluate Hybrid OLS + modified LOO over ordered candidate support sets."""
    n_samples = Phi.shape[0]
    max_feasible = max(1, n_samples - 2) if n_samples >= 4 else max(1, n_samples - 1)

    coef_0, loo_0, mloo_0 = compute_ols_and_loo(Phi[:, [0]], y_phys, var_y, reg=reg)
    best_support = np.array([0], dtype=int)
    best_coef = coef_0
    best_loo = loo_0
    best_mloo = mloo_0
    best_step = 0

    no_improve = 0
    patience = max(12, len(candidate_supports) // 3)
    seen = {(0,)}

    for step_idx, supp in enumerate(candidate_supports, start=1):
        full_supp = tuple(sorted({0, *[int(s) for s in supp]}))
        if full_supp in seen:
            continue
        seen.add(full_supp)

        if len(full_supp) > max_feasible and n_samples > 1:
            break

        supp_arr = np.asarray(full_supp, dtype=int)
        coef, loo, mloo = compute_ols_and_loo(Phi[:, supp_arr], y_phys, var_y, reg=reg)

        improved = np.isfinite(mloo) and (
            not np.isfinite(best_mloo) or mloo < best_mloo * (1.0 - 1e-12)
        )
        if improved:
            best_support = supp_arr
            best_coef = coef
            best_loo = loo
            best_mloo = mloo
            best_step = step_idx
            no_improve = 0
        else:
            no_improve += 1

        if early_stop and np.isfinite(best_mloo):
            if best_mloo <= 1e-24 and step_idx >= best_step + 3:
                break
            if no_improve >= patience and step_idx >= best_step + 12:
                break

    return best_support, best_coef, best_loo, best_mloo


def _prepare_centered_candidates(Phi, y_phys):
    """Prepare zero-mean unit-L2-norm candidate matrix and centered normalized target."""
    mean_y = float(np.mean(y_phys))
    y_c = y_phys - mean_y
    var_y = float(np.mean(y_c * y_c))
    y_norm_vec = y_c / (np.sqrt(var_y) if var_y > 0.0 else 1.0)

    if Phi.shape[1] <= 1:
        return np.empty(0, dtype=int), None, y_norm_vec, var_y

    X = Phi[:, 1:]
    X_c = X - np.mean(X, axis=0, keepdims=True)
    norms = np.linalg.norm(X_c, axis=0)
    valid_mask = norms > 1e-12 * np.sqrt(Phi.shape[0])
    valid_cols = np.flatnonzero(valid_mask) + 1
    if valid_cols.size == 0:
        return np.empty(0, dtype=int), None, y_norm_vec, var_y

    X_s = X_c[:, valid_mask] / norms[valid_mask]
    return valid_cols, X_s, y_norm_vec, var_y


def _fit_sparse_single_output(Phi, y_phys, method: str, early_stop: bool, reg: float):
    """Fit a single output column using the specified PCE solver."""
    n_samples, n_basis = Phi.shape
    mean_y = float(np.mean(y_phys))
    var_y = float(np.mean((y_phys - mean_y) ** 2))

    # Constant target output or N <= 1 or single constant basis term
    if var_y == 0.0 or n_samples <= 1 or n_basis == 1:
        coef_0, loo_0, mloo_0 = compute_ols_and_loo(Phi[:, [0]], y_phys, var_y, reg=reg)
        return np.array([0], dtype=int), coef_0, loo_0, mloo_0

    if method == "ols":
        support = np.arange(n_basis, dtype=int)
        coef, loo, mloo = compute_ols_and_loo(Phi, y_phys, var_y, reg=reg)
        return support, coef, loo, mloo

    valid_cols, X_s, y_norm, _ = _prepare_centered_candidates(Phi, y_phys)
    if valid_cols.size == 0:
        coef_0, loo_0, mloo_0 = compute_ols_and_loo(Phi[:, [0]], y_phys, var_y, reg=reg)
        return np.array([0], dtype=int), coef_0, loo_0, mloo_0

    max_feasible = max(1, n_samples - 2) if n_samples >= 4 else max(1, n_samples - 1)
    max_cand = min(valid_cols.size, max(1, max_feasible - 1))

    candidate_supports = []

    if method in ("lars", "lasso"):
        sklearn_method = "lar" if method == "lars" else "lasso"
        _, _, coefs = lars_path(
            X_s,
            y_norm,
            method=sklearn_method,
            max_iter=max(max_cand + 5, 1),
        )
        for k in range(1, coefs.shape[1]):
            nz = np.flatnonzero(np.abs(coefs[:, k]) > 0.0)
            if nz.size == 0:
                continue
            candidate_supports.append(valid_cols[nz])
    elif method == "omp":
        # Greedy Orthogonal Matching Pursuit with modified Gram-Schmidt
        residual = y_norm.copy()
        selected = []
        used = np.zeros(valid_cols.size, dtype=bool)
        Q_list = []

        for _ in range(max_cand):
            corr = np.abs(X_s.T @ residual)
            corr[used] = -1.0
            best_idx = int(np.argmax(corr))
            if corr[best_idx] <= 1e-12:
                break
            used[best_idx] = True
            selected.append(int(valid_cols[best_idx]))
            candidate_supports.append(list(selected))

            # Incremental Modified Gram-Schmidt step
            q = X_s[:, best_idx].copy()
            for q_prev in Q_list:
                q -= float(q_prev @ q) * q_prev
            q_norm = float(np.linalg.norm(q))
            if q_norm <= 1e-12:
                break
            q /= q_norm
            Q_list.append(q)
            residual = residual - float(q @ residual) * q
            if float(np.linalg.norm(residual)) <= 1e-12:
                break
    else:
        raise ValueError(f"Unsupported sparse solver method: {method!r}.")

    return _select_best_support_from_candidates(
        Phi, y_phys, var_y, candidate_supports, early_stop=early_stop, reg=reg
    )


def fit_pce_at_basis(
    Phi, y_2d, method: str, weights=None, early_stop: bool = True, reg: float = 1e-12
):
    """
    Fit all ``ny`` output columns on a fixed candidate design matrix ``Phi``.

    Returns dict with keys ``support``, ``coef`` (shape ``(n_active, ny)``),
    ``loo`` (shape ``(ny,)``), ``mloo`` (shape ``(ny,)``), and ``mean_mloo``.
    """
    n_samples, n_basis = Phi.shape
    ny = y_2d.shape[1]

    if method == "quadrature":
        if weights is None:
            w = np.full(n_samples, 1.0 / n_samples, dtype=float)
        else:
            w = np.asarray(weights, dtype=float).ravel()
            if w.shape != (n_samples,):
                raise ValueError(
                    f"Quadrature weights must have shape ({n_samples},), got {w.shape}."
                )
            w_sum = float(np.sum(w))
            if w_sum <= 0.0:
                raise ValueError("Sum of quadrature weights must be strictly positive.")
            w = w / w_sum

        coef = Phi.T @ (w[:, None] * y_2d)
        support = np.arange(n_basis, dtype=int)
        # In-sample normalized residual diagnostic (exact projection lacks OLS hat)
        pred = Phi @ coef
        loo_arr = np.empty(ny, dtype=float)
        for m in range(ny):
            var_m = float(np.var(y_2d[:, m]))
            if var_m <= 0.0:
                loo_arr[m] = 0.0
            else:
                loo_arr[m] = float(np.mean((y_2d[:, m] - pred[:, m]) ** 2)) / var_m
        return {
            "support": support,
            "coef": coef,
            "loo": loo_arr,
            "mloo": loo_arr.copy(),
            "mean_mloo": float(np.mean(loo_arr)),
        }

    supports = []
    coefs = []
    loo_arr = np.empty(ny, dtype=float)
    mloo_arr = np.empty(ny, dtype=float)

    for m in range(ny):
        supp_m, coef_m, loo_m, mloo_m = _fit_sparse_single_output(
            Phi, y_2d[:, m], method=method, early_stop=early_stop, reg=reg
        )
        supports.append(supp_m)
        coefs.append(coef_m)
        loo_arr[m] = loo_m
        mloo_arr[m] = mloo_m

    # Unify active supports across outputs into a sorted union matrix (n_active, ny)
    union_supp = np.unique(np.concatenate(supports)).astype(int)
    coef_full = np.zeros((union_supp.size, ny), dtype=float)
    pos_map = {int(idx): pos for pos, idx in enumerate(union_supp)}
    for m in range(ny):
        for local_i, global_idx in enumerate(supports[m]):
            coef_full[pos_map[int(global_idx)], m] = coefs[m][local_i]

    finite_mloo = mloo_arr[np.isfinite(mloo_arr)]
    mean_mloo = float(np.mean(finite_mloo)) if finite_mloo.size > 0 else np.inf
    return {
        "support": union_supp,
        "coef": coef_full,
        "loo": loo_arr,
        "mloo": mloo_arr,
        "mean_mloo": mean_mloo,
    }


def select_adaptive_pce(
    marginals,
    xt: np.ndarray,
    yt: np.ndarray,
    degrees,
    q_norms,
    method: str,
    max_interaction=None,
    anisotropy_weights=None,
    weights=None,
    early_stop: bool = True,
    loo_target: float = 1e-6,
    max_basis: int = 10000,
    reg: float = 1e-12,
):
    """
    Search over candidate ``(q_norm, degree)`` combinations and fit PCE coefficients.

    Caches 1D univariate polynomial evaluations across candidate degrees.
    """
    nx = len(marginals)
    n_samples = xt.shape[0]
    max_cand_deg = max(degrees)

    # Precompute 1D tables once up to maximum candidate degree
    V_cache = {
        j: marginals[j].eval_1d(xt[:, j], max_cand_deg, with_deriv=False)
        for j in range(nx)
    }

    best_result = None
    best_mloo = np.inf

    for q in q_norms:
        no_improve_deg = 0
        best_q_mloo = np.inf
        for idx_d, p in enumerate(degrees):
            try:
                cand_indices = generate_multi_indices(
                    ndim=nx,
                    degree=p,
                    q_norm=q,
                    max_interaction=max_interaction,
                    anisotropy_weights=anisotropy_weights,
                    max_basis=max_basis,
                )
            except OverflowError:
                if best_result is None and idx_d == 0:
                    raise ValueError(
                        f"Basis size at degree={p}, q_norm={q} in {nx}D exceeds max_basis={max_basis}. "
                        "Reduce degree, lower q_norm, set max_interaction, or increase max_basis."
                    )
                warnings.warn(
                    f"Stopping degree sweep before degree={p} because candidate basis exceeds max_basis={max_basis}.",
                    UserWarning,
                    stacklevel=3,
                )
                break

            Phi = assemble_basis_matrix(cand_indices, V_cache, n_samples)
            fit = fit_pce_at_basis(
                Phi, yt, method=method, weights=weights, early_stop=early_stop, reg=reg
            )

            score = fit["mean_mloo"]
            is_better = best_result is None or (
                np.isfinite(score)
                and (not np.isfinite(best_mloo) or score < best_mloo * (1.0 - 1e-10))
            )
            if is_better:
                best_mloo = score
                best_result = {
                    "degree": int(p),
                    "q_norm": float(q),
                    "n_basis_candidate": int(cand_indices.shape[0]),
                    "multi_indices": cand_indices[fit["support"]],
                    "coefficients": fit["coef"],
                    "loo": fit["loo"],
                    "mloo": fit["mloo"],
                }

            if np.isfinite(score) and (
                not np.isfinite(best_q_mloo) or score < best_q_mloo * (1.0 - 1e-10)
            ):
                best_q_mloo = score
                no_improve_deg = 0
            else:
                no_improve_deg += 1

            if early_stop and len(degrees) > 1:
                if np.isfinite(best_q_mloo) and best_q_mloo <= loo_target:
                    break
                if no_improve_deg >= 2 and np.isfinite(best_q_mloo):
                    break

    return best_result


def build_pce_jax_predictor(
    marginals,
    multi_indices: np.ndarray,
    coefficients: np.ndarray,
    ny: int,
    dtype=None,
):
    """Construct a pure-JAX end-to-end differentiable PCE evaluation callable."""
    for j, m in enumerate(marginals):
        if not m.is_askey:
            raise NotImplementedError(
                f"JAX predictor requires affine Askey marginal distributions; dimension {j} uses {m.family!r}."
            )

    import jax
    import jax.numpy as jnp

    if dtype is None:
        use_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    elif isinstance(dtype, str):
        use_dtype = jnp.float32 if "32" in dtype else jnp.float64
    else:
        use_dtype = jnp.dtype(dtype)

    coef_j = jnp.asarray(coefficients, dtype=use_dtype)
    squeeze_y = ny == 1

    max_degs = np.max(multi_indices, axis=0)
    used_dims = [int(j) for j in range(len(marginals)) if max_degs[j] > 0]

    # Precompute static Favard recurrence tables and active index arrays per used dimension
    dim_specs = []
    for j in used_dims:
        max_d_j = int(max_degs[j])
        a_j, b_j = marginals[j].recurrence_coefficients(max_d_j)
        dim_specs.append(
            (
                j,
                max_d_j,
                jnp.asarray(marginals[j].offset, dtype=use_dtype),
                jnp.asarray(marginals[j].scale, dtype=use_dtype),
                tuple(float(v) for v in a_j),
                tuple(float(v) for v in b_j),
                jnp.asarray(multi_indices[:, j], dtype=jnp.int32),
            )
        )

    c0_j = coef_j[0]

    def predict_fn(x):
        x_in = jnp.asarray(x, dtype=use_dtype)
        is_1d = x_in.ndim == 1
        x_arr = jnp.atleast_2d(x_in)

        if not dim_specs:
            # Pure constant polynomial chaos expansion
            y = jnp.broadcast_to(c0_j[None, :], (x_arr.shape[0], ny))
        else:
            phi_terms = None
            for j, max_d_j, off_j, sc_j, a_tuple, b_tuple, idx_j in dim_specs:
                z_j = (x_arr[:, j] - off_j) / sc_j
                cols_1d = [jnp.ones_like(z_j)]
                for k in range(max_d_j):
                    term = (z_j - a_tuple[k]) * cols_1d[k]
                    if k > 0:
                        term = term - b_tuple[k - 1] * cols_1d[k - 1]
                    cols_1d.append(term / b_tuple[k])
                V_j = jnp.stack(cols_1d, axis=1)
                factor = V_j[:, idx_j]
                phi_terms = factor if phi_terms is None else phi_terms * factor
            y = jnp.matmul(phi_terms, coef_j)

        if is_1d:
            return y[0, 0] if squeeze_y else y[0]
        return y

    return predict_fn
