"""
Contains the RBFSurrogate class.
"""

import numpy as np
from scipy.linalg import lstsq, null_space
from scipy.spatial.distance import cdist

from smt.surrogate_models.surrogate_model import SurrogateModel

try:
    import torch
    RBFGEN_AVAILABLE = True
except ImportError:
    RBFGEN_AVAILABLE = False


def gaussian_rbf(r, eps):
    return np.exp(-(eps * r)**2)


def rbf_features(X, C, eps):
    R = cdist(np.atleast_2d(X), C, metric="euclidean")
    return gaussian_rbf(R, eps)


def rbf_features_grad(X, C, eps, target_dims):
    Phi = rbf_features(X, C, eps)
    N = Phi.shape[0]

    X_target = X[np.arange(N), target_dims][:, np.newaxis]
    C_target = C.T[target_dims, :]

    diff = X_target - C_target
    grad_Phi = -2.0 * (eps**2) * diff * Phi

    return grad_Phi


def median_eps(X):
    D = cdist(X, X)
    vals = D[np.triu_indices_from(D, k=1)]
    vals = vals[vals > 0]
    return 1.0 / (np.sqrt(2) * (np.median(vals) if vals.size else 1.0))

# =========================
# LHS sampler
# =========================
def lhs(n, dim, low, high, rng):
    cut = np.linspace(0, 1, n + 1)
    u = rng.uniform(0, 1, size=(n, dim))
    a = cut[:-1][:, None]
    bnds = cut[1:][:, None]
    X01 = a + (bnds - a) * u
    for j in range(dim):
        rng.shuffle(X01[:, j])
    return low + (high - low) * X01


class NNRichRBF(SurrogateModel):
    """
    RBF Surrogate model that generates a nullspace for InfoGAN.
    """

    name = "NNRichRBF"

    def _initialize(self):
        if not RBFGEN_AVAILABLE:
            raise RuntimeError(
                'RBFGen not available. Please install RBFGen dependencies with: pip install smt["rbfgen"]'
            )

        super()._initialize()
        declare = self.options.declare

        declare(
            "m_centers",
            None,
            types=(int, type(None)),
            desc="Number of RBF centers. If None, defaults to max(3*[number of training points], 100).",
        )
        declare(
            "d0",
            None,
            types=(float, int, type(None)),
            desc="RBF width (epsilon). If None, computed via median heuristic.",
        )
        declare(
            "rng_seed",
            None,
            types=(int, np.random.Generator, type(None)),
            desc="Random seed or generator for center selection.",
        )
        declare(
            "centers_distribution",
            "random",
            values=("random", "linspace"),
            desc="Distribution of RBF centers: 'random' (uniform random) or 'linspace' (regular grid).",
        )

        # Attributes specific to this implementation
        self.rbf_centers = None
        self.d0 = None
        self.data_interp_coeffs = None
        self.nullspace = None
        self.rbf_evals = None

    def _train(self):
        """
        Train the model: select centers, compute features, solve for weights, and compute nullspace.
        """
        xt, yt = self.training_points[None][0]

        # Ensure xt, yt are proper shapes
        # xt is (nt, nx), yt is (nt, ny)

        m_centers = self.options["m_centers"]
        d0 = self.options["d0"]
        rng_seed = self.options["rng_seed"]

        # Helper implementation from original code
        self._build_rbf_nullspace(xt, yt, m_centers, d0, rng_seed)

    def _build_rbf_nullspace(self, Xtr, ytr, m_centers=None, eps=None, rng=None):
        ntr, d = Xtr.shape

        # Handle RNG
        if rng is None:
            rng = np.random.default_rng(0)
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)

        if m_centers is None:
            m_centers = max(3 * ntr, 100)  # rich basis -> nontrivial nullspace

        lo = Xtr.min(axis=0)
        hi = Xtr.max(axis=0)

        # Select centers
        centers_dist = self.options["centers_distribution"]

        if centers_dist == "linspace":
            # Grid of centers
            # k^d approx m_centers => k = m_centers^(1/d)
            k = int(np.round(m_centers**(1/d)))
            # Ensure at least 2 points per dim if m_centers is large enough, else k=1
            if k < 2:
                k = 2

            # Generate 1D linspaces
            ranges = [np.linspace(lo[i], hi[i], k) for i in range(d)]
            # Meshgrid
            mesh = np.meshgrid(*ranges, indexing='ij')
            # Flatten to (N, d)
            C = np.stack([m.flatten() for m in mesh], axis=-1)
            # Update m_centers to actual number of points
            m_centers = C.shape[0] # Not strictly necessary to update option, but C is what matters
        else:
            # Random uniform, but include training points
            n_random = m_centers - ntr
            if n_random > 0:
                # C_rand = lo + (hi - lo) * rng.random((n_random, d))
                C_rand = lhs(n_random, d, lo, hi, rng)  # use latin hypercube sampling to generate the rbf centers
                C = np.vstack([Xtr, C_rand])
            else:
                # If m_centers is less than ntr, we just use Xtr
                C = Xtr.copy()

        self.rbf_centers = C

        # Determine epsilon
        if eps is None:
            eps = median_eps(C)
        self.d0 = eps

        # Compute features
        Phi = rbf_features(Xtr, C, eps)
        self.rbf_evals = Phi

        # Solve for particular solution (weights)
        # lstsq returns (x, residues, rank, s)
        w_p, residues, rank, s = lstsq(Phi, ytr)
        self.data_interp_coeffs = torch.tensor(w_p[:, 0], dtype=torch.float32)

        # Compute nullspace
        N = null_space(Phi)
        self.nullspace = torch.tensor(N, dtype=torch.float32)

        # Recursive retry if nullspace is empty
        if N.shape[1] == 0:
            # Increase centers and retry
            new_m_centers = int(1.5 * m_centers)
            self._build_rbf_nullspace(Xtr, ytr, m_centers=new_m_centers, eps=eps, rng=rng)

    def _predict_values(self, x):
        """
        Predict values using the particular solution.
        """
        Phi_q = rbf_features(x, self.rbf_centers, self.d0)

        w_p_np = self.data_interp_coeffs.cpu().numpy()
        y_pred = Phi_q @ w_p_np
        return y_pred
