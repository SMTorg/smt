import numpy as np
try:
    import torch
    from smt.utils.nn_rich_rbf import (
        rbf_features,
        rbf_features_grad,
    )

    RBFGEN_AVAILABLE = True
except ImportError:
    RBFGEN_AVAILABLE = False


class LossTerm:
    def __init__(self, x_train, loss_term_weight=1.):
        self.x_train = x_train
        self.loss_term_weight = loss_term_weight

    def setup(self, rbf_surrogate):
        """
        Setup the loss term (e.g., build probes, evaluate RBF bases).
        """
        pass

    def __call__(self, W):
        """
        Calculate the loss given the network weights W.
        """
        raise NotImplementedError("LossTerm subclasses must implement __call__")

    def sample_within_convex_hull(self, n_pts, rng=None):
        """
        Sample n_pts points randomly within the convex hull of self.x_train.
        Uses a Dirichlet distribution to generate weights for a convex combination used to form the points.
        """
        if rng is None:
            rng = np.random.default_rng(1)

        if self.x_train is None:
             raise ValueError("x_train must be set to sample within convex hull.")

        n_samples, n_dim = self.x_train.shape
        
        # For each probe point, select k = n_dim + 1 points from x_train
        # and form a convex combination.
        # This guarantees membership in the convex hull.
        k = n_dim + 1
        
        # Select random indices for each probe point
        # Shape: (n_pts, k)
        indices = rng.integers(0, n_samples, size=(n_pts, k))
        
        # Gather the points
        # Shape: (n_pts, k, n_dim)
        selected_points = self.x_train[indices]
        
        # Generate random weights (Dirichlet distribution essentially)
        # Shape: (n_pts, k)
        weights = rng.exponential(size=(n_pts, k))
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Compute convex combination: sum(w_i * p_i)
        # Shape: (n_pts, n_dim)
        # einsum: 'nk, nkd -> nd'
        points = np.einsum('nk,nkd->nd', weights, selected_points)
        return points


class SliceBasedPriorLossTerm(LossTerm):
    """
    Penalizes deviations from a user-defined prior normal distribution at specific points 
    (e.g., along 1D slices). The loss uses the Wasserstein distance 
    between the modeled ensemble's normal distribution and the user's prior normal distribution.
    """
    def __init__(self, x_train, prior_points, prior_means, prior_stds, loss_term_weight=1.):
        super().__init__(x_train, loss_term_weight)
        self.prior_points = np.atleast_2d(prior_points)
        self.prior_means = np.atleast_1d(prior_means)
        self.prior_stds = np.atleast_1d(prior_stds)
        
        self.Phi_prior = None
        self.prior_means_t = None
        self.prior_vars_t = None

    def setup(self, rbf_surrogate):
        self.Phi_prior = torch.tensor(
            rbf_features(self.prior_points, rbf_surrogate.rbf_centers, rbf_surrogate.d0),
            dtype=torch.float32
        )
        self.prior_means_t = torch.tensor(self.prior_means, dtype=torch.float32)
        self.prior_vars_t = torch.tensor(self.prior_stds**2, dtype=torch.float32)

    def __call__(self, W):
        f = W @ self.Phi_prior.T
        
        mu_1 = f.mean(dim=0)
        std_1 = f.std(dim=0, unbiased=False) 

        mu_0 = torch.tensor(self.prior_means, dtype=torch.float32)
        std_0 = torch.tensor(self.prior_stds, dtype=torch.float32)

        # Use Huber loss. This acts like the Wasserstein distance near convergence
        # but prevents explosive quadratic gradients during initial epochs.
        loss_mu = torch.nn.functional.huber_loss(mu_1, mu_0, delta=1.0, reduction='sum')
        loss_std = torch.nn.functional.huber_loss(std_1, std_0, delta=1.0, reduction='sum')
        
        return loss_mu + loss_std



class MonotonicityLossTerm(LossTerm):
    """
    Penalizes violations of monotonicity in the surrogate model with respect to
    specified input dimensions. It evaluates the exact analytical gradient of the 
    surrogate at various points and applies a penalty if the gradient contradicts 
    the given target sign (e.g., enforcing strictly increasing or strictly decreasing relationships).
    """
    def __init__(self, x_train, sign=1, stepsize_frac=None, mono_pts_per_input_dim=5, random_base_points=False, inside_convex_hull=False, input_indices=None, loss_term_weight=1., **kwargs):
        super().__init__(x_train, loss_term_weight)
        self.mono_pts_per_input_dim = mono_pts_per_input_dim
        self.random_base_points = random_base_points
        self.inside_convex_hull = inside_convex_hull
        self.sign = sign
        self.input_indices = input_indices

        self.base_points_list = None
        self.target_dims_list = None
        self.rbf_grad_evals = None

    def setup(self, rbf_surrogate):
        self.build_mono_points()
        self.eval_rbf_grad_in_mono_pts(rbf_surrogate.rbf_centers, rbf_surrogate.d0)

    def build_mono_points(self, rng=None):
        if rng is None:
            rng = np.random.default_rng(1)
        lo = self.x_train.min(axis=0); hi = self.x_train.max(axis=0)
        
        indices = self.input_indices if self.input_indices is not None else range(self.x_train.shape[1])
        
        X_list = []
        dims_list = []
        
        for i in indices:
            if self.random_base_points:
                if self.inside_convex_hull:
                     base = self.sample_within_convex_hull(self.mono_pts_per_input_dim, rng=rng)
                else:
                     base = lo + (hi - lo) * rng.random((self.mono_pts_per_input_dim, self.x_train.shape[1]))
            else:
                 idx = rng.integers(0, len(self.x_train), size=self.mono_pts_per_input_dim)
                 base = self.x_train[idx].copy()

            X_list.append(base)
            dims_list.append(np.full(self.mono_pts_per_input_dim, i))
        
        self.base_points_list = np.vstack(X_list)
        self.target_dims_list = np.concatenate(dims_list)

    def eval_rbf_grad_in_mono_pts(self, rbf_centers, rbf_d0):
        grad_Phi = rbf_features_grad(self.base_points_list, rbf_centers, rbf_d0, self.target_dims_list)
        self.rbf_grad_evals = torch.tensor(grad_Phi, dtype=torch.float32)

    def __call__(self, W, beta=100):
        f_grad = W @ self.rbf_grad_evals.T 
        loss_mono = torch.nn.functional.softplus(-self.sign * f_grad, beta=beta).mean()
        return loss_mono


class PositivityLossTerm(LossTerm):
    """
    Penalizes negative prediction values from the surrogate model. 
    It probes the surrogate at randomly sampled points across the allowed space 
    (or within the convex hull of the training data) and applies a penalty for any point 
    where the predicted output is less than zero.
    """
    def __init__(self, x_train, n_pos_pts=128, loss_term_weight=1., inside_convex_hull=False):
        super().__init__(x_train, loss_term_weight)
        self.n_pos_pts = n_pos_pts
        self.inside_convex_hull = inside_convex_hull
        self.pos_probe_pts = None
        self.rbf_evals = None

    def setup(self, rbf_surrogate):
        self.build_pos_probes_with_rng()
        self.eval_rbf_basis_in_pos_probes(rbf_surrogate.rbf_centers, rbf_surrogate.d0)

    def build_pos_probes_with_rng(self, rng_seed=None):
        if rng_seed is None:
            rng_seed = np.random.default_rng(5)
        
        if self.inside_convex_hull:
            self.pos_probe_pts = self.sample_within_convex_hull(self.n_pos_pts, rng=rng_seed)
        
        else:
            lo = self.x_train.min(axis=0); hi = self.x_train.max(axis=0)
            self.pos_probe_pts = lo + (hi - lo) * rng_seed.random((self.n_pos_pts, self.x_train.shape[1]))

    def eval_rbf_basis_in_pos_probes(self, rbf_centers, rbf_d0):
        if self.pos_probe_pts is None:
            raise ValueError("Positivity probes need to be built before they can be evaluated")
        
        self.rbf_evals = torch.tensor(rbf_features(self.pos_probe_pts, rbf_centers, rbf_d0), dtype=torch.float32)

    def __call__(self, W, beta=100):
        f_pos = W @ self.rbf_evals.T
        loss_pos = torch.nn.functional.softplus(-f_pos, beta=beta).mean()
        # loss_pos = torch.nn.functional.relu( -f_pos).mean()
        return loss_pos
