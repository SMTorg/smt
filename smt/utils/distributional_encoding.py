import numpy as np


class DistributionalEncoder:
    def __init__(self, beta=1.0, n_quantiles=100):
        """
        Implements Distributional Encoding using the Wasserstein W2 distance.

        Parameters
        ----------
        beta : float
            Power parameter for the substitution kernel (exp(-gamma * W^beta)).
            Must be in (0, 2] to ensure positive definiteness.
        n_quantiles : int
            Number of points to sample from the quantile function.
        """
        if not (0 < beta <= 2.0):
            raise ValueError(
                "beta must be in (0, 2] for a valid positive definite kernel."
            )

        self.beta = beta
        self.n_quantiles = n_quantiles
        self.level_distributions = {}  # feature_idx -> {level -> quantiles}
        self.point_distributions = {}  # feature_idx -> np.ndarray (n_samples, n_quantiles)
        self.global_quantiles = None
        self.q_grid = np.linspace(0, 1, n_quantiles)

    def fit_distributions(
        self, X_cat, y_train, cat_features_indices, is_acting=None, loo=True
    ):
        """
        Extracts empirical distributions for each categorical level.

        If loo=True, it also computes Leave-One-Out distributions for each training point
         to prevent target leakage and ensure mathematical rigor.

        If is_acting is provided, it ensures only points where the feature is acting
        contribute to the distributions.
        """
        self.level_distributions = {}
        self.point_distributions = {}
        n_samples = X_cat.shape[0]

        # Precompute global distribution as fallback (using only acting points if available)
        y_acting_global = y_train.flatten()
        self.global_quantiles = np.quantile(y_acting_global, self.q_grid)

        for i, feat_idx in enumerate(cat_features_indices):
            self.level_distributions[feat_idx] = {}
            self.point_distributions[feat_idx] = np.zeros((n_samples, self.n_quantiles))

            unique_levels = np.unique(X_cat[:, i])

            # Mask for points where this feature is acting
            feat_acting_mask = np.ones(n_samples, dtype=bool)
            if is_acting is not None:
                feat_acting_mask = is_acting[:, feat_idx]

            for level in unique_levels:
                # points matching level AND acting
                mask = (X_cat[:, i] == level) & feat_acting_mask
                indices = np.where(mask)[0]
                y_subset = y_train[mask].flatten()

                # 1. Full level distribution (for prediction and general use)
                if len(y_subset) < 1:
                    level_q = self.global_quantiles
                else:
                    level_q = np.quantile(y_subset, self.q_grid)
                self.level_distributions[feat_idx][level] = level_q

                # 2. LOO distributions (for training rigor)
                if loo:
                    if len(y_subset) <= 1:
                        # Fallback to level_q if we can't do LOO (only 1 sample)
                        self.point_distributions[feat_idx][indices] = level_q
                    else:
                        for idx_in_level, global_idx in enumerate(indices):
                            y_loo = np.delete(y_subset, idx_in_level)
                            self.point_distributions[feat_idx][global_idx] = (
                                np.quantile(y_loo, self.q_grid)
                            )
                else:
                    self.point_distributions[feat_idx][indices] = level_q

            # For points where feature is NOT acting, we might assign global or zeros
            not_acting_indices = np.where(~feat_acting_mask)[0]
            self.point_distributions[feat_idx][not_acting_indices] = (
                self.global_quantiles
            )

    def get_w2_distance(self, feat_idx, level_a, level_b):
        """Computes W2 distance between two levels using their full distributions."""
        q_a = self.level_distributions[feat_idx].get(level_a, self.global_quantiles)
        q_b = self.level_distributions[feat_idx].get(level_b, self.global_quantiles)

        w2_sq = np.mean((q_a - q_b) ** 2)
        return np.sqrt(max(0, w2_sq)) ** self.beta

    def compute_distance_matrix(self, feat_idx, levels_i, levels_j=None):
        """
        Computes the W2 distance matrix for a specific feature using level distributions.
        Used primarily for prediction.
        """
        n = len(levels_i)
        if levels_j is None:
            # Training case: compute upper triangle
            m = n
            dist_matrix = np.zeros((n, m))
            for i in range(n):
                for j in range(i + 1, m):
                    d = self.get_w2_distance(feat_idx, levels_i[i], levels_i[j])
                    dist_matrix[i, j] = dist_matrix[j, i] = d
        else:
            # Prediction case
            m = len(levels_j)
            dist_matrix = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    dist_matrix[i, j] = self.get_w2_distance(
                        feat_idx, levels_i[i], levels_j[j]
                    )

        return dist_matrix
