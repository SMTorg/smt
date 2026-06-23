import numpy as np
from smt.surrogate_models.surrogate_model import SurrogateModel

from smt.utils.nn_rich_rbf import (
    NNRichRBF,
    rbf_features,
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    RBFGEN_AVAILABLE = True
    BaseGenerator = nn.Module
except ImportError:
    RBFGEN_AVAILABLE = False
    BaseGenerator = object

class Generator(BaseGenerator):
    def __init__(self, zdim, rdim, hidden=64):
        if not RBFGEN_AVAILABLE:
            raise RuntimeError(
                'RBFGen not available. Please install RBFGen dependencies with: pip install smt["rbfgen"]'
            )

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, rdim)
        )
        
        # Initialize weights and biases with values drawn from uniform distribution
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight)
                if m.bias is not None:
                    nn.init.uniform_(m.bias)

    def forward(self, z):
        return self.net(z)


class RBFGen(SurrogateModel):
    name = "RBFGen"

    def _initialize(self):
        if not RBFGEN_AVAILABLE:
            raise RuntimeError(
                'RBFGen not available. Please install RBFGen dependencies with: pip install smt["rbfgen"]'
            )
            
        super()._initialize()
        declare = self.options.declare
        
        declare("rbf_surrogate", None, types=(NNRichRBF, type(None)), desc="The RBF surrogate object")
        declare(
            "rbf_m_centers",
            None,
            types=(int, type(None)),
            desc="Number of RBF centers. If None, defaults to max(3*[number of training points], 100).",
        )
        declare(
            "rbf_d0",
            None,
            types=(float, int, type(None)),
            desc="RBF width (epsilon). If None, computed via median heuristic.",
        )
        declare(
            "rbf_rng_seed",
            1,
            types=(int, np.random.Generator, type(None)),
            desc="Random seed or generator for center selection.",
        )
        declare(
            "rbf_centers_distribution",
            "random",
            values=("random", "linspace"),
            desc="Distribution of RBF centers: 'random' (uniform random) or 'linspace' (regular grid).",
        )
        declare("learning_rate", 1e-3, types=(float), desc="Learning rate for the network optimizer")
        declare("alpha_scale", 1.0, types=(float), desc="Scaling factor for alpha")
        declare("epochs", 1000, types=(int), desc="Number of training epochs")
        declare("batch_size", 64, types=(int), desc="Batch size for training")
        declare("latent_space_dim", 12, types=(int), desc="Dimension of the latent space")
        declare("num_eval_pts", 100, types=(int), desc="Number of evaluation points for nullspace")


        self.supports["variances"] = True

        self.generator = None
        self.opt_generator = None
        self.network_weights = None
        
        self.loss_terms = []
        
        # We'll need to store training data because add_... methods use it
        self.x_train = None
        self.y_train = None

    def _train(self, K_ensemble=150):
        """
        Train the model
        """
        # Ensure RBF surrogate is ready
        rbf = self.options["rbf_surrogate"]
        if rbf is None:
            rbf_m_centers = self.options["rbf_m_centers"]
            rbf_d0 = self.options["rbf_d0"]
            rbf_rng_seed = self.options["rbf_rng_seed"]
            rbf_centers_dist = self.options["rbf_centers_distribution"]

            rbf = NNRichRBF(
                m_centers=rbf_m_centers, d0=rbf_d0, rng_seed=rbf_rng_seed, centers_distribution=rbf_centers_dist
            )
            # If creating a fresh one, we need to train it using our data
            # xt, yt are available in self.training_points after set_training_values is called
            # and SurrogateModel.train() calls _train()
            xt, yt = self.training_points[None][0]
            rbf.set_training_values(xt, yt)
            rbf.train()
            self.options["rbf_surrogate"] = rbf
        
        self.x_train, self.y_train = self.training_points[None][0]
        
        # Setup loss terms
        for loss_term in self.loss_terms:
            loss_term.setup(rbf)

        # Define network
        nullspace_dim = rbf.nullspace.shape[1]
        latent_dim = self.options["latent_space_dim"]
        self.generator = Generator(zdim=latent_dim, rdim=nullspace_dim).train()
        
        # Optimizer
        lr = self.options["learning_rate"]
        self.opt_generator = optim.Adam(self.generator.parameters(), lr=lr)
        
        # Training loop
        epochs = self.options["epochs"]
        batchsize = self.options["batch_size"]
        alpha_scale = self.options["alpha_scale"]
        
        for ep in range(epochs):
            latent_vars = torch.randn(batchsize, latent_dim)
            alpha = self.generator(latent_vars)
            W = (rbf.data_interp_coeffs + (alpha @ rbf.nullspace.T) * alpha_scale) # (B, m)

            total_loss = torch.tensor(0.0)

            # Loss Terms
            current_losses = {}
            for loss_term in self.loss_terms:
                term_loss = loss_term(W)
                # weighted_loss is tensor, keep graph
                weighted_loss = term_loss * loss_term.loss_term_weight
                total_loss += weighted_loss
                
                # For printing, extract scalar
                name = loss_term.__class__.__name__
                if name not in current_losses:
                    current_losses[name] = 0.0
                current_losses[name] += weighted_loss.item()

            self.opt_generator.zero_grad()
            total_loss.backward()
            self.opt_generator.step()

            if self.options["print_global"] and ((ep + 1) % 100 == 0 or ep == epochs - 1):
                loss_info = " | ".join([f"{name}: {val:.4e}" for name, val in current_losses.items()])
                print(f"Epoch {ep+1:5d}/{epochs} | Total Loss: {total_loss.item():.4e} | {loss_info}")

        # Sample ensemble of weights
        with torch.no_grad():
            z = torch.randn(K_ensemble, latent_dim)
            alpha = self.generator(z)
            W = (rbf.data_interp_coeffs + (alpha @ rbf.nullspace.T) * alpha_scale).cpu().numpy()

        self.network_weights = W

    def _predict_values(self, x):
        """
        Evaluates the model at a set of points.
        """
        rbf = self.options["rbf_surrogate"]
        C = rbf.rbf_centers
        eps = rbf.d0
        W = self.network_weights
        
        Phi_q = rbf_features(x, C, eps) 
        # W is (K, m), Phi_q is (nq, m)
        # We want mean prediction
        return (W @ Phi_q.T).mean(axis=0)

    def _predict_variances(self, x):
        """
        Predict the variances at a set of points.
        """
        rbf = self.options["rbf_surrogate"]
        C = rbf.rbf_centers
        eps = rbf.d0
        W = self.network_weights
        
        Phi_q = rbf_features(x, C, eps)
        # W is (K, m), Phi_q is (nq, m)
        
        # Compute ensemble predictions: (K, nq)
        # (W @ Phi_q.T) shape is (K, nq)
        y_ensemble = W @ Phi_q.T
        
        # Compute variance across the ensemble (axis 0)
        # Result should be (nq,) or (nq, 1) to match SMT standards (dataset dim)
        # SMT usually expects (nt, ny)
        return y_ensemble.var(axis=0).reshape(-1, 1)

    def add_loss_term(self, loss_term):
        self.loss_terms.append(loss_term)
