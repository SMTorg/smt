# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 2024

@author: Lisa Pretsch <<lisa.pretsch@tum.de>>

Based on KRG from SMT by Dr. Mohamed Amine Bouhlel

With cooperative components Kriging model fit inspired by
distributed multi-disciplinary design optimization (MDO) approaches and
Zhan, D., Wu, J., Xing, H. et al.
A cooperative approach to efficient global optimization.
J Glob Optim 88, 327–357 (2024). https://doi.org/10.1007/s10898-023-01316-6
"""

import numpy as np
from scipy import optimize

from smt.surrogate_models.krg_based import KrgBased
from smt.surrogate_models.krg_based.distances import componentwise_distance
from smt.surrogate_models.krg_based.hyperparam_optim import CobylaOptimizer


class CooperativeCobylaOptimizer(CobylaOptimizer):
    """COBYLA optimizer for cooperative component-wise Kriging.

    Only optimizes hyperparameters for active component variables,
    keeping inactive variables fixed at their current best values
    (context vector ``coop_theta``).

    Parameters
    ----------
    active_comp_var : np.ndarray[nx]
        Boolean mask selecting the active design variables.
    coop_theta : np.ndarray[nx]
        Context vector with current best hyperparameters (model space).
    uses_log_theta_space : bool
        Whether the optimizer works in log10(theta) space.
    rhobeg : float
        Initial step size for COBYLA.
    tol : float
        Convergence tolerance.
    """

    def __init__(
        self, active_comp_var, coop_theta, uses_log_theta_space, rhobeg=0.5, tol=1e-4
    ):
        super().__init__(rhobeg=rhobeg, tol=tol)
        self.active_comp_var = active_comp_var
        self.coop_theta = coop_theta.copy()
        self.uses_log_theta_space = uses_log_theta_space

    def _minimize(self, objective, x0, gradient, hessian, constraints, bounds, limit):
        # Build full-dim context in optimizer space (log10 or linear)
        if self.uses_log_theta_space:
            context = np.log10(self.coop_theta)
        else:
            context = self.coop_theta.copy()

        # Extract active-dim starting point
        active_x0 = x0[self.active_comp_var]

        # Wrap objective: active params -> full params -> original objective
        def active_objective(active_params):
            full = context.copy()
            full[self.active_comp_var] = active_params
            return objective(full)

        # Create active-dim box constraints from full-dim bounds
        active_indices = np.where(self.active_comp_var)[0]
        active_bounds = [bounds[i] for i in active_indices] if bounds else []
        active_constraints = []
        for j, (lo, hi) in enumerate(active_bounds):
            active_constraints.append(lambda x, j=j, lo=lo: x[j] - lo)
            active_constraints.append(lambda x, j=j, hi=hi: hi - x[j])

        # Run COBYLA on active dimensions only
        result = optimize.minimize(
            active_objective,
            active_x0,
            constraints=[{"fun": con, "type": "ineq"} for con in active_constraints],
            method="COBYLA",
            options={
                "rhobeg": self.rhobeg,
                "tol": self.tol,
                "maxiter": limit,
            },
        )

        # Reassemble full-dim result
        full_result = context.copy()
        full_result[self.active_comp_var] = result["x"]
        result = dict(result)
        result["x"] = full_result
        return result


class CoopCompKRG(KrgBased):
    name = "Cooperative Components Kriging"

    """
    Example
    -------
    # Example with random components
    # (use physical components if available)
    n_comp = 3

    # Random design variable to component allocation
    comps = [*range(n_comp)]
    vars = [*range(n_dim)]
    random.shuffle(vars)
    comp_var = np.full((n_dim, n_comp), False)
    for c in comps:
        comp_size = int(n_dim/n_comp)
        start = c*comp_size
        end = (c+1)*comp_size
        if c+1 == n_comp:
            end = max((c+1)*comp_size, n_dim)
        comp_var[vars[start:end],c] = True

    # Cooperative components Kriging model fit
    # comp_var can be provided explicitly or auto-computed from ncomp and seed
    model = CoopCompKRG(ncomp=n_comp)
    model.set_training_values(x_train, y_train)
    model.train()

    # Prediction as for ordinary Kriging
    y_pred = model.predict_values(x_test)
    """

    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        declare(
            "corr",
            "squar_exp",
            values=(
                "pow_exp",
                "abs_exp",
                "squar_exp",
                "squar_sin_exp",
                "matern52",
                "matern32",
            ),
            desc="Correlation function type",
            types=(str),
        )
        declare(
            "hyper_opt",
            "Cobyla",
            values=("Cobyla",),
            desc="Optimizer for hyperparameters optimisation",
            types=(str),
        )
        declare(
            "comp_var",
            None,
            desc="Boolean array [nx, n_comp] mapping design variables to components. "
            "If None, computed automatically from ncomp and seed.",
        )
        declare(
            "ncomp",
            3,
            types=(int,),
            desc="Number of components (used to build comp_var when not provided)",
        )
        supports = self.supports
        supports["variances"] = True
        supports["derivatives"] = True
        supports["variance_derivatives"] = True
        supports["x_hierarchy"] = False

    def _componentwise_distance(self, dx, theta=None, return_derivative=False):
        d = componentwise_distance(
            dx,
            self.options["corr"],
            self.nx,
            self._pow_exp_power,
            theta=theta,
            return_derivative=return_derivative,
        )
        return d

    def _build_comp_var(self, nx, ncomp, rng):
        """Build a random design-variable-to-component allocation."""
        vars_order = list(range(nx))
        rng.shuffle(vars_order)
        comp_var = np.full((nx, ncomp), False)
        for c in range(ncomp):
            comp_size = nx // ncomp
            start = c * comp_size
            end = (c + 1) * comp_size
            if c + 1 == ncomp:
                end = max(end, nx)
            for v in vars_order[start:end]:
                comp_var[v, c] = True
        return comp_var

    def _train(self):
        """Loop over all components and train cooperatively."""
        comp_var = self.options["comp_var"]

        if comp_var is None:
            comp_var = self._build_comp_var(self.nx, self.options["ncomp"], self.rng)

        if not isinstance(comp_var, np.ndarray) or comp_var.shape[0] != self.nx:
            raise ValueError("comp_var has the wrong data type or shape.")

        self.comp_var = [var for var in comp_var.T]
        n_comp = comp_var.shape[1]

        for active_coop_comp in range(n_comp):
            # Set context vector to current best hyperparameter (used for inactive components)
            try:
                self.coop_theta = self.optimal_theta
            except AttributeError:
                self.coop_theta = self._theta0 * np.ones((self.nx))

            # Hyperparameter optimization for active components
            self.active_comp_var = self.comp_var[active_coop_comp]
            self.num_active = np.count_nonzero(self.active_comp_var)
            super()._train()

    def _create_optimizer(self):
        """Create a cooperative COBYLA optimizer for active components only."""
        return CooperativeCobylaOptimizer(
            active_comp_var=self.active_comp_var,
            coop_theta=self.coop_theta,
            uses_log_theta_space=self._uses_log_theta_space(),
        )

    def _optimize_hyperparam(self, D):
        """Delegate to parent with iteration limit based on active dimensions."""
        return super()._optimize_hyperparam(D, limit=10 * self.num_active)
