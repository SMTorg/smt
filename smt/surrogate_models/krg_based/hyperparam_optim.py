"""
Hyperparameter optimizer strategy classes for Kriging-based surrogate models.

This module implements the Strategy design pattern for hyperparameter optimization,
allowing different optimization backends to be used interchangeably. Users can
select a built-in optimizer via the ``hyper_opt`` option string, or provide a custom
:class:`HyperparamOptimizer` subclass for full control.

Classes
-------
HyperparamOptimizer
    Abstract base for optimizer strategies.
CobylaOptimizer
    COBYLA (gradient-free, constraint-based) optimizer.
TNCOptimizer
    Truncated Newton Conjugate-gradient optimizer.
NoOpOptimizer
    Identity optimizer (returns initial theta unchanged).
"""

from abc import ABC, abstractmethod

from scipy import optimize


class HyperparamOptimizer(ABC):
    """Abstract base class for hyperparameter optimization strategies.

    Subclass this to implement a custom optimization algorithm. At minimum,
    override :meth:`_minimize` to define single-start optimization. The default
    :meth:`optimize` method handles multi-start by calling :meth:`_minimize`
    for each starting point and keeping the best result.
    """

    @abstractmethod
    def _minimize(self, objective, x0, gradient, hessian, constraints, bounds, limit):
        """Run optimization from a single starting point.

        Parameters
        ----------
        objective : callable(theta) -> float
            Objective function to minimize.
        x0 : np.ndarray [n_params]
            Starting point.
        gradient : callable(theta) -> np.ndarray or None
            Gradient of the objective function.
        hessian : callable(theta) -> np.ndarray or None
            Hessian of the objective function.
        constraints : list[callable]
            Inequality constraint functions; feasible when ``con(theta) >= 0``.
        bounds : list[tuple]
            ``(lower, upper)`` bounds for each parameter.
        limit : int
            Maximum number of iterations / function evaluations.

        Returns
        -------
        result : dict
            Must contain ``'x'`` (parameter vector) and ``'fun'`` (objective
            value). A *scipy.optimize.OptimizeResult* is also accepted.
        """

    def optimize(
        self,
        objective,
        theta_starts,
        gradient=None,
        hessian=None,
        constraints=None,
        bounds=None,
        limit=50,
    ):
        """Multi-start optimization over all starting points.

        Parameters
        ----------
        objective : callable(theta) -> float
            Objective function to minimize.
        theta_starts : np.ndarray [n_starts, n_params]
            Starting points for multi-start optimization.
        gradient : callable(theta) -> np.ndarray, optional
            Gradient of the objective function.
        hessian : callable(theta) -> np.ndarray, optional
            Hessian of the objective function.
        constraints : list[callable], optional
            Inequality constraint functions; feasible when ``con(theta) >= 0``.
        bounds : list[tuple], optional
            ``(lower, upper)`` bounds for each parameter.
        limit : int
            Maximum iterations / function evaluations per start.

        Returns
        -------
        result : dict or None
            Best result across all starts containing ``'x'`` and ``'fun'`` keys,
            or ``None`` if no valid result was found.
        """
        best = {"fun": float("inf")}
        for x0 in theta_starts:
            result = self._minimize(
                objective,
                x0,
                gradient,
                hessian,
                constraints or [],
                bounds or [],
                limit,
            )
            if result is not None and result.get("fun", float("inf")) < best.get(
                "fun", float("inf")
            ):
                best = result
        return best if "x" in best else None


class CobylaOptimizer(HyperparamOptimizer):
    """COBYLA (Constrained Optimization BY Linear Approximations) optimizer.

    Gradient-free, constraint-based optimization via :func:`scipy.optimize.minimize`.

    Parameters
    ----------
    rhobeg : float
        Initial step size for COBYLA.
    tol : float
        Convergence tolerance.
    """

    def __init__(self, rhobeg=0.5, tol=1e-4):
        self.rhobeg = rhobeg
        self.tol = tol

    def _minimize(self, objective, x0, gradient, hessian, constraints, bounds, limit):
        return optimize.minimize(
            objective,
            x0,
            constraints=[{"fun": con, "type": "ineq"} for con in constraints],
            method="COBYLA",
            options={
                "rhobeg": self.rhobeg,
                "tol": self.tol,
                "maxiter": limit,
            },
        )


class TNCOptimizer(HyperparamOptimizer):
    """Truncated Newton Conjugate-gradient (TNC) optimizer.

    Gradient-based, bounds-constrained optimization via
    :func:`scipy.optimize.minimize`.
    """

    def _minimize(self, objective, x0, gradient, hessian, constraints, bounds, limit):
        return optimize.minimize(
            objective,
            x0,
            method="TNC",
            jac=gradient,
            bounds=bounds,
            options={"maxfun": limit},
        )


class NoOpOptimizer(HyperparamOptimizer):
    """No-op optimizer: returns the initial theta without any optimization.

    Useful for testing or when hyperparameters are set externally.
    """

    def _minimize(self, objective, x0, gradient, hessian, constraints, bounds, limit):
        # Not used — optimize() is overridden.
        pass  # pragma: no cover

    def optimize(
        self,
        objective,
        theta_starts,
        gradient=None,
        hessian=None,
        constraints=None,
        bounds=None,
        limit=50,
    ):
        """Return initial theta without optimization."""
        return {"x": theta_starts[0]}
