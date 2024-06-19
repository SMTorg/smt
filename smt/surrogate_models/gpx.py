import numpy as np

from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.design_space import (
    BaseDesignSpace,
    ensure_design_space,
)

try:
    import egobox as egx

    GPX_AVAILABLE = True

    REGRESSIONS = {
        "constant": egx.RegressionSpec.CONSTANT,
        "linear": egx.RegressionSpec.LINEAR,
        "quadratic": egx.RegressionSpec.QUADRATIC,
    }
    CORRELATIONS = {
        "abs_exp": egx.CorrelationSpec.ABSOLUTE_EXPONENTIAL,
        "squar_exp": egx.CorrelationSpec.SQUARED_EXPONENTIAL,
        "matern32": egx.CorrelationSpec.MATERN32,
        "matern52": egx.CorrelationSpec.MATERN52,
    }
except ImportError:
    GPX_AVAILABLE = False


class GPX(SurrogateModel):
    name = "GPX"

    def _initialize(self):
        super(GPX, self)._initialize()

        if not GPX_AVAILABLE:
            raise RuntimeError(
                'GPX not available. Please install GPX dependencies with: pip install smt["gpx"]'
            )

        declare = self.options.declare

        declare(
            "poly",
            "constant",
            values=("constant", "linear", "quadratic"),
            desc="Regression function type",
            types=(str),
        )
        declare(
            "corr",
            "squar_exp",
            values=(
                "abs_exp",
                "squar_exp",
                "matern32",
                "matern52",
            ),
            desc="Correlation function type",
        )
        declare(
            "theta0", [1e-2], types=(list, np.ndarray), desc="Initial hyperparameters"
        )
        declare(
            "theta_bounds",
            [1e-6, 2e1],
            types=(list, np.ndarray),
            desc="Bounds for hyperparameters",
        )
        declare(
            "n_start",
            10,
            types=int,
            desc="Number of optimizer runs (multistart method)",
        )
        declare(
            "kpls_dim",
            None,
            types=(type(None), int),
            desc="Number of PLS components used for dimension reduction",
        )
        declare(
            "seed",
            default=42,
            types=int,
            desc="Seed number which controls random draws \
                for internal optim (set by default to get reproductibility)",
        )

        declare(
            "design_space",
            None,
            types=(BaseDesignSpace, list, np.ndarray),
            desc="definition of the (hierarchical) design space: "
            "use `smt.utils.design_space.DesignSpace` as the main API. Also accepts list of float variable bounds",
        )

        supports = self.supports
        supports["derivatives"] = True
        supports["variances"] = True
        supports["variance_derivatives"] = True

        self._gpx = None

    @property
    def design_space(self) -> BaseDesignSpace:
        xt = self.training_points.get(None)
        if xt is not None:
            xt = xt[0][0]

        if self.options["design_space"] is None:
            self.options["design_space"] = ensure_design_space(
                xt=xt, xlimits=self.options["xlimits"]
            )

        elif not isinstance(self.options["design_space"], BaseDesignSpace):
            ds_input = self.options["design_space"]
            self.options["design_space"] = ensure_design_space(
                xt=xt, xlimits=ds_input, design_space=ds_input
            )
        return self.options["design_space"]

    def _train(self):
        xt, yt = self.training_points[None][0]

        config = {
            "regr_spec": REGRESSIONS[self.options["poly"]],
            "corr_spec": CORRELATIONS[self.options["corr"]],
            "theta_init": np.array(self.options["theta0"]),
            "theta_bounds": np.array([self.options["theta_bounds"]]),
            "n_start": self.options["n_start"],
            "seed": self.options["seed"],
        }
        kpls_dim = self.options["kpls_dim"]
        if kpls_dim:
            config["kpls_dim"] = kpls_dim

        self._gpx = egx.Gpx.builder(**config).fit(xt, yt)

    def _predict_values(self, x):
        return self._gpx.predict(x)

    def _predict_variances(self, x):
        return self._gpx.predict_var(x)

    def _predict_derivatives(self, x, kx):
        return self._gpx.predict_gradients(x)[:, kx : kx + 1]

    def _predict_variance_derivatives(self, x, kx):
        return self._gpx.predict_var_gradients(x)[:, kx : kx + 1]

    def predict_gradients(self, x):
        """Predict derivatives wrt to all x components (eg the gradient)
        at n points given as [n, nx] matrix where nx is the dimension of x.
        Returns all gradients at the given x points as a [n, nx] matrix
        """
        return self._gpx.predict_gradients(x)

    def predict_variance_gradients(self, x):
        """Predict variance derivatives wrt to all x components (eg the variance gradient)
        at n points given as [n, nx] matrix where nx is the dimension of x.
        Returns all variance gradients at the given x points as a [n, nx] matrix"""
        return self._gpx.predict_var_gradients(x)
