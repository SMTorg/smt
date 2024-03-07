import numpy as np
from smt.surrogate_models.surrogate_model import SurrogateModel

try:
    import egobox as egx

    GPX_AVAILABLE = True
except ImportError:
    GPX_AVAILABLE = False
    print("Error: to use GPX you have to install dependencies: pip install smt['gpx']")

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


class GPX(SurrogateModel):
    name = "GPX"

    def _initialize(self):
        super(GPX, self)._initialize()
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

        supports = self.supports
        supports["variances"] = True

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

        self.gpx = egx.Gpx.builder(**config).fit(xt, yt)

    def _predict_values(self, xt):
        y = self.gpx.predict_values(xt)
        return y

    def _predict_variances(self, xt):
        s2 = self.gpx.predict_variances(xt)
        return s2
