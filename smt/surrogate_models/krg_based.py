"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
This package is distributed under New BSD license.
"""

import sys
import warnings
from copy import deepcopy
from enum import Enum

import numpy as np
from scipy import linalg

from smt.design_space import (
    BaseDesignSpace,
    ensure_design_space,
)
from smt.kernels import (
    ActExp,
    Kernel,
    Matern32,
    Matern52,
    Operator,
    PowExp,
    SquarSinExp,
)
from smt.kernels.kernels import _Constant
from smt.sampling_methods import LHS
from smt.surrogate_models.hyperparam_optim import (
    CobylaOptimizer,
    HyperparamOptimizer,
    NoOpOptimizer,
    TNCOptimizer,
)
from smt.surrogate_models.mixed_int_corr import (
    MixedIntegerCorrelation,
    compute_n_param as _compute_n_param,
    correct_distances_cat_decreed as _correct_distances_cat_decreed,
)
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils import persistence
from smt.utils.checks import check_support, ensure_2d_array
from smt.utils.kriging import (
    MixHrcKernelType,
    compute_X_cont,
    constant,
    cross_distances,
    cross_levels,
    differences,
    gower_componentwise_distances,
    linear,
    quadratic,
)
from smt.utils.misc import standardization


class MixIntKernelType(Enum):
    EXP_HOMO_HSPHERE = "EXP_HOMO_HSPHERE"
    HOMO_HSPHERE = "HOMO_HSPHERE"
    CONT_RELAX = "CONT_RELAX"
    GOWER = "GOWER"
    COMPOUND_SYMMETRY = "COMPOUND_SYMMETRY"


# Nb of tries when inner optimization fails
MAX_RETRY = 5


class KrgBased(SurrogateModel):
    _regression_types = {"constant": constant, "linear": linear, "quadratic": quadratic}

    _correlation_class = {
        "pow_exp": PowExp,
        "abs_exp": PowExp,
        "squar_exp": PowExp,
        "squar_sin_exp": SquarSinExp,
        "matern52": Matern52,
        "matern32": Matern32,
        "act_exp": ActExp,
    }
    name = "KrigingBased"

    def _initialize(self):
        super(KrgBased, self)._initialize()
        declare = self.options.declare
        supports = self.supports
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
                "pow_exp",
                "abs_exp",
                "squar_exp",
                "matern52",
                "matern32",
            ),
            types=(str, Kernel),
            desc="Correlation function type",
        )
        declare(
            "pow_exp_power",
            1.9,
            types=(float),
            desc="Power for the pow_exp kernel function (valid values in (0.0, 2.0]). \
                This option is set automatically when corr option is squar, abs, or matern.",
        )
        declare(
            "categorical_kernel",
            MixIntKernelType.CONT_RELAX,
            values=[
                MixIntKernelType.CONT_RELAX,
                MixIntKernelType.GOWER,
                MixIntKernelType.EXP_HOMO_HSPHERE,
                MixIntKernelType.HOMO_HSPHERE,
                MixIntKernelType.COMPOUND_SYMMETRY,
            ],
            desc="The kernel to use for categorical inputs. Only for non continuous Kriging",
        )
        declare(
            "hierarchical_kernel",
            MixHrcKernelType.ALG_KERNEL,
            values=[
                MixHrcKernelType.ALG_KERNEL,
                MixHrcKernelType.ARC_KERNEL,
            ],
            desc="The kernel to use for mixed hierarchical inputs. Only for non continuous Kriging",
        )
        declare(
            "nugget",
            100.0 * np.finfo(np.double).eps,
            types=(float),
            desc="a jitter for numerical stability",
        )
        declare(
            "theta0", [1e-2], types=(list, np.ndarray), desc="Initial hyperparameters"
        )
        # In practice, in 1D and for X in [0,1], theta^{-2} in [1e-2,infty), i.e.
        # theta in (0,1e1], is a good choice to avoid overfitting. By standardising
        # X in R, X_norm = (X-X_mean)/X_std, then X_norm in [-1,1] if considering
        # one std intervals. This leads to theta in (0,2e1]
        declare(
            "theta_bounds",
            [1e-6, 2e1],
            types=(list, np.ndarray),
            desc="bounds for hyperparameters",
        )
        declare(
            "hyper_opt",
            "TNC",
            values=("Cobyla", "TNC", "NoOp"),
            desc="Optimiser for hyperparameters optimisation",
        )
        declare(
            "eval_noise",
            False,
            types=bool,
            values=(True, False),
            desc="noise evaluation flag",
        )
        declare(
            "noise0",
            [0.0],
            types=(list, np.ndarray),
            desc="Initial noise hyperparameters",
        )
        declare(
            "noise_bounds",
            [100.0 * np.finfo(np.double).eps, 1e10],
            types=(list, np.ndarray),
            desc="bounds for noise hyperparameters",
        )
        declare(
            "use_het_noise",
            False,
            types=bool,
            values=(True, False),
            desc="heteroscedastic noise evaluation flag",
        )
        declare(
            "n_start",
            10,
            types=int,
            desc="number of optimizer runs (multistart method)",
        )
        declare(
            "xlimits",
            None,
            types=(list, np.ndarray),
            desc="definition of a design space of float (continuous) variables: "
            "array-like of size nx x 2 (lower, upper bounds)",
        )
        declare(
            "design_space",
            None,
            types=(BaseDesignSpace, list, np.ndarray),
            desc="definition of the (hierarchical) design space: "
            "use `smt.design_space.DesignSpace` as the main API. Also accepts list of float variable bounds",
        )
        declare(
            "is_ri", False, types=bool, desc="activate reinterpolation for noisy cases"
        )
        self.options.declare(
            "seed",
            default=41,
            types=(type(None), int, np.random.Generator),
            desc="Numpy Generator object or seed number which controls random draws \
                for internal optim (set by default to get reproductibility)",
        )
        self.options.declare(
            "random_state",
            types=(type(None), int, np.random.RandomState),
            desc="DEPRECATED (use seed instead): Numpy RandomState object or seed number which controls random draws \
                for internal optim (set by default to get reproductibility)",
        )
        self.kplsk_second_loop = None
        self.best_iteration_fail = None
        self.retry = MAX_RETRY
        self.is_acting_points = {}

        supports["derivatives"] = True
        supports["variances"] = True
        supports["variance_derivatives"] = True
        supports["x_hierarchy"] = True

    def _final_initialize(self):
        if isinstance(self.options["random_state"], np.random.RandomState):
            raise ValueError(
                "np.random.RandomState object is not handled anymore. Please use seed and np.random.Generator"
            )
        elif isinstance(self.options["random_state"], int):
            warnings.warn(
                "Using random_state is deprecated and will raise an error in a future version. "
                "Please use seed parameter",
                DeprecationWarning,
                stacklevel=2,
            )
            self.random_state = np.random.default_rng(self.options["random_state"])
        else:
            self.random_state = np.random.default_rng()

        if self.options["seed"]:
            self.random_state = np.random.default_rng(self.options["seed"])

        # initialize default power values
        if self.options["corr"] == "squar_exp":
            self.options["pow_exp_power"] = 2.0
        elif self.options["corr"] in [
            "abs_exp",
            "squar_sin_exp",
            "matern32",
            "matern52",
        ] or isinstance(self.options["corr"], Kernel):
            self.options["pow_exp_power"] = 1.0
        # initialize kernel or link model with user defined kernel
        if isinstance(self.options["corr"], Kernel):
            if isinstance(self.options["corr"], Operator):
                self.corr = self.options["corr"] * _Constant(
                    1 / (self.options["corr"].nbaddition + 1)
                )
            else:
                self.corr = self.options["corr"]
            self.options["theta0"] = self.corr.theta
        elif (
            type(self.options["corr"]) is str
            and self.options["corr"] in self._correlation_class
        ):
            self.corr = self._correlation_class[self.options["corr"]](
                self.options["theta0"]
            )
        else:
            raise ValueError("The correlation kernel has not been correctly defined.")
        # Check the pow_exp_power is >0 and <=2
        assert (
            self.options["pow_exp_power"] > 0 and self.options["pow_exp_power"] <= 2
        ), (
            "The power value for exponential power function can only be >0 and <=2, but %s was given"
            % self.options["pow_exp_power"]
        )

    # --- Polymorphic hooks (override in subclasses instead of checking self.name) ---

    @property
    def _use_pls(self) -> bool:
        """Whether this model uses PLS dimensionality reduction.
        Override in PLS-based subclasses (KPLS, KPLSK, GEKPLS).
        """
        return False

    @property
    def _should_compute_distances_in_train(self) -> bool:
        """Whether cross-distances should be computed during training.
        Override in SGP (returns False since SGP uses inducing points).
        """
        return True

    def _get_fidelity_training_data(self):
        """Return (X, y) training data for the current fidelity level.
        Override in MFK to return data for self._lvl.
        """
        return self.training_points[None][0][0], self.training_points[None][0][1]

    def _get_pq(self):
        """Return (p, q) regression/correlation size parameters.
        Override in MFK to return (self.p, self.q).
        """
        return 0, 0

    def _reduced_log_prior(self, theta, grad=False, hessian=False):
        """Return the log prior contribution for Bayesian models.
        Default is zero (no prior). Override in MGP.
        """
        if grad:
            return np.zeros((len(theta), 1))
        if hessian:
            return np.zeros(len(theta))
        return 0.0

    def _post_optim_hook(self):
        """Post-optimization hook called after _optimize_hyperparam in _new_train.
        Default extracts noise from optimal_theta when eval_noise is enabled.
        Override in MGP (calls _specific_train) and SGP (extracts sigma2).
        """
        if self.options["eval_noise"] and not self.options["use_het_noise"]:
            self.optimal_noise = self.optimal_theta[-1]
            self.optimal_theta = self.optimal_theta[:-1]

    def _uses_log_theta_space(self) -> bool:
        """Whether the optimizer works in log10(theta) space.
        Default is True. Override in MGP (works in linear theta space).
        """
        return True

    def _get_optimizer_theta_bounds(self, theta_bounds, i):
        """Return (constraints, bounds_hyp_entry, theta0_clamped) for one theta dimension.
        Override in MGP to use linear-space bounds instead of log-space.
        """
        log10t_bounds = np.log10(theta_bounds)
        constraints = [
            lambda log10t, i=i: log10t[i] - log10t_bounds[0],
            lambda log10t, i=i: log10t_bounds[1] - log10t[i],
        ]
        return constraints, log10t_bounds

    def _get_optimizer_initial_theta(self, theta_bounds):
        """Return (theta0, theta0_rand) initial theta values for the optimizer.
        Override in MGP to sample from prior distribution.
        """
        log10t_bounds = np.log10(theta_bounds)
        theta0_rand = self.random_state.random(len(self.theta0))
        theta0_rand = (
            theta0_rand * (log10t_bounds[1] - log10t_bounds[0]) + log10t_bounds[0]
        )
        theta0 = np.log10(self.theta0)
        return theta0, theta0_rand

    def _transform_optimal_theta(self, optimal_theta):
        """Transform optimizer output back to model theta space.
        Default: 10**optimal_theta (from log10 space). Override in MGP (identity).
        """
        return 10**optimal_theta

    def _get_cont_relax_dx(self, dx):
        """Compute CONT_RELAX distances for the current fidelity level.
        Default: return dx unchanged. Override in MFK for multi-fidelity
        CONT_RELAX distance computation per level.
        """
        return dx

    def _create_optimizer(self):
        """Create the hyperparameter optimizer strategy.

        Maps the ``hyper_opt`` option string to a :class:`HyperparamOptimizer`
        instance. Override this method to inject a custom optimizer.

        Returns
        -------
        HyperparamOptimizer
        """
        hyper_opt = self.options["hyper_opt"]
        if isinstance(hyper_opt, HyperparamOptimizer):
            return hyper_opt
        if hyper_opt == "Cobyla":
            return CobylaOptimizer()
        elif hyper_opt == "TNC":
            if self.options["use_het_noise"]:
                raise ValueError("For heteroscedastic noise, please use Cobyla")
            return TNCOptimizer()
        elif hyper_opt == "NoOp":
            return NoOpOptimizer()
        else:
            raise ValueError(f"Unknown optimizer: {hyper_opt}")

    @property
    def _n_outer_iterations(self):
        """Number of outer optimization passes (0 = single pass).
        Override in KPLSK for two-pass (PLS then full Kriging) optimization.
        """
        return 0

    def _handle_theta0_out_of_bounds(self, theta0_i, i, theta_bounds):
        """Handle a theta0 value that is outside the feasible bounds.

        Parameters
        ----------
        theta0_i : float
            The out-of-bounds theta0 value.
        i : int
            Index of the parameter.
        theta_bounds : array-like
            ``[lower, upper]`` bounds.

        Returns
        -------
        float
            Corrected theta0 value.
        """
        warnings.warn(
            f"theta0 is out the feasible bounds ({self.theta0}[{i}] out of \
                                [{theta_bounds[0]}, {theta_bounds[1]}]). \
                                    A random initialisation is used instead."
        )
        val = self.random_state.random()
        val = val * (theta_bounds[1] - theta_bounds[0]) + theta_bounds[0]
        return val

    @property
    def _optimize_sigma2(self):
        """Whether to optimize GP variance (sigma2) as a hyperparameter.
        Default False. Override in SGP to return True.
        """
        return False

    def _should_sample_multistart(self, ii):
        """Whether to add LHS multistart samples in this outer iteration.

        Parameters
        ----------
        ii : int
            Current outer-loop iteration index.

        Returns
        -------
        bool
        """
        return True

    def _finalize_outer_loop(
        self,
        best_optimal_rlf_value,
        best_optimal_par,
        best_optimal_theta,
        exit_function,
    ):
        """Finalize an outer optimization loop iteration.

        Called at the end of each outer-loop iteration in
        :meth:`_optimize_hyperparam`. The default implementation requests an
        immediate return (single-pass). Override in KPLSK for two-pass logic.

        Parameters
        ----------
        best_optimal_rlf_value : float
        best_optimal_par : dict
        best_optimal_theta : np.ndarray
        exit_function : bool

        Returns
        -------
        should_return : bool
            If ``True``, return current best values and stop looping.
        exit_function : bool
            Updated flag for next iteration.
        new_limit : int or None
            If not ``None``, overrides the iteration limit for the next pass.
        """
        return True, exit_function, None

    # --- End hooks ---

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

    @property
    def is_continuous(self) -> bool:
        return self.design_space.is_all_cont

    def set_training_values(
        self, xt: np.ndarray, yt: np.ndarray, name=None, is_acting=None
    ) -> None:
        """
        Set training data (values).

        Parameters
        ----------
        xt : np.ndarray[nt, nx] or np.ndarray[nt]
            The input values for the nt training points.
        yt : np.ndarray[nt, ny] or np.ndarray[nt]
            The output values for the nt training points.
        name : str or None
            An optional label for the group of training points being set.
            This is only used in special situations (e.g., multi-fidelity applications).
        is_acting : np.ndarray[nt, nx] or np.ndarray[nt]
            Matrix specifying which of the design variables is acting in a hierarchical design space
        """
        super().set_training_values(xt, yt, name=name)
        if self.ny > 1:
            warnings.warn(
                "Kriging-based surrogate is not intended to handle multiple "
                f"training output data (yt dim should be 1, got {self.ny}). "
                "The quality of the resulting surrogate might not be as good as "
                "if each training output is used separately to build a dedicated surrogate. "
                "This warning might become a hard error in future SMT versions."
            )
        if is_acting is not None:
            self.is_acting_points[name] = is_acting

    def _correct_distances_cat_decreed(
        self,
        D,
        is_acting,
        listcatdecreed,
        ij,
        is_acting_y=None,
        mixint_type=MixIntKernelType.CONT_RELAX,
    ):
        """Delegate to :func:`mixed_int_corr.correct_distances_cat_decreed`."""
        return _correct_distances_cat_decreed(
            D,
            is_acting,
            listcatdecreed,
            ij,
            design_space=self.design_space,
            cat_features=self.cat_features,
            n_levels=self.n_levels,
            X2_offset=getattr(self, "X2_offset", None),
            X2_scale=getattr(self, "X2_scale", None),
            is_acting_y=is_acting_y,
            mixint_type=mixint_type,
        )

    def _new_train(self):
        X, y, is_acting = self._prepare_training_data()
        D = self._compute_training_distances(X, y, is_acting)
        self._build_regression_matrix()

        # Optimization
        (
            self.optimal_rlf_value,
            self.optimal_par,
            self.optimal_theta,
        ) = self._optimize_hyperparam(D)
        self._post_optim_hook()

    def _prepare_training_data(self):
        """Prepare training data: load, validate, apply PLS, standardize, and handle noise."""
        # Sampling points X and y
        X = self.training_points[None][0][0]
        y = self.training_points[None][0][1]

        # Get is_acting status from design space model if needed (might correct training points)
        is_acting = self.is_acting_points.get(None)
        if is_acting is None and not self.is_continuous:
            X, is_acting = self.design_space.correct_get_acting(X)
            self.training_points[None][0][0] = X
            self.is_acting_points[None] = is_acting

        # Compute PLS-coefficients (attr of self) and modified X and y (if GEKPLS is used)
        if self._use_pls:
            if self.is_continuous:
                X, y = self._compute_pls(X.copy(), y.copy())

        self._check_param()
        self.X_train = X
        self.is_acting_train = is_acting
        self._mix_int_corr = MixedIntegerCorrelation(self)
        _, self.cat_features = compute_X_cont(self.X_train, self.design_space)
        # Center and scale X and y
        (
            self.X_norma,
            self.y_norma,
            self.X_offset,
            self.y_mean,
            self.X_scale,
            self.y_std,
        ) = standardization(X.copy(), y.copy())

        if not self.options["eval_noise"]:
            self.optimal_noise = np.array(self.options["noise0"])
        elif self.options["use_het_noise"]:
            # hetGP works with unique design variables when noise variance are not given
            (
                self.X_norma,
                index_unique,
                nt_reps,
            ) = np.unique(self.X_norma, return_inverse=True, return_counts=True, axis=0)
            self.nt = self.X_norma.shape[0]

            # computing the mean of the output per unique design variable (see Binois et al., 2018)
            y_norma_unique = []
            for i in range(self.nt):
                y_norma_unique.append(np.mean(self.y_norma[index_unique == i]))
            # pointwise sensible estimates of the noise variances (see Ankenman et al., 2010)
            self.optimal_noise = self.options["noise0"] * np.ones(self.nt)
            for i in range(self.nt):
                diff = self.y_norma[index_unique == i] - y_norma_unique[i]
                if np.sum(diff**2) != 0.0:
                    self.optimal_noise[i] = np.std(diff, ddof=1) ** 2
            self.optimal_noise = self.optimal_noise / nt_reps
            self.y_norma = y_norma_unique

        return X, y, is_acting

    def _compute_training_distances(self, X, y, is_acting):
        """Compute cross-distances for training, handling continuous and mixed-integer cases."""
        D = None  # For SGP, D is not computed at all

        if not (self.is_continuous):
            D, self.ij, X_cont, D_num = gower_componentwise_distances(
                X=X,
                x_is_acting=is_acting,
                design_space=self.design_space,
                hierarchical_kernel=self.options["hierarchical_kernel"],
            )
            self.Lij, self.n_levels = cross_levels(
                X=self.X_train, ij=self.ij, design_space=self.design_space
            )
            listcatdecreed = self.design_space.is_conditionally_acting[
                self.cat_features
            ]
            if np.any(listcatdecreed):
                D = self._correct_distances_cat_decreed(
                    D,
                    is_acting,
                    listcatdecreed,
                    self.ij,
                    mixint_type=MixIntKernelType.GOWER,
                )
            if self.options["categorical_kernel"] == MixIntKernelType.CONT_RELAX:
                X2, _, self.unfolded_cat = self.design_space.unfold_x(X)
                (
                    self.X2_norma,
                    _,
                    self.X2_offset,
                    _,
                    self.X2_scale,
                    _,
                ) = standardization(X2.copy(), y.copy())
                D, _ = cross_distances(self.X2_norma)
                D = np.abs(D)
                self.Lij, self.n_levels = cross_levels(
                    X=self.X_train, ij=self.ij, design_space=self.design_space
                )
                if (
                    "n_comp" not in self.options._dict.keys()
                    and "cat_kernel_comp" not in self.options._dict.keys()
                ):
                    listcatdecreed = self.design_space.is_conditionally_acting[
                        self.cat_features
                    ]
                    if np.any(listcatdecreed):
                        D = self._correct_distances_cat_decreed(
                            D,
                            is_acting,
                            listcatdecreed,
                            self.ij,
                            mixint_type=MixIntKernelType.GOWER,
                        )
                    if np.any(self.design_space.is_conditionally_acting):
                        D[:, np.logical_not(self.unfolded_cat)] = (
                            D_num / self.X2_scale[np.logical_not(self.unfolded_cat)]
                        )
            # Center and scale X_cont and y
            (
                self.X_norma,
                self.y_norma,
                self.X_offset,
                self.y_mean,
                self.X_scale,
                self.y_std,
            ) = standardization(X_cont.copy(), y.copy())

        if self._should_compute_distances_in_train:
            if self.is_continuous:
                # Calculate matrix of distances D between samples
                D, self.ij = cross_distances(self.X_norma)

            if np.min(np.sum(np.abs(D), axis=1)) == 0.0:
                warnings.warn(
                    "Warning: multiple x input features have the same value (at least same row twice)."
                )

        return D

    def _build_regression_matrix(self):
        """Build the regression matrix F and validate its shape."""
        self.F = self._regression_types[self.options["poly"]](self.X_norma)
        n_samples_F = self.F.shape[0]
        if self.F.ndim > 1:
            p = self.F.shape[1]
        else:
            p = 1
        self._check_F(n_samples_F, p)

    def is_training_ill_conditioned(self):
        """
        Check if the training dataset could be an issue and print both
        the dataset correlation matrix condition number and
        minimal distance between two points.
        ----
        Returns true if R is ill_conditionned
        """
        R = self.optimal_par["C"] @ self.optimal_par["C"]
        condR = np.linalg.cond(R)
        print(
            "Minimal distance between two points in any dimension is",
            "{:.2e}".format(np.min(self.D)),
        )
        print(
            "Correlation matrix R condition number is",
            "{:.2e}".format(condR),
        )
        return (
            linalg.svd(R, compute_uv=False)[-1]
            < (1.5 * 100.0 * np.finfo(np.double).eps)
            and condR > 1e9
        )

    def _train(self):
        """
        Train the model
        """
        # outputs['sol'] = self.sol

        self._new_train()

    def _initialize_theta(self, theta, n_levels, cat_features, cat_kernel):
        """Delegate to :class:`MixedIntegerCorrelation._initialize_theta`."""
        if not hasattr(self, "_mix_int_corr") or self._mix_int_corr is None:
            self._mix_int_corr = MixedIntegerCorrelation(self)
        return self._mix_int_corr._initialize_theta(
            theta, n_levels, cat_features, cat_kernel
        )

    def _matrix_data_corr(
        self,
        corr,
        design_space,
        power,
        theta,
        theta_bounds,
        dx,
        Lij,
        n_levels,
        cat_features,
        cat_kernel,
        x=None,
        kplsk_second_loop=False,
    ):
        """Delegate to :class:`MixedIntegerCorrelation.compute`.

        The signature is preserved for backward compatibility (used by
        subclasses such as MFK).
        """
        if not hasattr(self, "_mix_int_corr") or self._mix_int_corr is None:
            self._mix_int_corr = MixedIntegerCorrelation(self)
        return self._mix_int_corr.compute(
            corr=corr,
            design_space=design_space,
            power=power,
            theta=theta,
            theta_bounds=theta_bounds,
            dx=dx,
            Lij=Lij,
            n_levels=n_levels,
            cat_features=cat_features,
            cat_kernel=cat_kernel,
            x=x,
            kplsk_second_loop=kplsk_second_loop,
        )

    def _reduced_likelihood_function(self, theta):
        """
        This function determines the BLUP parameters and evaluates the reduced
        likelihood function for the given autocorrelation parameters theta.
        Maximizing this function wrt the autocorrelation parameters theta is
        equivalent to maximizing the likelihood of the assumed joint Gaussian
        distribution of the observations y evaluated onto the design of
        experiments X.

        Parameters
        ----------
        theta: list(n_comp), optional
            - An array containing the autocorrelation parameters at which the
              Gaussian Process model parameters should be determined.

        Returns
        -------
        reduced_likelihood_function_value: real
            - The value of the reduced likelihood function associated to the
              given autocorrelation parameters theta.
        par: dict()
            - A dictionary containing the requested Gaussian Process model
              parameters:
            sigma2
            sigma2_ri
            Gaussian Process variance.
            beta
            Generalized least-squares regression weights for
            Universal Kriging or for Ordinary Kriging.
            gamma
            Gaussian Process weights.
            C
            Cholesky decomposition of the correlation matrix [R].
            Ft
            Solution of the linear equation system : [R] x Ft = F
            Q, G
            QR decomposition of the matrix Ft.
        """
        # Initialize output

        reduced_likelihood_function_value = -np.inf
        par = {}
        # Set up R
        nugget = self.options["nugget"]
        # Nugget to ensure that the Cholesky decomposition can be performed
        if self.options["eval_noise"]:
            if self.options["is_ri"]:
                nugget = 100.0 * np.finfo(np.double).eps
            else:
                nugget = 0
        noise = self.noise0
        tmp_var = theta
        if self.options["use_het_noise"]:
            noise = self.optimal_noise
        if self.options["eval_noise"] and not self.options["use_het_noise"]:
            theta = tmp_var[0:-1]
            noise = tmp_var[-1]
        if not (self.is_continuous):
            dx = self.D
            if self.options["categorical_kernel"] == MixIntKernelType.CONT_RELAX:
                dx = self._get_cont_relax_dx(dx)

            try:
                r = self._matrix_data_corr(
                    corr=self.options["corr"],
                    design_space=self.design_space,
                    power=self.options["pow_exp_power"],
                    theta=theta,
                    theta_bounds=self.options["theta_bounds"],
                    dx=dx,
                    Lij=self.Lij,
                    n_levels=self.n_levels,
                    cat_features=self.cat_features,
                    cat_kernel=self.options["categorical_kernel"],
                    kplsk_second_loop=self.kplsk_second_loop,
                ).reshape(-1, 1)

                if np.isnan(r).any():
                    return reduced_likelihood_function_value, par
            except FloatingPointError:
                warnings.warn(
                    "Theta upper bound is too high.  please reduced it in the parameter theta_bounds."
                )
                return reduced_likelihood_function_value, par
        else:
            self.corr.theta = theta
            try:
                r = self.corr(self.D).reshape(-1, 1)
                if np.isnan(r).any():
                    return reduced_likelihood_function_value, par
            except FloatingPointError:
                warnings.warn(
                    "Theta upper bound is too high.  please reduced it in the parameter theta_bounds."
                )
                return reduced_likelihood_function_value, par

        if self.options["is_ri"]:
            R_noisy = np.eye(self.nt) * (1.0 + nugget + noise)
            R_noisy[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
            R_noisy[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]
            R = np.eye(self.nt) * (1.0 + nugget)
        else:
            R = np.eye(self.nt) * (1.0 + nugget + noise)

        R[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
        R[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]

        p, q = self._get_pq()

        # Cholesky decomposition of R and computation of its inverse
        C = None
        try:
            C = linalg.cholesky(R, lower=True)
        except (linalg.LinAlgError, ValueError) as e:
            print("exception : ", e)
            print(np.linalg.eig(R)[0])
            return reduced_likelihood_function_value, par

        if self.options["is_ri"]:
            # Computation of R_ri for the reinterpolation case
            C_inv = np.linalg.inv(C)
            R_inv = np.dot(C_inv.T, C_inv)
            R_ri = R_noisy @ R_inv @ R_noisy
            par["C"] = C

            par["sigma2"] = None
            par["sigma2_ri"] = None
            _, _, sigma2_ri = self._compute_sigma2(
                R_ri, reduced_likelihood_function_value, par, p, q, is_ri=True
            )
            if sigma2_ri is not None:
                par["sigma2_ri"] = sigma2_ri * self.y_std**2.0

            reduced_likelihood_function_value, par, sigma2 = self._compute_sigma2(
                R_noisy, reduced_likelihood_function_value, par, p, q, is_ri=False
            )
            if sigma2 is not None:
                par["sigma2"] = sigma2 * self.y_std**2.0

            reduced_likelihood_function_value += self._reduced_log_prior(theta)

            # A particular case when f_min_cobyla fail
            if (self.best_iteration_fail is not None) and (
                not np.isinf(reduced_likelihood_function_value)
            ):
                if reduced_likelihood_function_value > self.best_iteration_fail:
                    self.best_iteration_fail = reduced_likelihood_function_value
                    self._thetaMemory = np.array(tmp_var)

            elif (self.best_iteration_fail is None) and (
                not np.isinf(reduced_likelihood_function_value)
            ):
                self.best_iteration_fail = reduced_likelihood_function_value
                self._thetaMemory = np.array(tmp_var)
            if reduced_likelihood_function_value > 1e15:
                reduced_likelihood_function_value = 1e15
            return reduced_likelihood_function_value, par
        else:
            # Get generalized least squared solution
            Ft = linalg.solve_triangular(C, self.F, lower=True)
            Q, G = linalg.qr(Ft, mode="economic")
            sv = linalg.svd(G, compute_uv=False)
            rcondG = sv[-1] / sv[0]
            if rcondG < 1e-10:
                # Check F
                sv = linalg.svd(self.F, compute_uv=False)
                condF = sv[0] / sv[-1]
                if condF > 1e15:
                    raise Exception(
                        "F is too ill conditioned. Poor combination "
                        "of regression model and observations."
                    )

                else:
                    # Ft is too ill conditioned, get out (try different theta)
                    return reduced_likelihood_function_value, par

            Yt = linalg.solve_triangular(C, self.y_norma, lower=True)
            beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
            rho = Yt - np.dot(Ft, beta)

            # The determinant of R is equal to the squared product of the diagonal
            # elements of its Cholesky decomposition C
            detR = (np.diag(C) ** (2.0 / self.nt)).prod()

            sigma2 = (rho**2.0).sum(axis=0) / (self.nt - p - q)
            reduced_likelihood_function_value = -(self.nt - p - q) * np.log10(
                sigma2.sum()
            ) - self.nt * np.log10(detR)
            par["sigma2"] = sigma2 * self.y_std**2.0
            par["beta"] = beta
            par["gamma"] = linalg.solve_triangular(C.T, rho)
            par["C"] = C
            par["Ft"] = Ft
            par["G"] = G
            par["Q"] = Q

            reduced_likelihood_function_value += self._reduced_log_prior(theta)

            # A particular case when f_min_cobyla fail
            if (self.best_iteration_fail is not None) and (
                not np.isinf(reduced_likelihood_function_value)
            ):
                if reduced_likelihood_function_value > self.best_iteration_fail:
                    self.best_iteration_fail = reduced_likelihood_function_value
                    self._thetaMemory = np.array(tmp_var)

            elif (self.best_iteration_fail is None) and (
                not np.isinf(reduced_likelihood_function_value)
            ):
                self.best_iteration_fail = reduced_likelihood_function_value
                self._thetaMemory = np.array(tmp_var)
            if reduced_likelihood_function_value > 1e15:
                reduced_likelihood_function_value = 1e15
            return reduced_likelihood_function_value, par

    def _compute_sigma2(
        self, R, reduced_likelihood_function_value, par, p, q, is_ri=False
    ):
        """
        This function computes the Gaussian Process variance (sigma2) and updates
        the reduced likelihood function value given the correlation matrix R.

        Parameters
        ----------
        R: array-like of shape (n_samples, n_samples)
            - The correlation matrix for which the Gaussian Process variance should be computed.
        reduced_likelihood_function_value: float
            - The current value of the reduced likelihood function.
        par: dict
            - A dictionary containing the Gaussian Process model parameters.
        p: int
            - The number of regression weights for Universal Kriging or for Ordinary Kriging.
        q: int
            - The number of Gaussian Process weights.
        is_ri: bool, optional (default: False)
            - A boolean indicating if one wants to reinterpolate the variance in the case of noisy GP.

        Returns
        -------
        reduced_likelihood_function_value: float
            - The updated value of the reduced likelihood function.
        par: dict
            - The dictionary containing the updated Gaussian Process model parameters:
            - sigma2
            - sigma2_ri
            - beta
            - gamma
            - C_noisy
            - Ft
            - Q
            - G
        sigma2: float or None
            - The computed Gaussian Process variance, or None if the computation fails.
        """
        #  Cholesky decomposition
        try:
            C = linalg.cholesky(R, lower=True)
        except (linalg.LinAlgError, ValueError) as e:
            print("exception : ", e)
            print(np.linalg.eig(R)[0])
            sigma2 = par["sigma2"]
            if is_ri:
                sigma2 = par["sigma2_ri"]
            return reduced_likelihood_function_value, par, sigma2

        # Get generalized least squared solution
        Ft = linalg.solve_triangular(C, self.F, lower=True)
        Q, G = linalg.qr(Ft, mode="economic")
        sv = linalg.svd(G, compute_uv=False)
        rcondG = sv[-1] / sv[0]
        if rcondG < 1e-10:
            # Check F
            sv = linalg.svd(self.F, compute_uv=False)
            condF = sv[0] / sv[-1]
            if condF > 1e15:
                raise Exception(
                    "F is too ill conditioned. Poor combination "
                    "of regression model and observations."
                )
            else:
                # Ft is too ill conditioned, get out (try different theta)
                return reduced_likelihood_function_value, par, None

        Yt = linalg.solve_triangular(C, self.y_norma, lower=True)
        beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
        rho = Yt - np.dot(Ft, beta)

        # The determinant of R is equal to the squared product of the diagonal
        # elements of its Cholesky decomposition C
        detR = (np.diag(C) ** (2.0 / self.nt)).prod()

        sigma2 = (rho**2.0).sum(axis=0) / (self.nt - p - q)
        reduced_likelihood_function_value = -(self.nt - p - q) * np.log10(
            sigma2.sum()
        ) - self.nt * np.log10(detR)

        if not is_ri:
            par["beta"] = beta
            par["gamma"] = linalg.solve_triangular(C.T, rho)
            par["Ft"] = Ft
            par["G"] = G
            par["Q"] = Q
            par["C_noisy"] = C

        return reduced_likelihood_function_value, par, sigma2

    def _reduced_likelihood_gradient(self, theta):
        """
        Evaluates the reduced_likelihood_gradient at a set of hyperparameters.

        Parameters
        ---------
        theta : list(n_comp), optional
            - An array containing the autocorrelation parameters at which the
              Gaussian Process model parameters should be determined.

        Returns
        -------
        grad_red : np.ndarray (dim,1)
            Derivative of the reduced_likelihood
        par: dict()
            - A dictionary containing the requested Gaussian Process model
              parameters:
            sigma2
            Gaussian Process variance.
            beta
            Generalized least-squares regression weights for
            Universal Kriging or for Ordinary Kriging.
            gamma
            Gaussian Process weights.
            C
            Cholesky decomposition of the correlation matrix [R].
            Ft
            Solution of the linear equation system : [R] x Ft = F
            Q, G
            QR decomposition of the matrix Ft.
            dr
            List of all the correlation matrix derivative
            tr
            List of all the trace part in the reduce likelihood derivatives
            dmu
            List of all the mean derivatives
            arg
            List of all minus_Cinv_dRdomega_gamma
            dsigma
            List of all sigma derivatives
        """
        grad_red = 1
        par = {}
        red, par = self._reduced_likelihood_function(theta)
        try:
            C = par["C"]
            gamma = par["gamma"]
            Q = par["Q"]
            G = par["G"]
            sigma_2 = par["sigma2"] + self.options["nugget"]
        except KeyError:
            return grad_red, par

        C = par["C"]
        gamma = par["gamma"]
        Q = par["Q"]
        G = par["G"]
        sigma_2 = par["sigma2"] + self.options["nugget"]

        nb_theta = len(theta)
        grad_red = np.zeros(nb_theta)

        dr_all = []
        tr_all = []
        dmu_all = []
        arg_all = []
        dsigma_all = []
        dbeta_all = []
        for i_der in range(nb_theta):
            # Compute R derivatives
            dr = self.corr(self.D, grad_ind=i_der)
            dr_all.append(dr)

            dR = np.zeros((self.nt, self.nt))
            dR[self.ij[:, 0], self.ij[:, 1]] = dr[:, 0]
            dR[self.ij[:, 1], self.ij[:, 0]] = dr[:, 0]

            # Compute beta derivatives
            Cinv_dR_gamma = linalg.solve_triangular(C, np.dot(dR, gamma), lower=True)
            dbeta = -linalg.solve_triangular(G, np.dot(Q.T, Cinv_dR_gamma))
            arg_all.append(Cinv_dR_gamma)

            dbeta_all.append(dbeta)

            # Compute mu derivatives
            dmu = np.dot(self.F, dbeta)
            dmu_all.append(dmu)

            # Compute log(detR) derivatives
            tr_1 = linalg.solve_triangular(C, dR, lower=True)
            tr = linalg.solve_triangular(C.T, tr_1)
            tr_all.append(tr)

            # Compute Sigma2 Derivatives
            dsigma_2 = (
                (1 / self.nt)
                * (
                    -dmu.T.dot(gamma)
                    - gamma.T.dot(dmu)
                    - np.dot(gamma.T, dR.dot(gamma))
                )
                * self.y_std**2.0
            )
            dsigma_all.append(dsigma_2)

            # Compute reduced log likelihood derivatives
            grad_red[i_der] = (
                -self.nt / np.log(10) * (dsigma_2 / sigma_2 + np.trace(tr) / self.nt)
            ).item()

        par["dr"] = dr_all
        par["tr"] = tr_all
        par["dmu"] = dmu_all
        par["arg"] = arg_all
        par["dsigma"] = dsigma_all
        par["dbeta_all"] = dbeta_all

        grad_red = np.atleast_2d(grad_red).T

        grad_red += self._reduced_log_prior(theta, grad=True)
        return grad_red, par

    def _reduced_likelihood_hessian(self, theta):
        """
        Evaluates the reduced_likelihood_gradient at a set of hyperparameters.

        Parameters
        ----------
        theta : list(n_comp), optional
            - An array containing the autocorrelation parameters at which the
              Gaussian Process model parameters should be determined.

        Returns
        -------
        hess : np.ndarray
            Hessian values.
        hess_ij: np.ndarray [nb_theta * (nb_theta + 1) / 2, 2]
            - The indices i and j of the vectors in theta associated to the hessian in hess.
        par: dict()
            - A dictionary containing the requested Gaussian Process model
              parameters:
            sigma2
            Gaussian Process variance.
            beta
            Generalized least-squared regression weights for
            Universal Kriging or for Ordinary Kriging.
            gamma
            Gaussian Process weights.
            C
            Cholesky decomposition of the correlation matrix [R].
            Ft
            Solution of the linear equation system : [R] x Ft = F
            Q, G
            QR decomposition of the matrix Ft.
            dr
            List of all the correlation matrix derivative
            tr
            List of all the trace part in the reduce likelihood derivatives
            dmu
            List of all the mean derivatives
            arg
            List of all minus_Cinv_dRdomega_gamma
            dsigma
            List of all sigma derivatives
        """
        dred, par = self._reduced_likelihood_gradient(theta)

        C = par["C"]
        gamma = par["gamma"]
        Q = par["Q"]
        G = par["G"]
        sigma_2 = par["sigma2"]

        nb_theta = len(theta)

        dr_all = par["dr"]
        tr_all = par["tr"]
        dmu_all = par["dmu"]
        arg_all = par["arg"]
        dsigma = par["dsigma"]
        Rinv_dRdomega_gamma_all = []
        Rinv_dmudomega_all = []

        n_val_hess = nb_theta * (nb_theta + 1) // 2
        hess_ij = np.zeros((n_val_hess, 2), dtype=np.int32)
        hess = np.zeros((n_val_hess, 1))
        ind_1 = 0
        log_prior = self._reduced_log_prior(theta, hessian=True)

        for omega in range(nb_theta):
            ind_0 = ind_1
            ind_1 = ind_0 + nb_theta - omega
            hess_ij[ind_0:ind_1, 0] = omega
            hess_ij[ind_0:ind_1, 1] = np.arange(omega, nb_theta)

            dRdomega = np.zeros((self.nt, self.nt))
            dRdomega[self.ij[:, 0], self.ij[:, 1]] = dr_all[omega][:, 0]
            dRdomega[self.ij[:, 1], self.ij[:, 0]] = dr_all[omega][:, 0]

            dmudomega = dmu_all[omega]
            Cinv_dmudomega = linalg.solve_triangular(C, dmudomega, lower=True)
            Rinv_dmudomega = linalg.solve_triangular(C.T, Cinv_dmudomega)
            Rinv_dmudomega_all.append(Rinv_dmudomega)
            Rinv_dRdomega_gamma = linalg.solve_triangular(C.T, arg_all[omega])
            Rinv_dRdomega_gamma_all.append(Rinv_dRdomega_gamma)

            for i, eta in enumerate(hess_ij[ind_0:ind_1, 1]):
                dRdeta = np.zeros((self.nt, self.nt))
                dRdeta[self.ij[:, 0], self.ij[:, 1]] = dr_all[eta][:, 0]
                dRdeta[self.ij[:, 1], self.ij[:, 0]] = dr_all[eta][:, 0]
                dr_eta_omega = self.corr(self.D, grad_ind=omega, hess_ind=eta)
                dRdetadomega = np.zeros((self.nt, self.nt))
                dRdetadomega[self.ij[:, 0], self.ij[:, 1]] = dr_eta_omega[:, 0]
                dRdetadomega[self.ij[:, 1], self.ij[:, 0]] = dr_eta_omega[:, 0]

                # Compute beta second derivatives
                dRdeta_Rinv_dmudomega = np.dot(dRdeta, Rinv_dmudomega)

                dmudeta = dmu_all[eta]
                Cinv_dmudeta = linalg.solve_triangular(C, dmudeta, lower=True)
                Rinv_dmudeta = linalg.solve_triangular(C.T, Cinv_dmudeta)
                dRdomega_Rinv_dmudeta = np.dot(dRdomega, Rinv_dmudeta)

                dRdeta_Rinv_dRdomega_gamma = np.dot(dRdeta, Rinv_dRdomega_gamma)

                Rinv_dRdeta_gamma = linalg.solve_triangular(C.T, arg_all[eta])
                dRdomega_Rinv_dRdeta_gamma = np.dot(dRdomega, Rinv_dRdeta_gamma)

                dRdetadomega_gamma = np.dot(dRdetadomega, gamma)

                beta_sum = (
                    dRdeta_Rinv_dmudomega
                    + dRdomega_Rinv_dmudeta
                    + dRdeta_Rinv_dRdomega_gamma
                    + dRdomega_Rinv_dRdeta_gamma
                    - dRdetadomega_gamma
                )

                Qt_Cinv_beta_sum = np.dot(
                    Q.T, linalg.solve_triangular(C, beta_sum, lower=True)
                )
                dbetadetadomega = linalg.solve_triangular(G, Qt_Cinv_beta_sum)

                # Compute mu second derivatives
                dmudetadomega = np.dot(self.F, dbetadetadomega)

                # Compute sigma2 second derivatives
                sigma_arg_1 = (
                    -np.dot(dmudetadomega.T, gamma)
                    + np.dot(dmudomega.T, Rinv_dRdeta_gamma)
                    + np.dot(dmudeta.T, Rinv_dRdomega_gamma)
                )

                sigma_arg_2 = (
                    -np.dot(gamma.T, dmudetadomega)
                    + np.dot(gamma.T, dRdeta_Rinv_dmudomega)
                    + np.dot(gamma.T, dRdomega_Rinv_dmudeta)
                )

                sigma_arg_3 = np.dot(dmudeta.T, Rinv_dmudomega) + np.dot(
                    dmudomega.T, Rinv_dmudeta
                )

                sigma_arg_4_in = (
                    -dRdetadomega_gamma
                    + dRdeta_Rinv_dRdomega_gamma
                    + dRdomega_Rinv_dRdeta_gamma
                )
                sigma_arg_4 = np.dot(gamma.T, sigma_arg_4_in)

                dsigma2detadomega = (
                    (1 / self.nt)
                    * (sigma_arg_1 + sigma_arg_2 + sigma_arg_3 + sigma_arg_4)
                    * self.y_std**2.0
                )

                # Compute Hessian
                dreddetadomega_tr_1 = np.trace(np.dot(tr_all[eta], tr_all[omega]))

                dreddetadomega_tr_2 = np.trace(
                    linalg.solve_triangular(
                        C.T, linalg.solve_triangular(C, dRdetadomega, lower=True)
                    )
                )

                dreddetadomega_arg1 = (self.nt / sigma_2) * (
                    dsigma2detadomega - (1 / sigma_2) * dsigma[omega] * dsigma[eta]
                )
                dreddetadomega = (
                    -(dreddetadomega_arg1 - dreddetadomega_tr_1 + dreddetadomega_tr_2)
                    / self.nt
                )

                hess[ind_0 + i, 0] = (self.nt / np.log(10) * dreddetadomega).item()

                if eta == omega:
                    hess[ind_0 + i, 0] += log_prior[eta].item()
            par["Rinv_dR_gamma"] = Rinv_dRdomega_gamma_all
            par["Rinv_dmu"] = Rinv_dmudomega_all
        return hess, hess_ij, par

    def _predict_init(self, x, is_acting):
        if not (self.is_continuous):
            if is_acting is None:
                x, is_acting = self.design_space.correct_get_acting(x)
            n_eval, _ = x.shape
            _, ij = cross_distances(x, self.X_train)
            dx, dnum = gower_componentwise_distances(
                x,
                x_is_acting=is_acting,
                design_space=self.design_space,
                hierarchical_kernel=self.options["hierarchical_kernel"],
                y=np.copy(self.X_train),
                y_is_acting=self.is_acting_train,
            )
            listcatdecreed = self.design_space.is_conditionally_acting[
                self.cat_features
            ]
            if np.any(listcatdecreed):
                dx = self._correct_distances_cat_decreed(
                    dx,
                    is_acting,
                    listcatdecreed,
                    ij,
                    is_acting_y=self.is_acting_train,
                    mixint_type=MixIntKernelType.GOWER,
                )
            if self.options["categorical_kernel"] == MixIntKernelType.CONT_RELAX:
                Xpred, _, _ = self.design_space.unfold_x(x)
                Xpred_norma = (Xpred - self.X2_offset) / self.X2_scale
                dx = np.abs(differences(Xpred_norma, Y=self.X2_norma.copy()))

                if (
                    "n_comp" not in self.options._dict.keys()
                    and "cat_kernel_comp" not in self.options._dict.keys()
                ):
                    listcatdecreed = self.design_space.is_conditionally_acting[
                        self.cat_features
                    ]
                    if np.any(listcatdecreed):
                        dx = self._correct_distances_cat_decreed(
                            dx,
                            is_acting,
                            listcatdecreed,
                            ij,
                            is_acting_y=self.is_acting_train,
                            mixint_type=MixIntKernelType.GOWER,
                        )
                if np.any(self.design_space.is_conditionally_acting):
                    dx[:, np.logical_not(self.unfolded_cat)] = (
                        dnum / self.X2_scale[np.logical_not(self.unfolded_cat)]
                    )
            Lij, _ = cross_levels(
                X=x, ij=ij, design_space=self.design_space, y=self.X_train
            )
            self.ij = ij
        else:
            n_eval, _ = x.shape
            X_cont = (np.copy(x) - self.X_offset) / self.X_scale
            dx = differences(X_cont, Y=self.X_norma.copy())
            ij = 0
            Lij = 0
        return x, is_acting, n_eval, ij, Lij, dx

    def predict_values(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        """
        Predict the output values at a set of points.

        Parameters
        ----------
        x : np.ndarray[nt, nx] or np.ndarray[nt]
            Input values for the prediction points.
        is_acting : np.ndarray[nt, nx] or np.ndarray[nt]
            Matrix specifying for each design variable whether it is acting or not (for hierarchical design spaces)

        Returns
        -------
        y : np.ndarray[nt, ny]
            Output values at the prediction points.
        """
        x = ensure_2d_array(x, "x")
        self._check_xdim(x)

        if is_acting is not None:
            is_acting = ensure_2d_array(is_acting, "is_acting")
            if is_acting.shape != x.shape:
                raise ValueError(
                    f"is_acting should have the same dimensions as x: {is_acting.shape} != {x.shape}"
                )

        n = x.shape[0]
        x2 = np.copy(x)
        self.printer.active = (
            self.options["print_global"] and self.options["print_prediction"]
        )

        if self.name == "MixExp":
            # Mixture of experts model
            self.printer._title("Evaluation of the Mixture of experts")
        else:
            self.printer._title("Evaluation")
        self.printer("   %-12s : %i" % ("# eval points.", n))
        self.printer()

        # Evaluate the unknown points using the specified model-method
        with self.printer._timed_context("Predicting", key="prediction"):
            y = self._predict_values(x2, is_acting=is_acting)
        time_pt = self.printer._time("prediction")[-1] / n
        self.printer()
        self.printer("Prediction time/pt. (sec) : %10.7f" % time_pt)
        self.printer()
        return y.reshape((n, self.ny))

    def _predict_values(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        """
        Evaluates the model at a set of points.

        Parameters
        ----------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values
        is_acting : np.ndarray[nt, nx] or np.ndarray[nt]
            Matrix specifying for each design variable whether it is acting or not (for hierarchical design spaces)

        Returns
        -------
        y : np.ndarray
            Evaluation point output variable values
        """
        # Initialization
        if not (self.is_continuous):
            x, is_acting, n_eval, ij, Lij, dx = self._predict_init(x, is_acting)
            r = self._matrix_data_corr(
                corr=self.options["corr"],
                design_space=self.design_space,
                power=self.options["pow_exp_power"],
                theta=self.optimal_theta,
                theta_bounds=self.options["theta_bounds"],
                dx=dx,
                Lij=Lij,
                n_levels=self.n_levels,
                cat_features=self.cat_features,
                cat_kernel=self.options["categorical_kernel"],
                x=x,
                kplsk_second_loop=self.kplsk_second_loop,
            ).reshape(n_eval, self.nt)

            X_cont, _ = compute_X_cont(x, self.design_space)

        else:
            _, _, n_eval, _, _, dx = self._predict_init(x, is_acting)
            X_cont = np.copy(x)
            d = self._componentwise_distance(dx)
            # Compute the correlation function
            r = self.corr(d).reshape(n_eval, self.nt)
            y = np.zeros(n_eval)
        X_cont = (X_cont - self.X_offset) / self.X_scale
        # Compute the regression function
        f = self._regression_types[self.options["poly"]](X_cont)
        # Scaled predictor
        y_ = np.dot(f, self.optimal_par["beta"]) + np.dot(r, self.optimal_par["gamma"])
        # Predictor
        y = (self.y_mean + self.y_std * y_).ravel()
        return y

    def _predict_derivatives(self, x, kx):
        """
        Evaluates the derivatives at a set of points.

        Parameters
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values
        kx : int
            The 0-based index of the input variable with respect to which derivatives are desired.

        Returns
        -------
        y : np.ndarray
            Derivative values.
        """
        # Initialization
        n_eval, _ = x.shape

        x = (x - self.X_offset) / self.X_scale
        # Get pairwise componentwise L1-distances to the input training set

        dx = differences(x, Y=self.X_norma.copy())
        d = self._componentwise_distance(dx)
        if self.options["corr"] == "squar_sin_exp" or isinstance(
            self.options["corr"], Kernel
        ):
            dd = 0
        else:
            dd = self._componentwise_distance(
                dx, theta=self.optimal_theta, return_derivative=True
            )

        # Compute the correlation function
        derivative_dic = {"dx": dx, "dd": dd}
        r, dr = self.corr(d, derivative_params=derivative_dic)
        r = r.reshape(n_eval, self.nt)

        drx = dr[:, kx].reshape(n_eval, self.nt)

        if self.options["poly"] == "constant":
            df = np.zeros((1, self.nx))
        elif self.options["poly"] == "linear":
            df = np.zeros((self.nx + 1, self.nx))
            df[1:, :] = np.eye(self.nx)
        else:
            raise ValueError(
                "The derivative is only available for ordinary kriging or "
                + "universal kriging using a linear trend"
            )
        # Beta and gamma = R^-1(y-FBeta)
        beta = self.optimal_par["beta"]
        gamma = self.optimal_par["gamma"]
        df_dx = np.dot(df.T, beta)

        y = (df_dx[kx] + np.dot(drx, gamma)) * self.y_std / self.X_scale[kx]
        return y

    def predict_variances(
        self, x: np.ndarray, is_acting=None, is_ri=False
    ) -> np.ndarray:
        """
        Predict the variances at a set of points.

        Parameters
        ----------
        x : np.ndarray[nt, nx] or np.ndarray[nt]
            Input values for the prediction points.
        is_acting : np.ndarray[nt, nx] or np.ndarray[nt]
            Matrix specifying for each design variable whether it is acting or not (for hierarchical design spaces)

        Returns
        -------
        s2 : np.ndarray[nt, ny]
            Variances.
        """
        check_support(self, "variances")
        x = ensure_2d_array(x, "x")
        self._check_xdim(x)

        if is_acting is not None:
            is_acting = ensure_2d_array(is_acting, "is_acting")
            if is_acting.shape != x.shape:
                raise ValueError(
                    f"is_acting should have the same dimensions as x: {is_acting.shape} != {x.shape}"
                )

        n = x.shape[0]
        x2 = np.copy(x)
        s2 = self._predict_variances(x2, is_acting=is_acting, is_ri=is_ri)
        return s2.reshape((n, self.ny))

    def _predict_variances(
        self, x: np.ndarray, is_acting=None, is_ri=False
    ) -> np.ndarray:
        """
        Provide uncertainty of the model at a set of points
        Parameters
        ----------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values
        is_acting : np.ndarray[nt, nx] or np.ndarray[nt]
            Matrix specifying for each design variable whether it is acting or not (for hierarchical design spaces)
        Returns
        -------
        s2 : np.ndarray
            Evaluation point output variable s2
        """
        # Initialization
        if not (self.is_continuous):
            x, is_acting, n_eval, ij, Lij, dx = self._predict_init(x, is_acting)
            X_cont = x

            r = self._matrix_data_corr(
                corr=self.options["corr"],
                design_space=self.design_space,
                power=self.options["pow_exp_power"],
                theta=self.optimal_theta,
                theta_bounds=self.options["theta_bounds"],
                dx=dx,
                Lij=Lij,
                n_levels=self.n_levels,
                cat_features=self.cat_features,
                cat_kernel=self.options["categorical_kernel"],
                x=x,
                kplsk_second_loop=self.kplsk_second_loop,
            ).reshape(n_eval, self.nt)

            X_cont, _ = compute_X_cont(x, self.design_space)
        else:
            _, _, n_eval, _, _, dx = self._predict_init(x, is_acting)
            X_cont = np.copy(x)
            d = self._componentwise_distance(dx)
            # Compute the correlation function
            r = self.corr(d).reshape(n_eval, self.nt)
        X_cont = (X_cont - self.X_offset) / self.X_scale
        C = self.optimal_par["C"]
        rt = linalg.solve_triangular(C, r.T, lower=True)

        u = linalg.solve_triangular(
            self.optimal_par["G"].T,
            np.dot(self.optimal_par["Ft"].T, rt)
            - self._regression_types[self.options["poly"]](X_cont).T,
        )
        is_noisy = np.max(self.options["noise0"]) > 0.0 or self.options["eval_noise"]
        if is_noisy and is_ri:
            A = self.optimal_par["sigma2_ri"]
        else:
            A = self.optimal_par["sigma2"]
        B = 1.0 - (rt**2.0).sum(axis=0) + (u**2.0).sum(axis=0)
        s2 = np.einsum("i,j -> ji", A, B)
        # Mean Squared Error might be slightly negative depending on
        # machine precision: force to zero!
        s2[s2 < 0.0] = 0.0
        return s2

    def save(self, filename):
        persistence.save(self, filename)

    @staticmethod
    def load(filename):
        return persistence.load(filename)

    def _predict_variance_derivatives(self, x, kx):
        """
        Provide the derivatives of the variance of the model at a set of points

        Parameters
        -----------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values
        kx : int
            The 0-based index of the input variable with respect to which derivatives are desired.

        Returns
        -------
        derived_variance:  np.ndarray
            The derivatives wrt kx-th component of the variance of the kriging model
        """
        return self._internal_predict_variance(x, kx)

    def _predict_variance_gradient(self, x):
        """
        Provide the gradient of the variance of the model at a given point
        (ie the derivatives wrt to all component at a unique point x)

        Parameters
        -----------
        x : np.ndarray [1, dim]
            Evaluation point input variable values

        Returns
        -------
         derived_variance:  np.ndarray
            The gradient of the variance of the kriging model
        """
        return self._internal_predict_variance(x)

    def _internal_predict_variance(self, x, kx=None):
        """
        When kx is None gradient is computed at the location x
        otherwise partial derivatives wrt kx-th component of a set of points x
        """

        # Initialization
        n_eval, _ = x.shape
        x = (x - self.X_offset) / self.X_scale
        # Get pairwise componentwise L1-distances to the input training set
        dx = differences(x, Y=self.X_norma.copy())
        d = self._componentwise_distance(dx)
        if self.options["corr"] == "squar_sin_exp" or isinstance(
            self.options["corr"], Kernel
        ):
            dd = 0
        else:
            dd = self._componentwise_distance(
                dx, theta=self.optimal_theta, return_derivative=True
            )
        derivative_dic = {"dx": dx, "dd": dd}

        sigma2 = self.optimal_par["sigma2"]
        C = self.optimal_par["C"]

        # p1 : derivative of (rt**2.0).sum(axis=0)
        r, dr = self.corr(d, derivative_params=derivative_dic)
        if kx is None:
            rt = linalg.solve_triangular(C, r, lower=True)
            drx = dr.T
        else:
            r = r.reshape(n_eval, self.nt)
            rt = linalg.solve_triangular(C, r.T, lower=True)
            drx = dr[:, kx].reshape(n_eval, self.nt)

        invKr = linalg.solve_triangular(C.T, rt)
        p1 = 2 * np.dot(drx, invKr).T

        # p2 : derivative of (u**2.0).sum(axis=0)
        f_x = self._regression_types[self.options["poly"]](x).T
        F = self.F
        rho2 = linalg.solve_triangular(C, F, lower=True)
        invKF = linalg.solve_triangular(C.T, rho2)

        if kx is None:
            A = f_x.T - np.dot(r.T, invKF)
        else:
            A = f_x.T - np.dot(r, invKF)

        B = np.dot(F.T, invKF)
        rho3 = linalg.cholesky(B, lower=True)
        invBAt = linalg.solve_triangular(rho3, A.T, lower=True)
        D = linalg.solve_triangular(rho3.T, invBAt)

        if self.options["poly"] == "constant":
            df = np.zeros((1, self.nx))
        elif self.options["poly"] == "linear":
            df = np.zeros((self.nx + 1, self.nx))
            df[1:, :] = np.eye(self.nx)
        else:
            raise ValueError(
                "The derivative is only available for ordinary kriging or "
                + "universal kriging using a linear trend"
            )

        if kx is None:
            dA = df.T - np.dot(dr.T, invKF)
        else:
            dA = df[:, kx].T - np.dot(drx, invKF)

        p3 = 2 * np.dot(dA, D).T

        # prime : derivative of MSE
        # MSE ~1.0 - (rt**2.0).sum(axis=0) + (u**2.0).sum(axis=0)
        prime = 0 - p1 + p3
        ## scaling factors
        if kx is None:
            derived_variance = []
            x_std = np.resize(self.X_scale, self.nx)
            for i in range(len(x_std)):
                derived_variance.append(sigma2 * prime.T[i] / x_std[i])
            return np.array(derived_variance).T
        else:
            x_std = self.X_scale[kx]
            derived_variance = np.array((np.outer(sigma2, np.diag(prime.T)) / x_std))
            return np.atleast_2d(derived_variance).T

    def _optimize_hyperparam(self, D):
        """
        This function evaluates the Gaussian Process model at x.

        Parameters
        ----------
        D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The componentwise cross-spatial-correlation-distance between the
              vectors in X.
           For SGP surrogate, D is not used

        Returns
        -------
        best_optimal_rlf_value: real
            - The value of the reduced likelihood function associated to the
              best autocorrelation parameters theta.
        best_optimal_par: dict()
            - A dictionary containing the requested Gaussian Process model
              parameters.
        best_optimal_theta: list(n_comp) or list(dim)
            - The best hyperparameters found by the optimization.
        """
        # reinitialize optimization best values
        self.best_iteration_fail = None
        self._thetaMemory = None
        # Initialize the hyperparameter-optimization
        if self._uses_log_theta_space():

            def minus_reduced_likelihood_function(log10t):
                return -self._reduced_likelihood_function(theta=10.0**log10t)[0]

            def grad_minus_reduced_likelihood_function(log10t):
                log10t_2d = np.atleast_2d(log10t).T
                res = (
                    -np.log(10.0)
                    * (10.0**log10t_2d)
                    * (self._reduced_likelihood_gradient(10.0**log10t_2d)[0])
                )
                return res

            def hessian_minus_reduced_likelihood_function(log10t):
                log10t_2d = np.atleast_2d(log10t).T
                res = (
                    -np.log(10.0)
                    * (10.0**log10t_2d)
                    * (self._reduced_likelihood_hessian(10.0**log10t_2d)[0])
                )
                return res

        else:

            def minus_reduced_likelihood_function(theta):
                res = -self._reduced_likelihood_function(theta)[0]
                return res

            def grad_minus_reduced_likelihood_function(theta):
                grad = -self._reduced_likelihood_gradient(theta)[0]
                return grad

            def hessian_minus_reduced_likelihood_function(theta):
                hess = -self._reduced_likelihood_hessian(theta)[0]
                return hess

        limit, _rhobeg = max(12 * len(self.options["theta0"]), 50), 0.5
        exit_function = False
        if self.kplsk_second_loop is None:
            self.kplsk_second_loop = False
        elif self.kplsk_second_loop is True:
            exit_function = True
        n_iter = self._n_outer_iterations

        # Create optimizer strategy
        optimizer = self._create_optimizer()

        (
            best_optimal_theta,
            best_optimal_rlf_value,
            best_optimal_par,
            constraints,
        ) = (
            [],
            [],
            [],
            [],
        )

        for ii in range(n_iter, -1, -1):
            bounds_hyp = []
            self.kplsk_second_loop = (
                self._n_outer_iterations > 0 and ii == 0
            ) or self.kplsk_second_loop
            self.theta0 = deepcopy(self.options["theta0"])
            self.corr.theta = deepcopy(self.options["theta0"])
            for i in range(len(self.theta0)):
                # In practice, in 1D and for X in [0,1], theta^{-2} in [1e-2,infty),
                # i.e. theta in (0,1e1], is a good choice to avoid overfitting.
                # By standardising X in R, X_norm = (X-X_mean)/X_std, then
                # X_norm in [-1,1] if considering one std intervals. This leads
                # to theta in (0,2e1]
                theta_bounds = self.options["theta_bounds"]
                if self.theta0[i] < theta_bounds[0] or self.theta0[i] > theta_bounds[1]:
                    self.theta0[i] = self._handle_theta0_out_of_bounds(
                        self.theta0[i], i, theta_bounds
                    )

                new_constraints, bounds_entry = self._get_optimizer_theta_bounds(
                    theta_bounds, i
                )
                constraints.extend(new_constraints)
                bounds_hyp.append(bounds_entry)

            theta0, theta0_rand = self._get_optimizer_initial_theta(
                self.options["theta_bounds"]
            )

            if self._should_compute_distances_in_train:
                if not (self.is_continuous):
                    self.D = D
                else:
                    ##from abs distance to kernel distance
                    self.D = self._componentwise_distance(D, opt=ii)
            else:  # SGP case, D is not used
                pass

            # Initialization
            k, stop, best_optimal_rlf_value = 0, 1, -1e20
            while k < stop:
                # Use specified starting point as first guess
                self.noise0 = np.array(self.options["noise0"])
                noise_bounds = self.options["noise_bounds"]

                # GP variance is optimized too when _optimize_sigma2 is True
                offset = 0
                if self._optimize_sigma2 and self.retry == MAX_RETRY:
                    sigma2_0 = np.log10(np.array([self.y_std[0] ** 2]))
                    theta0_sigma2 = np.concatenate([theta0, sigma2_0])
                    sigma2_bounds = np.log10(
                        np.array([1e-12, (3.0 * self.y_std[0]) ** 2])
                    )
                    constraints.append(
                        lambda log10t: log10t[len(self.theta0)] - sigma2_bounds[0]
                    )
                    constraints.append(
                        lambda log10t: sigma2_bounds[1] - log10t[len(self.theta0)]
                    )
                    bounds_hyp.append(sigma2_bounds)
                    offset = 1
                    theta0 = theta0_sigma2
                    theta0_rand = np.concatenate([theta0_rand, sigma2_0])

                if (
                    self.options["eval_noise"]
                    and not self.options["use_het_noise"]
                    and self.retry == MAX_RETRY
                ):
                    self.noise0[self.noise0 == 0.0] = noise_bounds[0]
                    for i in range(len(self.noise0)):
                        if (
                            self.noise0[i] < noise_bounds[0]
                            or self.noise0[i] > noise_bounds[1]
                        ):
                            self.noise0[i] = noise_bounds[0]
                            warnings.warn(
                                "Warning: noise0 is out the feasible bounds. The lowest possible value is used instead."
                            )

                    theta0 = np.concatenate(
                        [theta0, np.log10(np.array([self.noise0]).flatten())]
                    )
                    theta0_rand = np.concatenate(
                        [
                            theta0_rand,
                            np.log10(np.array([self.noise0]).flatten()),
                        ]
                    )

                    for i in range(len(self.noise0)):
                        noise_bounds = np.log10(noise_bounds)
                        constraints.append(
                            lambda log10t, i=i: (
                                log10t[offset + i + len(self.theta0)] - noise_bounds[0]
                            )
                        )
                        constraints.append(
                            lambda log10t, i=i: (
                                noise_bounds[1] - log10t[offset + i + len(self.theta0)]
                            )
                        )
                        bounds_hyp.append(noise_bounds)
                theta_limits = np.repeat(
                    np.log10([theta_bounds]), repeats=len(theta0), axis=0
                )
                theta_all_loops = np.vstack((theta0, theta0_rand))
                if self._should_sample_multistart(ii):
                    if self.options["n_start"] > 1:
                        sampling = LHS(
                            xlimits=theta_limits,
                            criterion="maximin",
                            seed=self.random_state,
                        )
                        theta_lhs_loops = sampling(self.options["n_start"])
                        theta_all_loops = np.vstack((theta_all_loops, theta_lhs_loops))

                try:
                    # Delegate to optimizer strategy
                    optimal_theta_res = optimizer.optimize(
                        objective=minus_reduced_likelihood_function,
                        theta_starts=theta_all_loops,
                        gradient=grad_minus_reduced_likelihood_function,
                        hessian=hessian_minus_reduced_likelihood_function,
                        constraints=constraints,
                        bounds=bounds_hyp,
                        limit=limit,
                    )

                    if optimal_theta_res is None or "x" not in optimal_theta_res:
                        raise ValueError(
                            "Optimizer encountered a problem: no valid result found"
                        )
                    optimal_theta = optimal_theta_res["x"]

                    optimal_theta = self._transform_optimal_theta(optimal_theta)

                    optimal_rlf_value, optimal_par = self._reduced_likelihood_function(
                        theta=optimal_theta
                    )
                    # Compare the new optimizer to the best previous one
                    if k > 0:
                        if np.isinf(optimal_rlf_value):
                            raise RuntimeError(
                                "Cannot train the model: infinite likelihood found"
                            )
                        else:
                            if (
                                self.best_iteration_fail is not None
                                and optimal_rlf_value >= self.best_iteration_fail
                            ):
                                if optimal_rlf_value > best_optimal_rlf_value:
                                    best_optimal_rlf_value = optimal_rlf_value
                                    best_optimal_par = optimal_par
                                    best_optimal_theta = optimal_theta
                                else:
                                    if (
                                        self.best_iteration_fail
                                        > best_optimal_rlf_value
                                    ):
                                        best_optimal_theta = self._thetaMemory
                                        (
                                            best_optimal_rlf_value,
                                            best_optimal_par,
                                        ) = self._reduced_likelihood_function(
                                            theta=best_optimal_theta
                                        )
                    else:
                        if np.isinf(optimal_rlf_value):
                            stop += 1
                        else:
                            best_optimal_rlf_value = optimal_rlf_value
                            best_optimal_par = optimal_par
                            best_optimal_theta = optimal_theta
                    k += 1
                except ValueError as ve:
                    # raise ve
                    # If iteration is max when fmin_cobyla fail is not reached
                    if self.retry > 0:
                        self.retry -= 1
                        k += 1
                        stop += 1
                        # One evaluation objectif function is done at least
                        if self.best_iteration_fail is not None:
                            if self.best_iteration_fail > best_optimal_rlf_value:
                                best_optimal_theta = self._thetaMemory
                                (
                                    best_optimal_rlf_value,
                                    best_optimal_par,
                                ) = self._reduced_likelihood_function(
                                    theta=best_optimal_theta
                                )
                        else:
                            self.print_failed_optimization_msg()
                            raise ve
                    # Optimization fail
                    elif np.size(best_optimal_par) == 0:
                        self.print_failed_optimization_msg()
                        raise ve
                    # Break the while loop
                    else:
                        k = stop + 1
                        print("fmin_cobyla failed but the best value is retained")

            should_return, exit_function, new_limit = self._finalize_outer_loop(
                best_optimal_rlf_value,
                best_optimal_par,
                best_optimal_theta,
                exit_function,
            )
            if new_limit is not None:
                limit = new_limit
            if should_return:
                return best_optimal_rlf_value, best_optimal_par, best_optimal_theta
        return best_optimal_rlf_value, best_optimal_par, best_optimal_theta

    def print_failed_optimization_msg(self):
        nugget = self.options["nugget"]
        print(
            "\033[91mOptimization failed.\033[0m",
            end="",
            file=sys.stderr,
        )
        print(
            f" Try increasing the 'nugget' above its current value of {nugget}.",
            file=sys.stderr,
        )

    def _check_param(self):
        """
        This function checks some parameters of the model
        and amend theta0 if possible (see _amend_theta0_option).
        """
        d = self.options["n_comp"] if "n_comp" in self.options else self.nx

        mat_dim = (
            self.options["cat_kernel_comps"]
            if "cat_kernel_comps" in self.options
            else None
        )

        n_comp = self.options["n_comp"] if "n_comp" in self.options else None
        self.n_param = compute_n_param(
            self.design_space,
            self.options["categorical_kernel"],
            d,
            n_comp,
            mat_dim,
        )
        if type(self.options["corr"]) is str:
            if self.options["corr"] == "squar_sin_exp":
                if (
                    self.is_continuous
                    or self.options["categorical_kernel"] == MixIntKernelType.GOWER
                ):
                    self.options["theta0"] *= np.ones(2 * self.n_param)
                else:
                    self.n_param += len([self.design_space.is_cat_mask])
                    self.options["theta0"] *= np.ones(self.n_param)

            else:
                self.options["theta0"] *= np.ones(self.n_param)
        if (
            self.options["corr"] not in ["squar_exp", "abs_exp", "pow_exp"]
            and not (self.is_continuous)
            and self.options["categorical_kernel"]
            not in [
                MixIntKernelType.GOWER,
                MixIntKernelType.COMPOUND_SYMMETRY,
                MixIntKernelType.HOMO_HSPHERE,
            ]
        ):
            raise ValueError(
                "Categorical kernels should be matrix or exponential based."
            )

        if len(self.options["theta0"]) != d and (
            self.options["categorical_kernel"]
            in [MixIntKernelType.GOWER, MixIntKernelType.COMPOUND_SYMMETRY]
            or self.is_continuous
        ):
            if len(self.options["theta0"]) == 1:
                self.options["theta0"] *= np.ones(d)
            else:
                if self.options["corr"] in (
                    "pow_exp",
                    "abs_exp",
                    "squar_exp",
                    "act_exp",
                    "matern52",
                    "matern32",
                ):
                    raise ValueError(
                        "the length of theta0 (%s) should be equal to the number of dim (%s)."
                        % (len(self.options["theta0"]), d)
                    )
        if (
            self.options["eval_noise"] or np.max(self.options["noise0"]) > 1e-12
        ) and self.options["hyper_opt"] == "TNC":
            warnings.warn(
                "TNC not available yet for noise handling. Switching to Cobyla"
            )
            self.options["hyper_opt"] = "Cobyla"

        if self.options["use_het_noise"] and not self.options["eval_noise"]:
            if len(self.options["noise0"]) != self.nt:
                if len(self.options["noise0"]) == 1:
                    self.options["noise0"] *= np.ones(self.nt)
                else:
                    raise ValueError(
                        "for the heteroscedastic case, the length of noise0 (%s) \
                            should be equal to the number of observations (%s)."
                        % (len(self.options["noise0"]), self.nt)
                    )
        if not self.options["use_het_noise"]:
            if len(self.options["noise0"]) != 1:
                raise ValueError(
                    "for the homoscedastic noise case, the length of noise0 (%s) should be equal to one."
                    % (len(self.options["noise0"]))
                )

        if self.supports["training_derivatives"]:
            if 1 not in self.training_points[None]:
                raise Exception("Derivative values are needed for using the GEK model.")

    def _check_F(self, n_samples_F, p):
        """
        This function check the F-parameters of the model.
        """

        if n_samples_F != self.nt:
            raise Exception(
                "Number of rows in F and X do not match. Most "
                "likely something is going wrong with the "
                "regression model."
            )
        if p > n_samples_F:
            raise Exception(
                (
                    "Ordinary least squares problem is undetermined "
                    "n_samples=%d must be greater than the "
                    "regression model size p=%d."
                )
                % (self.nt, p)
            )


def compute_n_param(design_space, cat_kernel, d, n_comp, mat_dim):
    """Backward-compatible wrapper: delegates to :func:`mixed_int_corr.compute_n_param`."""
    return _compute_n_param(design_space, cat_kernel, d, n_comp, mat_dim)
