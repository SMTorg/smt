import unittest
from copy import deepcopy
from typing import Optional

import numpy as np

import smt.surrogate_models.krg_based.distances as dist
from smt.design_space import (
    BaseDesignSpace,
    CategoricalVariable,
    DesignSpace,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
)
from smt.surrogate_models import KRG, SGP
from smt.surrogate_models.krg_based.kernel_types import (
    MixHrcKernelType,
    MixIntKernelType,
)
from smt.surrogate_models.krg_based.mixed_int_corr import (
    compute_X_cont,
    gower_componentwise_distances,
)


def xsinx(x: np.ndarray) -> np.ndarray:
    """Evaluate x * sin(x) test function."""
    return x * np.sin(x)


def _setup_mixed_reference(ref, base_model, X_all, Y_all):
    """Rebuild distance, scaling, and normalized data for mixed reference model."""
    if isinstance(ref.design_space, BaseDesignSpace):
        X_all, is_acting = ref.design_space.correct_get_acting(X_all)
        ref.training_points[None][0][0] = X_all
        ref.X_train = X_all
        if ref.is_acting_points.get(None) is not None:
            ref.is_acting_points[None] = is_acting
            ref.is_acting_train = is_acting
    else:
        is_acting = ref.is_acting_points.get(None)

    ref.D = ref._compute_training_distances(X_all, Y_all, is_acting)
    X_cont, _ = compute_X_cont(X_all, ref.design_space)
    ref.X_norma = (X_cont - base_model.X_offset) / base_model.X_scale
    ref.y_norma = (Y_all - base_model.y_mean) / base_model.y_std
    ref.X_offset, ref.X_scale = base_model.X_offset, base_model.X_scale
    ref.y_mean, ref.y_std = base_model.y_mean, base_model.y_std

    if base_model.options["categorical_kernel"] == MixIntKernelType.CONT_RELAX:
        _restore_categorical_scaling(ref, base_model, X_all, is_acting)


def _restore_categorical_scaling(ref, base_model, X_all, is_acting):
    """Restore continuous relaxation scaling and unfolded distance matrix."""
    ref.X2_offset, ref.X2_scale = base_model.X2_offset, base_model.X2_scale
    X2, _, unfolded_cat = ref.design_space.unfold_x(X_all)
    X2_norma = (X2 - ref.X2_offset) / ref.X2_scale

    D_unfolded, ref.ij = dist.cross_distances(X2_norma)
    D_unfolded = np.abs(D_unfolded)
    listcatdecreed = ref.design_space.is_conditionally_acting[ref.cat_features]
    if np.any(listcatdecreed):
        D_unfolded = ref._correct_distances_cat_decreed(
            D_unfolded,
            is_acting,
            listcatdecreed,
            ref.ij,
            mixint_type=MixIntKernelType.GOWER,
        )
    if np.any(ref.design_space.is_conditionally_acting):
        _, _, _, D_num = gower_componentwise_distances(
            X_all,
            x_is_acting=is_acting,
            design_space=ref.design_space,
            hierarchical_kernel=ref.options["hierarchical_kernel"],
        )
        D_unfolded[:, np.logical_not(unfolded_cat)] = (
            D_num / ref.X2_scale[np.logical_not(unfolded_cat)]
        )
    ref.D = D_unfolded
    ref.X2_norma = X2_norma


def _setup_continuous_reference(ref, X_all, Y_all):
    """Rebuild distance and normalized matrices for continuous reference model."""
    ref.X_norma = (X_all - ref.X_offset) / ref.X_scale
    ref.y_norma = (Y_all - ref.y_mean) / ref.y_std
    D, ref.ij = dist.cross_distances(ref.X_norma)
    ref.D = ref._componentwise_distance(D)


def _compute_reference_optimal_par(ref):
    """Evaluate reduced likelihood to extract GLS optimal parameters."""
    theta = ref.optimal_theta
    if ref._eval_noise:
        theta = np.append(theta, ref.optimal_noise)
    _, par = ref._reduced_likelihood_function(theta=theta)
    return par


def reference_model(base_model, X_all, Y_all):
    """Return copy of base_model with GLS parameters recomputed from scratch."""
    ref = deepcopy(base_model)
    ref.training_points[None][0] = [X_all, Y_all]
    ref.X_train = X_all
    ref.nt = X_all.shape[0]

    if not ref.is_continuous:
        _setup_mixed_reference(ref, base_model, X_all, Y_all)
    else:
        _setup_continuous_reference(ref, X_all, Y_all)

    ref.F = ref._regression_types[ref.options["poly"]](ref.X_norma)
    ref.optimal_par = _compute_reference_optimal_par(ref)
    return ref


class TestModelBlockUpdate(unittest.TestCase):
    _ALL_CAT_KERNELS = (
        MixIntKernelType.CONT_RELAX,
        MixIntKernelType.GOWER,
        MixIntKernelType.DIST_ENCODING,
        MixIntKernelType.HOMO_HSPHERE,
        MixIntKernelType.EXP_HOMO_HSPHERE,
        MixIntKernelType.COMPOUND_SYMMETRY,
    )
    _ALL_HRC_KERNELS = (
        MixHrcKernelType.ALG_KERNEL,
        MixHrcKernelType.ARC_KERNEL,
    )
    _ALL_POLYS = ("constant", "linear", "quadratic")

    def setUp(self):
        self.xt = np.array([[3.5], [8.0], [12.0], [17.0]])
        self.yt = xsinx(self.xt)
        self.x_test = np.linspace(0, 25, 50).reshape(-1, 1)

    @classmethod
    def _apply_spherical_defaults(cls, opts: dict, options: dict) -> None:
        """Apply theta bounds for spherical and compound symmetry kernels."""
        cat_kernel = options.get("categorical_kernel")
        if cat_kernel in (
            MixIntKernelType.HOMO_HSPHERE,
            MixIntKernelType.EXP_HOMO_HSPHERE,
            MixIntKernelType.COMPOUND_SYMMETRY,
        ):
            opts.setdefault("theta0", [1.0, 1.0])
            opts.setdefault("theta_bounds", [1e-2, 1e2])

    def _fit_krg(self, ds, xt: np.ndarray, yt: np.ndarray, **options) -> KRG:
        """Instantiate and train KRG with given design space and options."""
        opts = dict(print_global=False, seed=42)
        self._apply_spherical_defaults(opts, options)
        opts.update(options)
        if ds is not None:
            opts["design_space"] = ds
        sm = KRG(**opts)
        sm.set_training_values(xt, yt)
        sm.train()
        return sm

    def _trained_krg(self, **options) -> KRG:
        """Train standard 1D continuous KRG."""
        opts = dict(theta0=[1e-1], theta_bounds=[1e-2, 1e2])
        opts.update(options)
        return self._fit_krg(
            DesignSpace([FloatVariable(0, 25)]), self.xt, self.yt, **opts
        )

    def _trained_mixed_krg(self, **options) -> KRG:
        """Train 2D mixed (continuous + categorical) KRG."""
        ds = options.pop("design_space", None) or DesignSpace(
            [FloatVariable(0, 25), CategoricalVariable(["A", "B"])]
        )
        xt = np.array([[3.5, 0], [8.0, 1], [12.0, 0], [17.0, 1]])
        return self._fit_krg(ds, xt, xsinx(xt[:, :1]), **options)

    def _trained_hierarchical_krg(
        self,
        hierarchical_kernel: MixHrcKernelType = MixHrcKernelType.ALG_KERNEL,
        **options,
    ) -> KRG:
        """Train hierarchical KRG with categorical meta-variable."""
        ds = DesignSpace([CategoricalVariable(["A", "B"]), FloatVariable(0, 25)])
        ds.declare_decreed_var(decreed_var=1, meta_var=0, meta_value="A")
        xt = np.array([[0, 3.5], [1, 8.0], [0, 12.0], [0, 17.0]])
        yt = np.where(xt[:, :1] == 0, xsinx(xt[:, 1:2]), 10.0)
        return self._fit_krg(
            ds, xt, yt, hierarchical_kernel=hierarchical_kernel, **options
        )

    def _trained_hierarchical_continuous_krg(
        self,
        hierarchical_kernel: MixHrcKernelType = MixHrcKernelType.ALG_KERNEL,
        **options,
    ) -> KRG:
        """Train hierarchical KRG with continuous meta-variable."""
        ds = DesignSpace([FloatVariable(0, 1), FloatVariable(0, 25)])
        ds.declare_decreed_var(decreed_var=1, meta_var=0, meta_value=0.0)
        xt = np.array([[0.0, 3.5], [1.0, 8.0], [0.0, 12.0], [0.0, 17.0]])
        yt = np.where(xt[:, :1] == 0.0, xsinx(xt[:, 1:2]), 10.0)
        return self._fit_krg(
            ds, xt, yt, hierarchical_kernel=hierarchical_kernel, **options
        )

    def _build_mixed_int_ord_krg(self) -> KRG:
        """Train KRG with continuous, integer, ordinal, and categorical variables."""
        ds = DesignSpace(
            [
                FloatVariable(0, 25),
                IntegerVariable(0, 5),
                OrdinalVariable([1.0, 2.5, 5.0]),
                CategoricalVariable(["A", "B", "C"]),
            ]
        )
        xt = np.array(
            [
                [3.5, 1, 1.0, 0],
                [8.0, 2, 2.5, 1],
                [12.0, 3, 5.0, 2],
                [17.0, 4, 1.0, 0],
                [20.0, 5, 2.5, 1],
            ]
        )
        yt = (xt[:, :1] * np.sin(xt[:, :1])) + xt[:, 1:2] + xt[:, 2:3]
        return self._fit_krg(ds, xt, yt)

    def _build_hierarchical_scenario_krg(
        self, scenario: str, hrc_kernel: MixHrcKernelType
    ) -> KRG:
        """Construct hierarchical KRG models by scenario configuration name."""
        if scenario == "int_ord":
            ds = DesignSpace(
                [
                    CategoricalVariable(["A", "B"]),
                    IntegerVariable(0, 5),
                    OrdinalVariable([1.0, 2.5, 5.0]),
                    FloatVariable(0, 25),
                ]
            )
            ds.declare_decreed_var(decreed_var=1, meta_var=0, meta_value="A")
            ds.declare_decreed_var(decreed_var=2, meta_var=0, meta_value="B")
            xt = np.array(
                [
                    [0, 1, 1.0, 3.5],
                    [0, 3, 1.0, 8.0],
                    [1, 0, 2.5, 12.0],
                    [1, 0, 5.0, 17.0],
                    [0, 5, 1.0, 20.0],
                ]
            )
            yt = np.where(
                xt[:, :1] == 0,
                xt[:, 3:4] * np.sin(xt[:, 3:4]) + xt[:, 1:2],
                xt[:, 3:4] + xt[:, 2:3],
            )
        elif scenario == "multi_decreed":
            ds = DesignSpace(
                [
                    CategoricalVariable(["A", "B"]),
                    FloatVariable(0, 10),
                    FloatVariable(0, 10),
                    FloatVariable(0, 25),
                ]
            )
            ds.declare_decreed_var(decreed_var=1, meta_var=0, meta_value="A")
            ds.declare_decreed_var(decreed_var=2, meta_var=0, meta_value="A")
            xt = np.array(
                [
                    [0, 2.0, 3.0, 3.5],
                    [0, 5.0, 1.0, 8.0],
                    [1, 0.0, 0.0, 12.0],
                    [0, 1.0, 4.0, 17.0],
                    [1, 0.0, 0.0, 20.0],
                ]
            )
            yt = np.where(
                xt[:, :1] == 0,
                xt[:, 3:4] * np.sin(xt[:, 3:4]) + xt[:, 1:2] + xt[:, 2:3],
                10.0,
            )
        elif scenario == "multi_meta":
            ds = DesignSpace(
                [
                    CategoricalVariable(["A", "B"]),
                    CategoricalVariable(["C", "D"]),
                    FloatVariable(0, 10),
                    FloatVariable(0, 10),
                    FloatVariable(0, 25),
                ]
            )
            ds.declare_decreed_var(decreed_var=2, meta_var=0, meta_value="A")
            ds.declare_decreed_var(decreed_var=3, meta_var=1, meta_value="C")
            xt = np.array(
                [
                    [0, 0, 2.0, 3.0, 3.5],
                    [0, 1, 5.0, 0.0, 8.0],
                    [1, 0, 0.0, 4.0, 12.0],
                    [1, 1, 0.0, 0.0, 17.0],
                    [0, 0, 1.0, 2.0, 20.0],
                ]
            )
            yt = (
                xt[:, 4:5] * 0.1
                + np.where(xt[:, 0:1] == 0, xt[:, 2:3], 0.0)
                + np.where(xt[:, 1:2] == 0, xt[:, 3:4], 0.0)
            )
        else:
            raise ValueError(f"Unknown scenario {scenario}")
        return self._fit_krg(ds, xt, yt, hierarchical_kernel=hrc_kernel)

    def _build_2d_continuous_krg(self, poly: str) -> KRG:
        """Train 2D continuous KRG with specified polynomial trend."""
        ds = DesignSpace([FloatVariable(0, 10), FloatVariable(0, 10)])
        xt = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 1.0],
                [7.0, 8.0],
                [9.0, 6.0],
                [2.0, 9.0],
                [8.0, 3.0],
            ]
        )
        return self._fit_krg(ds, xt, np.sin(xt[:, :1]) * np.cos(xt[:, 1:2]), poly=poly)

    def _build_mixed_poly_krg(self, poly: str) -> KRG:
        """Train mixed KRG with specified polynomial trend."""
        ds = DesignSpace([FloatVariable(0, 25), CategoricalVariable(["A", "B"])])
        xt = np.array(
            [
                [3.5, 0],
                [8.0, 1],
                [12.0, 0],
                [17.0, 1],
                [20.0, 0],
                [22.0, 1],
            ]
        )
        return self._fit_krg(ds, xt, xt[:, :1] * np.sin(xt[:, :1]), poly=poly)

    def _get_default_x_test(self, sm, is_hierarchical: bool = False) -> np.ndarray:
        """Generate default test grid matching model dimension and hierarchy."""
        nx = sm.training_points[None][0][0].shape[1]
        if nx == 2:
            is_hier = is_hierarchical or (
                isinstance(sm.design_space, BaseDesignSpace)
                and bool(np.any(sm.design_space.is_conditionally_acting))
            )
            col_cat = np.zeros((50, 1))
            col_float = np.linspace(0, 25, 50).reshape(-1, 1)
            return (
                np.hstack((col_cat, col_float))
                if is_hier
                else np.hstack((col_float, col_cat))
            )
        return self.x_test

    def _assert_optimal_parameters(
        self, updated, ref, rtol: float = 1e-4, atol: float = 1e-7
    ) -> None:
        """Assert equality of GLS optimal parameters between updated and reference."""
        for key in ("C", "beta", "gamma", "sigma2"):
            np.testing.assert_allclose(
                updated.optimal_par[key],
                ref.optimal_par[key],
                rtol=rtol,
                atol=atol,
                err_msg=key,
            )

    def _assert_predictions_and_variances(
        self,
        updated,
        ref,
        x_test: np.ndarray,
        pred_rtol: float = 1e-4,
        pred_atol: float = 1e-7,
        var_rtol: float = 1e-4,
        var_atol: float = 1e-7,
    ) -> None:
        """Assert equality of predicted values and variances on test grid."""
        np.testing.assert_allclose(
            updated.predict_values(x_test),
            ref.predict_values(x_test),
            rtol=pred_rtol,
            atol=pred_atol,
        )
        np.testing.assert_allclose(
            updated.predict_variances(x_test),
            ref.predict_variances(x_test),
            rtol=var_rtol,
            atol=var_atol,
        )

    def _assert_interpolation(
        self,
        sm,
        updated,
        X_new: np.ndarray,
        Y_new: np.ndarray,
        atol: float = 1e-6,
    ) -> None:
        """Assert exact interpolation at added points for noise-free models."""
        is_noisy = bool(sm.options["eval_noise"]) or (
            sm.options["noise0"] is not None and sm.options["noise0"][0] > 0
        )
        if not is_noisy:
            X_eval = X_new if X_new.ndim == 2 else X_new[:, None]
            np.testing.assert_allclose(updated.predict_values(X_eval), Y_new, atol=atol)

    def assert_matches_reference(
        self,
        sm,
        updated,
        X_new: np.ndarray,
        Y_new: np.ndarray,
        x_test: Optional[np.ndarray] = None,
        is_hierarchical: bool = False,
        par_rtol: float = 1e-4,
        par_atol: float = 1e-7,
        var_rtol: float = 1e-4,
        var_atol: float = 1e-7,
        pred_rtol: float = 1e-4,
        pred_atol: float = 1e-7,
        interp_atol: float = 1e-6,
    ) -> None:
        """Validate updated model against full retrained reference model."""
        ref = reference_model(
            sm,
            np.vstack((sm.training_points[None][0][0], X_new)),
            np.vstack((sm.training_points[None][0][1], Y_new)),
        )
        if x_test is None:
            x_test = self._get_default_x_test(sm, is_hierarchical=is_hierarchical)

        self._assert_optimal_parameters(updated, ref, rtol=par_rtol, atol=par_atol)
        self._assert_predictions_and_variances(
            updated,
            ref,
            x_test,
            pred_rtol=pred_rtol,
            pred_atol=pred_atol,
            var_rtol=var_rtol,
            var_atol=var_atol,
        )
        self._assert_interpolation(sm, updated, X_new, Y_new, atol=interp_atol)

    def _check_update(
        self,
        sm,
        X_new: np.ndarray,
        Y_new: np.ndarray,
        is_acting: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Update model and assert agreement with reference model."""
        updated = deepcopy(sm)
        updated.update_model(X_new, Y_new, is_acting=is_acting)
        self.assert_matches_reference(sm, updated, X_new, Y_new, **kwargs)
        return updated

    def _check_successive_updates(
        self,
        sm,
        updates: list,
        **kwargs,
    ):
        """Apply successive updates and assert agreement with cumulative reference."""
        updated = deepcopy(sm)
        X_all, Y_all = [], []
        for X_new, Y_new in updates:
            updated.update_model(X_new, Y_new)
            X_all.append(X_new)
            Y_all.append(Y_new)
        self.assert_matches_reference(
            sm, updated, np.vstack(X_all), np.vstack(Y_all), **kwargs
        )
        return updated

    def _test_hierarchical_kernel_branch(self, hrc_kernel: MixHrcKernelType) -> None:
        """Test hierarchical updates with single and multi points for a kernel."""
        sm = self._trained_hierarchical_krg(hierarchical_kernel=hrc_kernel)
        for X in (np.array([[0, 23.0]]), np.array([[0, 21.0], [0, 23.0]])):
            self._check_update(sm, X, xsinx(X[:, 1:2]), is_hierarchical=True)

    def test_single_point_update(self):
        """Test single-point block update on continuous 1D space."""
        sm = self._trained_krg()
        X_new = np.array([[23.0]])
        updated = self._check_update(sm, X_new, xsinx(X_new))
        self.assertEqual(updated.nt, 5)

    def test_multiple_points_update(self):
        """Test multi-point block update on continuous 1D space across polynomial trends."""
        X_new = np.array([[16.0], [19.0], [23.0], [25.0]])
        for poly in self._ALL_POLYS:
            with self.subTest(poly=poly):
                sm = self._trained_krg(poly=poly)
                self._check_update(sm, X_new, xsinx(X_new))

    def test_successive_updates(self):
        """Test successive block updates on continuous 1D space."""
        sm = self._trained_krg()
        updates = [
            (np.array([[1.0], [20.0]]), xsinx(np.array([[1.0], [20.0]]))),
            (np.array([[23.0]]), xsinx(np.array([[23.0]]))),
        ]
        self._check_successive_updates(sm, updates)

    def test_noisy_update(self):
        """Test block update with noise options."""
        X_new = np.array([[16.0], [23.0]])
        for options in ({"noise0": [1e-2]}, {"eval_noise": True}):
            with self.subTest(**options):
                sm = self._trained_krg(**options)
                self._check_update(sm, X_new, xsinx(X_new))

    def test_multi_output_update(self):
        """Test multi-output block update."""
        yt_multi = np.hstack((self.yt, np.cos(self.xt)))
        sm = self._fit_krg(None, self.xt, yt_multi, theta0=[1e-1])
        X_new = np.array([[16.0], [23.0]])
        Y_new = np.hstack((xsinx(X_new), np.cos(X_new)))
        self._check_update(sm, X_new, Y_new)

    def test_variance_derivatives_after_update(self):
        """Test variance derivatives computation after block update."""
        sm = self._trained_krg()
        X_new = np.array([[16.0], [23.0]])
        sm.update_model(X_new, xsinx(X_new))
        derivs = sm.predict_variance_derivatives(self.x_test, 0)
        self.assertEqual(derivs.shape, (self.x_test.shape[0], 1))
        self.assertTrue(np.all(np.isfinite(derivs)))

    def test_invalid_inputs(self):
        """Test invalid input dimensions and untrained model error handling."""
        sm = self._trained_krg()
        invalid_cases = (
            (np.array([[1.0, 2.0]]), np.array([[1.0]])),
            (np.array([[1.0], [2.0]]), np.array([[1.0]])),
            (np.array([[1.0]]), np.array([[1.0, 2.0]])),
        )
        for X, Y in invalid_cases:
            with self.assertRaises(ValueError):
                sm.update_model(X, Y)

        with self.assertRaises(ValueError):
            sm.update_model(
                np.array([[1.0]]),
                np.array([[1.0]]),
                is_acting=np.array([[True, True]]),
            )
        with self.assertRaises(RuntimeError):
            KRG().update_model(np.array([[1.0]]), np.array([[1.0]]))

    def test_supports_block_update_property(self):
        """Test supports_block_update property across supported and unsupported models."""
        supported = [
            KRG(print_global=False),
            KRG(
                categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE,
                print_global=False,
            ),
            KRG(
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                print_global=False,
            ),
            KRG(hierarchical_kernel=MixHrcKernelType.ALG_KERNEL, print_global=False),
            KRG(hierarchical_kernel=MixHrcKernelType.ARC_KERNEL, print_global=False),
        ]
        for sm in supported:
            self.assertTrue(sm.supports_block_update)
            self.assertTrue(sm.is_block_updatable)
            self.assertIsNone(sm._check_model_compatibility())

        sm_def = supported[0]
        self.assertFalse(sm_def.is_trained)
        sm_def.set_training_values(self.xt, self.yt)
        sm_def.train()
        self.assertTrue(sm_def.is_trained)

        unsupported = [
            KRG(use_het_noise=True, print_global=False),
            KRG(is_ri=True, print_global=False),
            SGP(print_global=False),
        ]
        for sm in unsupported:
            self.assertFalse(sm.supports_block_update)
            self.assertFalse(sm.is_block_updatable)
            with self.assertRaises(NotImplementedError):
                sm._check_model_compatibility()

    def test_unsupported_models(self):
        """Test that update_model raises NotImplementedError on unsupported models."""
        X_new, Y_new = np.array([[23.0]]), np.array([[1.0]])

        xt = np.repeat(np.linspace(0, 25, 6), 2).reshape(-1, 1)
        noise = 0.1 * np.random.default_rng(42).standard_normal(xt.shape)
        sm_het = KRG(use_het_noise=True, eval_noise=True, print_global=False)
        sm_het.set_training_values(xt, xsinx(xt) + noise)
        sm_het.train()

        x = np.linspace(0, 25, 20).reshape(-1, 1)
        sm_sgp = SGP(print_global=False, seed=42)
        sm_sgp.set_training_values(x, xsinx(x))
        sm_sgp.set_inducing_inputs(Z=x[::4].copy())
        sm_sgp.train()

        for sm in (sm_het, sm_sgp):
            self.assertFalse(sm.supports_block_update)
            with self.assertRaises(NotImplementedError):
                sm.update_model(X_new, Y_new)

    def test_mixed_variables_update(self):
        """Test single- and multi-point block update on mixed design space."""
        sm = self._trained_mixed_krg()
        for X in (np.array([[23.0, 0]]), np.array([[16.0, 1], [23.0, 0]])):
            self._check_update(sm, X, xsinx(X[:, :1]))

    def test_mixed_categorical_kernels(self):
        """Test single- and multi-point block update across all categorical kernels."""
        cases = (np.array([[23.0, 0]]), np.array([[16.0, 1], [23.0, 0]]))
        for cat_kernel in self._ALL_CAT_KERNELS:
            with self.subTest(cat_kernel=cat_kernel):
                sm = self._trained_mixed_krg(categorical_kernel=cat_kernel)
                for X in cases:
                    self._check_update(sm, X, xsinx(X[:, :1]))

    def test_mixed_successive_updates(self):
        """Test successive block updates across all categorical kernels."""
        updates = [
            (np.array([[16.0, 1]]), xsinx(np.array([[16.0]]))),
            (np.array([[23.0, 0]]), xsinx(np.array([[23.0]]))),
        ]
        for cat_kernel in self._ALL_CAT_KERNELS:
            with self.subTest(cat_kernel=cat_kernel):
                sm = self._trained_mixed_krg(categorical_kernel=cat_kernel)
                self._check_successive_updates(sm, updates)

    def test_mixed_integer_and_ordinal_variables(self):
        """Test block update with Float, Integer, Ordinal, and Categorical variables."""
        sm = self._build_mixed_int_ord_krg()
        x_test = np.array([[5.0, 1, 1.0, 0], [10.0, 3, 2.5, 1], [15.0, 4, 5.0, 2]])
        cases = (
            np.array([[22.0, 2, 5.0, 1]]),
            np.array([[6.0, 0, 1.0, 2], [24.0, 3, 2.5, 0]]),
        )
        for X in cases:
            Y = (X[:, :1] * np.sin(X[:, :1])) + X[:, 1:2] + X[:, 2:3]
            self._check_update(sm, X, Y, x_test=x_test)

    def test_hierarchical_variables_update(self):
        """Test hierarchical block update across hierarchical and categorical kernels."""
        cases = (np.array([[0, 23.0]]), np.array([[0, 21.0], [0, 23.0]]))
        for hrc_kernel in self._ALL_HRC_KERNELS:
            for cat_kernel in self._ALL_CAT_KERNELS:
                with self.subTest(
                    hierarchical_kernel=hrc_kernel, cat_kernel=cat_kernel
                ):
                    sm = self._trained_hierarchical_krg(
                        hierarchical_kernel=hrc_kernel,
                        categorical_kernel=cat_kernel,
                    )
                    for X in cases:
                        self._check_update(
                            sm, X, xsinx(X[:, 1:2]), is_hierarchical=True
                        )

    def test_hierarchical_algebraic_kernel(self):
        """Test hierarchical block update with algebraic kernel."""
        self._test_hierarchical_kernel_branch(MixHrcKernelType.ALG_KERNEL)

    def test_hierarchical_arc_kernel(self):
        """Test hierarchical block update with arc kernel."""
        self._test_hierarchical_kernel_branch(MixHrcKernelType.ARC_KERNEL)

    def test_hierarchical_decreed_categorical_variable(self):
        """Test hierarchical block update with a decreed categorical variable."""
        ds = DesignSpace(
            [
                CategoricalVariable(["A", "B"]),
                CategoricalVariable(["C", "D"]),
                FloatVariable(0, 25),
            ]
        )
        ds.declare_decreed_var(decreed_var=1, meta_var=0, meta_value="A")
        xt = np.array([[0, 0, 3.5], [0, 1, 8.0], [1, 0, 12.0], [0, 0, 17.0]])
        yt = np.array([[xsinx(3.5)], [xsinx(8.0)], [10.0], [xsinx(17.0)]])
        X_new = np.array([[0, 1, 23.0]])
        Y_new = np.array([[xsinx(23.0)]])
        x_test = np.array([[0, 0, 5.0], [0, 1, 10.0], [1, 0, 15.0]])

        for hrc_kernel in self._ALL_HRC_KERNELS:
            with self.subTest(hierarchical_kernel=hrc_kernel):
                sm = self._fit_krg(ds, xt, yt, hierarchical_kernel=hrc_kernel)
                self._check_update(sm, X_new, Y_new, x_test=x_test)

    def test_hierarchical_successive_updates(self):
        """Test successive block updates across hierarchical kernels."""
        updates = [
            (np.array([[0, 21.0]]), xsinx(np.array([[21.0]]))),
            (np.array([[0, 23.0]]), xsinx(np.array([[23.0]]))),
        ]
        for hrc_kernel in self._ALL_HRC_KERNELS:
            with self.subTest(hierarchical_kernel=hrc_kernel):
                sm = self._trained_hierarchical_krg(hierarchical_kernel=hrc_kernel)
                self._check_successive_updates(sm, updates, is_hierarchical=True)

    def test_hierarchical_integer_and_ordinal_variables(self):
        """Test hierarchical space with Integer and Ordinal decreed variables."""
        x_test = np.array([[0, 2, 1.0, 5.0], [1, 0, 2.5, 10.0], [0, 4, 5.0, 18.0]])
        cases = (
            (np.array([[0, 2, 1.0, 23.0]]), np.array([[23.0 * np.sin(23.0) + 2]])),
            (
                np.array([[0, 4, 1.0, 15.0], [1, 0, 1.0, 22.0]]),
                np.array([[15.0 * np.sin(15.0) + 4], [22.0 + 1.0]]),
            ),
        )
        for hrc_kernel in self._ALL_HRC_KERNELS:
            with self.subTest(hierarchical_kernel=hrc_kernel):
                sm = self._build_hierarchical_scenario_krg("int_ord", hrc_kernel)
                for X, Y in cases:
                    self._check_update(sm, X, Y, x_test=x_test, is_hierarchical=True)

    def test_hierarchical_multiple_decreed_variables(self):
        """Test hierarchical spaces with multiple decreed and multiple meta variables."""
        test_configs = (
            (
                "multi_decreed",
                np.array([[0, 4.0, 2.0, 23.0], [1, 0.0, 0.0, 5.0]]),
                np.array([[23.0 * np.sin(23.0) + 4.0 + 2.0], [10.0]]),
                np.array([[0, 2.0, 2.0, 5.0], [1, 0.0, 0.0, 10.0]]),
            ),
            (
                "multi_meta",
                np.array([[0, 1, 3.0, 0.0, 23.0], [1, 0, 0.0, 1.0, 15.0]]),
                np.array([[23.0 * 0.1 + 3.0], [15.0 * 0.1 + 1.0]]),
                np.array([[0, 0, 1.0, 1.0, 5.0], [1, 1, 0.0, 0.0, 10.0]]),
            ),
        )
        for hrc_kernel in self._ALL_HRC_KERNELS:
            for scenario, X_new, Y_new, x_test in test_configs:
                with self.subTest(hrc_kernel=hrc_kernel, scenario=scenario):
                    sm = self._build_hierarchical_scenario_krg(scenario, hrc_kernel)
                    self._check_update(
                        sm, X_new, Y_new, x_test=x_test, is_hierarchical=True
                    )
                    self._check_update(
                        sm, X_new, Y_new, x_test=x_test, is_hierarchical=True
                    )

    def test_continuous_hierarchical_variables_update(self):
        """Test continuous hierarchical block updates with single and multi points."""
        X_single = np.array([[0.0, 23.0]])
        cases = (X_single, np.array([[0.0, 21.0], [0.0, 23.0]]))
        for hrc_kernel in self._ALL_HRC_KERNELS:
            with self.subTest(hierarchical_kernel=hrc_kernel):
                sm = self._trained_hierarchical_continuous_krg(
                    hierarchical_kernel=hrc_kernel
                )
                self.assertFalse(sm.is_continuous)
                self.assertTrue(sm.supports_block_update)
                for X in cases:
                    self._check_update(sm, X, xsinx(X[:, 1:2]), is_hierarchical=True)
                self._check_update(
                    sm,
                    X_single,
                    xsinx(X_single[:, 1:2]),
                    is_acting=np.array([[True, True]]),
                    is_hierarchical=True,
                )

    def test_continuous_hierarchical_inactive_branch_update(self):
        """Test continuous hierarchical update with points in the inactive branch."""
        cases = (
            (np.array([[0.5, 5.0]]), np.array([[10.0]])),
            (
                np.array([[0.3, 5.0], [0.0, 23.0]]),
                np.array([[10.0], [23.0 * np.sin(23.0)]]),
            ),
        )
        for hrc_kernel in self._ALL_HRC_KERNELS:
            with self.subTest(hierarchical_kernel=hrc_kernel):
                sm = self._trained_hierarchical_continuous_krg(
                    hierarchical_kernel=hrc_kernel
                )
                for X, Y in cases:
                    self._check_update(sm, X, Y, is_hierarchical=True)

    def test_continuous_hierarchical_successive_updates(self):
        """Test continuous hierarchical successive block updates."""
        updates = [
            (np.array([[0.0, 21.0]]), xsinx(np.array([[21.0]]))),
            (np.array([[0.0, 23.0]]), xsinx(np.array([[23.0]]))),
        ]
        for hrc_kernel in self._ALL_HRC_KERNELS:
            with self.subTest(hierarchical_kernel=hrc_kernel):
                sm = self._trained_hierarchical_continuous_krg(
                    hierarchical_kernel=hrc_kernel
                )
                self._check_successive_updates(sm, updates, is_hierarchical=True)

    def test_polynomial_trends_update(self):
        """Test block update across polynomial trend functions in continuous and mixed spaces."""
        configs = (
            (
                "2d_continuous",
                self._build_2d_continuous_krg,
                np.array([[4.0, 5.0], [6.0, 7.0]]),
                lambda x: np.sin(x[:, :1]) * np.cos(x[:, 1:2]),
                np.array([[2.5, 3.5], [5.5, 6.5]]),
            ),
            (
                "mixed",
                self._build_mixed_poly_krg,
                np.array([[23.0, 0]]),
                lambda x: x[:, :1] * np.sin(x[:, :1]),
                None,
            ),
        )
        for space, builder, X_new, y_fn, x_test in configs:
            for poly in self._ALL_POLYS:
                with self.subTest(space=space, poly=poly):
                    sm = builder(poly)
                    self._check_update(sm, X_new, y_fn(X_new), x_test=x_test)


if __name__ == "__main__":
    unittest.main()
