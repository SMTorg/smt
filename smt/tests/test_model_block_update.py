import unittest
from copy import deepcopy

import numpy as np

import smt.surrogate_models.krg_based.distances as dist
from smt.surrogate_models.krg_based.mixed_int_corr import (
    compute_X_cont,
    gower_componentwise_distances,
)

from smt.design_space import (
    CategoricalVariable,
    DesignSpace,
    FloatVariable,
)
from smt.surrogate_models import KRG, SGP
from smt.surrogate_models.krg_based.kernel_types import MixIntKernelType


def xsinx(x: np.ndarray) -> np.ndarray:
    return x * np.sin(x)


def _setup_mixed_reference(ref, base_model, X_all, Y_all):
    is_acting = ref.is_acting_points.get(None)
    if is_acting is not None:
        _, is_acting = ref.design_space.correct_get_acting(X_all)
        ref.is_acting_points[None] = is_acting
        ref.is_acting_train = is_acting

    ref.D = ref._compute_training_distances(X_all, Y_all, is_acting)

    X_cont, _ = compute_X_cont(X_all, ref.design_space)
    ref.X_norma = (X_cont - base_model.X_offset) / base_model.X_scale
    ref.y_norma = (Y_all - base_model.y_mean) / base_model.y_std
    ref.X_offset = base_model.X_offset
    ref.X_scale = base_model.X_scale
    ref.y_mean = base_model.y_mean
    ref.y_std = base_model.y_std

    if hasattr(base_model, "X2_offset"):
        _restore_categorical_scaling(ref, base_model, X_all, is_acting)

def _restore_categorical_scaling(ref, base_model, X_all, is_acting):
    ref.X2_offset = base_model.X2_offset
    ref.X2_scale = base_model.X2_scale
    X2, _, unfolded_cat = ref.design_space.unfold_x(X_all)
    X2_norma = (X2 - ref.X2_offset) / ref.X2_scale


    D_unfolded, _ = dist.cross_distances(X2_norma)
    D_unfolded = np.abs(D_unfolded)
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
    ref.X_norma = (X_all - ref.X_offset) / ref.X_scale
    ref.y_norma = (Y_all - ref.y_mean) / ref.y_std
    ref.F = ref._regression_types[ref.options["poly"]](ref.X_norma)
    D, ref.ij = dist.cross_distances(ref.X_norma)
    ref.D = ref._componentwise_distance(D)

def _compute_reference_optimal_par(ref):
    theta = ref.optimal_theta
    if ref._eval_noise:
        theta = np.append(theta, ref.optimal_noise)
    _, par = ref._reduced_likelihood_function(theta=theta)
    return par

def reference_model(base_model, X_all, Y_all):
    """
    Return a copy of base_model holding the GLS parameters recomputed from
    scratch on (X_all, Y_all) by the regular likelihood evaluation, with the
    hyperparameters and the scaling factors of base_model kept fixed.
    """
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
    def _assert_mixed_matches(self, sm, updated, X_new, Y_new, is_hierarchical=False):
        original_x_test = self.x_test
        if is_hierarchical:
            self.x_test = np.hstack((np.zeros((50, 1)), np.linspace(0, 25, 50).reshape(-1, 1)))
            self.assert_matches_reference(sm, updated, X_new, Y_new)
        else:
            self.x_test = np.hstack((np.linspace(0, 25, 50).reshape(-1, 1), np.zeros((50, 1))))
            self.assert_matches_reference(sm, updated, X_new, Y_new)
        self.x_test = original_x_test

    def setUp(self):
        self.xt = np.array([[3.5], [8.0], [12.0], [17.0]])
        self.yt = xsinx(self.xt)
        self.x_test = np.linspace(0, 25, 50).reshape(-1, 1)

    def _trained_krg(self, **options):
        sm = KRG(
            design_space=DesignSpace([FloatVariable(0, 25)]),
            theta0=[1e-1],
            theta_bounds=[1e-2, 1e2],
            seed=42,
            print_global=False,
            **options,
        )
        sm.set_training_values(self.xt, self.yt)
        sm.train()
        return sm

    def _trained_mixed_krg(self, **options):
        ds = DesignSpace([FloatVariable(0, 25), CategoricalVariable(["A", "B"])])
        xt = np.array([[3.5, 0], [8.0, 1], [12.0, 0], [17.0, 1]])
        yt = xsinx(xt[:, :1])
        sm = KRG(design_space=ds, print_global=False, seed=42, **options)
        sm.set_training_values(xt, yt)
        sm.train()
        return sm

    def _trained_hierarchical_krg(self, **options):
        ds = DesignSpace(
            [
                CategoricalVariable(["A", "B"]),
                FloatVariable(0, 25),
            ]
        )
        ds.declare_decreed_var(decreed_var=1, meta_var=0, meta_value="A")

        xt = np.array([[0, 3.5], [1, 8.0], [0, 12.0], [0, 17.0]])
        yt = np.zeros((4, 1))
        for i in range(4):
            if xt[i, 0] == 0:
                yt[i] = xsinx(xt[i : i + 1, 1:2])
            else:
                yt[i] = 10.0

        sm = KRG(design_space=ds, print_global=False, seed=42, **options)
        sm.set_training_values(xt, yt)
        sm.train()
        return sm

    def assert_matches_reference(self, sm, updated, X_new, Y_new, par_rtol=1e-4, par_atol=1e-7, var_rtol=1e-4, var_atol=1e-7, pred_rtol=1e-4, pred_atol=1e-7):
        ref = reference_model(
            sm,
            np.vstack((sm.training_points[None][0][0], X_new)),
            np.vstack((sm.training_points[None][0][1], Y_new)),
        )
        for key in ["C", "beta", "gamma", "sigma2"]:
            np.testing.assert_allclose(
                updated.optimal_par[key],
                ref.optimal_par[key],
                rtol=par_rtol,
                atol=par_atol,
                err_msg=key,
            )
        np.testing.assert_allclose(
            updated.predict_values(self.x_test),
            ref.predict_values(self.x_test),
            rtol=pred_rtol,
            atol=pred_atol,
        )
        np.testing.assert_allclose(
            updated.predict_variances(self.x_test),
            ref.predict_variances(self.x_test),
            rtol=var_rtol,
            atol=var_atol,
        )

    def test_single_point_update(self):
        sm = self._trained_krg()
        X_new = np.array([[23.0]])
        updated = deepcopy(sm)
        updated.update_model(X_new, xsinx(X_new))

        self.assertEqual(updated.nt, 5)
        self.assert_matches_reference(sm, updated, X_new, xsinx(X_new))
        # noiseless model interpolates the added point
        np.testing.assert_allclose(
            updated.predict_values(X_new), xsinx(X_new), atol=1e-6
        )

    def test_multiple_points_update(self):
        for poly in ["constant", "linear", "quadratic"]:
            with self.subTest(poly=poly):
                sm = self._trained_krg(poly=poly)
                X_new = np.array([[16.0], [19.0], [23.0], [25.0]])
                updated = deepcopy(sm)
                updated.update_model(X_new, xsinx(X_new))
                self.assert_matches_reference(sm, updated, X_new, xsinx(X_new))

    def test_successive_updates(self):
        sm = self._trained_krg()
        X1 = np.array([[1.0], [20.0]])
        X2 = np.array([[23.0]])
        updated = deepcopy(sm)
        updated.update_model(X1, xsinx(X1))
        updated.update_model(X2, xsinx(X2))

        X_new = np.vstack((X1, X2))
        self.assert_matches_reference(sm, updated, X_new, xsinx(X_new))

    def test_noisy_update(self):
        for options in [{"noise0": [1e-2]}, {"eval_noise": True}]:
            with self.subTest(**options):
                sm = self._trained_krg(**options)
                X_new = np.array([[16.0], [23.0]])
                updated = deepcopy(sm)
                updated.update_model(X_new, xsinx(X_new))
                self.assert_matches_reference(sm, updated, X_new, xsinx(X_new))

    def test_multi_output_update(self):
        sm = KRG(theta0=[1e-1], print_global=False)
        sm.set_training_values(self.xt, np.hstack((self.yt, np.cos(self.xt))))
        sm.train()

        X_new = np.array([16.0, 23.0])
        Y_new = np.hstack((xsinx(X_new[:, None]), np.cos(X_new[:, None])))
        updated = deepcopy(sm)
        updated.update_model(X_new, Y_new)
        self.assert_matches_reference(sm, updated, X_new[:, None], Y_new)

    def test_variance_derivatives_after_update(self):
        sm = self._trained_krg()
        X_new = np.array([[16.0], [23.0]])
        sm.update_model(X_new, xsinx(X_new))

        derivs = sm.predict_variance_derivatives(self.x_test, 0)
        self.assertEqual(derivs.shape, (self.x_test.shape[0], 1))
        self.assertTrue(np.all(np.isfinite(derivs)))

    def test_invalid_inputs(self):
        sm = self._trained_krg()
        with self.assertRaises(ValueError):
            sm.update_model(np.array([[1.0, 2.0]]), np.array([[1.0]]))
        with self.assertRaises(ValueError):
            sm.update_model(np.array([[1.0], [2.0]]), np.array([[1.0]]))
        with self.assertRaises(ValueError):
            sm.update_model(np.array([[1.0]]), np.array([[1.0, 2.0]]))

        with self.assertRaises(RuntimeError):
            KRG().update_model(np.array([[1.0]]), np.array([[1.0]]))

    def test_supports_block_update_property(self):
        sm_default = KRG(print_global=False)
        self.assertTrue(sm_default.supports_block_update)
        self.assertTrue(sm_default.is_block_updatable)
        self.assertFalse(sm_default.is_trained)

        sm_default.set_training_values(self.xt, self.yt)
        sm_default.train()
        self.assertTrue(sm_default.is_trained)
        self.assertTrue(sm_default.supports_block_update)
        self.assertTrue(sm_default.is_block_updatable)
        self.assertIsNone(sm_default._check_model_compatibility())

        sm_het = KRG(use_het_noise=True, print_global=False)
        self.assertFalse(sm_het.supports_block_update)
        self.assertFalse(sm_het.is_block_updatable)
        with self.assertRaises(NotImplementedError):
            sm_het._check_model_compatibility()

        sm_ri = KRG(is_ri=True, print_global=False)
        self.assertFalse(sm_ri.supports_block_update)
        self.assertFalse(sm_ri.is_block_updatable)
        with self.assertRaises(NotImplementedError):
            sm_ri._check_model_compatibility()

        sm_cat_exp = KRG(
            categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE, print_global=False
        )
        self.assertFalse(sm_cat_exp.supports_block_update)
        self.assertFalse(sm_cat_exp.is_block_updatable)
        with self.assertRaises(NotImplementedError):
            sm_cat_exp._check_model_compatibility()

        sm_cat_homo = KRG(
            categorical_kernel=MixIntKernelType.HOMO_HSPHERE, print_global=False
        )
        self.assertFalse(sm_cat_homo.supports_block_update)
        self.assertFalse(sm_cat_homo.is_block_updatable)
        with self.assertRaises(NotImplementedError):
            sm_cat_homo._check_model_compatibility()

        sm_sgp = SGP(print_global=False)
        self.assertFalse(sm_sgp.supports_block_update)
        self.assertFalse(sm_sgp.is_block_updatable)
        with self.assertRaises(NotImplementedError):
            sm_sgp._check_model_compatibility()

    def test_unsupported_models(self):
        X_new, Y_new = np.array([[23.0]]), np.array([[1.0]])

        xt = np.repeat(np.linspace(0, 25, 6), 2).reshape(-1, 1)
        noise = 0.1 * np.random.default_rng(42).standard_normal(xt.shape)
        sm = KRG(use_het_noise=True, eval_noise=True, print_global=False)
        sm.set_training_values(xt, xsinx(xt) + noise)
        sm.train()
        self.assertIn("C", sm.optimal_par)
        self.assertFalse(sm.supports_block_update)
        self.assertFalse(sm.is_block_updatable)
        with self.assertRaises(NotImplementedError):
            sm.update_model(X_new, Y_new)

        x = np.linspace(0, 25, 20).reshape(-1, 1)
        sm = SGP(print_global=False, seed=42)
        sm.set_training_values(x, xsinx(x))
        sm.set_inducing_inputs(Z=x[::4].copy())
        sm.train()
        self.assertFalse(sm.supports_block_update)
        self.assertFalse(sm.is_block_updatable)
        with self.assertRaises(NotImplementedError):
            sm.update_model(X_new, Y_new)


    def test_mixed_variables_update(self):
        sm = self._trained_mixed_krg()

        X_new_single = np.array([[23.0, 0]])
        Y_new_single = xsinx(X_new_single[:, :1])
        updated_single = deepcopy(sm)
        updated_single.update_model(X_new_single, Y_new_single)

        self._assert_mixed_matches(sm, updated_single, X_new_single, Y_new_single)

        X_new_multi = np.array([[16.0, 1], [23.0, 0]])
        Y_new_multi = xsinx(X_new_multi[:, :1])
        updated_multi = deepcopy(sm)
        updated_multi.update_model(X_new_multi, Y_new_multi)
        self._assert_mixed_matches(sm, updated_multi, X_new_multi, Y_new_multi)

    def test_hierarchical_variables_update(self):
        sm = self._trained_hierarchical_krg()

        X_new = np.array([[0, 23.0]])
        Y_new = xsinx(X_new[:, 1:2])

        updated = deepcopy(sm)
        updated.update_model(X_new, Y_new)

        self._assert_mixed_matches(sm, updated, X_new, Y_new, is_hierarchical=True)

if __name__ == "__main__":
    unittest.main()
