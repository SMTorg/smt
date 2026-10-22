import unittest
from copy import deepcopy

import numpy as np

from smt.design_space import (
    CategoricalVariable,
    DesignSpace,
    FloatVariable,
)
from smt.surrogate_models import KRG, SGP
from smt.surrogate_models.krg_based.distances import cross_distances


def xsinx(x: np.ndarray) -> np.ndarray:
    return x * np.sin(x)


def reference_model(base_model, X_all, Y_all):
    """
    Return a copy of base_model holding the GLS parameters recomputed from
    scratch on (X_all, Y_all) by the regular likelihood evaluation, with the
    hyperparameters and the scaling factors of base_model kept fixed.
    """
    ref = deepcopy(base_model)
    ref.training_points[None][0] = [X_all, Y_all]
    ref.X_train = X_all
    ref.X_norma = (X_all - ref.X_offset) / ref.X_scale
    ref.y_norma = (Y_all - ref.y_mean) / ref.y_std
    ref.nt = X_all.shape[0]
    ref.F = ref._regression_types[ref.options["poly"]](ref.X_norma)
    D, ref.ij = cross_distances(ref.X_norma)
    ref.D = ref._componentwise_distance(D)

    theta = ref.optimal_theta
    if ref._eval_noise:
        theta = np.append(theta, ref.optimal_noise)
    _, par = ref._reduced_likelihood_function(theta=theta)
    ref.optimal_par = par
    return ref


class TestModelBlockUpdate(unittest.TestCase):
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

    def assert_matches_reference(self, sm, updated, X_new, Y_new):
        ref = reference_model(
            sm,
            np.vstack((sm.training_points[None][0][0], X_new)),
            np.vstack((sm.training_points[None][0][1], Y_new)),
        )
        for key in ["C", "beta", "gamma", "sigma2"]:
            np.testing.assert_allclose(
                updated.optimal_par[key],
                ref.optimal_par[key],
                rtol=1e-6,
                atol=1e-8,
                err_msg=key,
            )
        np.testing.assert_allclose(
            updated.predict_values(self.x_test),
            ref.predict_values(self.x_test),
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            updated.predict_variances(self.x_test),
            ref.predict_variances(self.x_test),
            rtol=1e-6,
            atol=1e-8,
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

        X_new = np.array([16.0, 23.0])  # 1-D input is accepted
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

    def test_unsupported_models(self):
        X_new, Y_new = np.array([[23.0]]), np.array([[1.0]])

        xt = np.repeat(np.linspace(0, 25, 6), 2).reshape(-1, 1)
        noise = 0.1 * np.random.default_rng(42).standard_normal(xt.shape)
        sm = KRG(use_het_noise=True, eval_noise=True, print_global=False)
        sm.set_training_values(xt, xsinx(xt) + noise)
        sm.train()
        self.assertIn("C", sm.optimal_par)
        with self.assertRaises(NotImplementedError):
            sm.update_model(X_new, Y_new)

        ds = DesignSpace([FloatVariable(0, 25), CategoricalVariable(["a", "b"])])
        xt = np.array([[3.5, 0], [8.0, 1], [12.0, 0], [17.0, 1]])
        sm = KRG(design_space=ds, print_global=False)
        sm.set_training_values(xt, xsinx(xt[:, :1]))
        sm.train()
        with self.assertRaises(NotImplementedError):
            sm.update_model(np.array([[23.0, 0]]), Y_new)

        x = np.linspace(0, 25, 20).reshape(-1, 1)
        sm = SGP(print_global=False, seed=42)
        sm.set_training_values(x, xsinx(x))
        sm.set_inducing_inputs(Z=x[::4].copy())
        sm.train()
        with self.assertRaises(NotImplementedError):
            sm.update_model(X_new, Y_new)


if __name__ == "__main__":
    unittest.main()
