"""Tests for `RMTBUQ`'s closed-form posterior variance.

This package is distributed under New BSD license.

The weight-space route implemented in `rmtb_uq.py`
(`Var(x) = sigma2 * f(x)^T mtx^-1 f(x)`) is cross-checked here against a
second, independent derivation of the same posterior variance -- the
dual/kernel-space route, which eliminates the spline's own basis/coefficient
representation and instead works directly in the n x n data-kernel space
(`K_kappa = Phi @ full_hess^-1 @ Phi^T`). Both routes solve the *same*
Gaussian-process posterior variance; agreeing to numerical precision rules
out implementation bugs (a sign flip, a transpose, an off-by-one in `edf`)
that a test against only one route's own internal consistency could miss.

The dual route is deliberately kept here, as test-only machinery, rather
than as a second public method on `RMTBUQ` itself: it is `6x-8x` slower
than the weight-space route (a dense `n x n` factorization vs. one shared
Cholesky factor already built for the mean fit), it is not recommended for
end users, and `RMTB`'s own default `regularization_weight` leaves its
`full_hess` too near-singular to invert directly, so this helper refits a
separate, regularization-boosted sibling model on the same training data
purely to get a well-conditioned `full_hess` for the cross-check.
"""

import unittest
import warnings

import numpy as np

from smt.surrogate_models import RMTB, RMTBUQ
from smt.utils.silence import Silence
from smt.utils.sm_test_case import SMTestCase

RMTB_KW = dict(
    xlimits=np.array([[0.0, 1.0], [0.0, 1.0]]),
    order=4,
    approx_order=2,
    num_ctrl_pts=20,
    energy_weight=1e-2,
    regularization_weight=1e-14,
    solver="dense-lu",
    print_global=False,
)

# A basis flexible enough, relative to a regular 8x8=64-point grid and this
# tiny energy_weight, to push edf to within a hair of n -- the
# near-exact-interpolation regime the Wahba dof warning exists to catch.
NEAR_INTERPOLATION_KW = dict(RMTB_KW, energy_weight=1e-6)


def _make_dataset(spacing, n_side=8, seed=1):
    rng = np.random.default_rng(seed)
    if spacing == "regular":
        g = np.linspace(0, 1, n_side)
        g1, g2 = np.meshgrid(g, g, indexing="ij")
        x = np.column_stack([g1.ravel(), g2.ravel()])
    else:
        x = rng.uniform(0, 1, size=(n_side * n_side, 2))
    y = np.sin(3 * x[:, 0]) * np.cos(3 * x[:, 1]) + 0.05 * rng.standard_normal(len(x))
    return x, y


def _dual_route_variances(model, x, regularization_boost=1e-6):
    """Independent kernel-space re-derivation of `model.predict_variances`,
    used only to cross-check it in these tests -- see module docstring."""
    x_train, y_train = model.training_points[None][0]
    n = len(y_train)

    # Mirror the trained model's own options (not the module-level
    # RMTB_KW default), so this helper stays correct when a test trains
    # RMTBUQ with a different order/num_ctrl_pts/etc.
    boosted_kw = {
        name: model.options[name]
        for name in (
            "xlimits",
            "order",
            "approx_order",
            "num_ctrl_pts",
            "energy_weight",
            "solver",
            "print_global",
        )
    }
    boosted_kw["regularization_weight"] = regularization_boost
    boosted = RMTB(**boosted_kw)
    boosted.set_training_values(x_train, y_train)
    with Silence():
        boosted.train()

    h = boosted.full_hess.toarray()
    cond_h = np.linalg.cond(h)
    if cond_h >= 1e8:
        raise RuntimeError(
            f"dual route unsafe at regularization_boost={regularization_boost:.0e}: "
            f"cond(full_hess)={cond_h:.3e} exceeds the 1e8 safety line"
        )

    phi, _, _ = boosted.full_jac_dict[0]
    phi = phi.toarray()
    h_inv = np.linalg.inv(h)
    phi_h_inv = phi @ h_inv
    k_kappa = phi_h_inv @ phi.T
    chol_dual = np.linalg.cholesky(k_kappa + np.eye(n))

    f = model._compute_prediction_mtx(x, 0)
    f = f.toarray() if hasattr(f, "toarray") else np.asarray(f)
    k_self = h_inv @ f.T
    kappa_xx = np.sum(f * k_self.T, axis=1)
    k_star_train = phi_h_inv @ f.T
    tmp = np.linalg.solve(chol_dual, k_star_train)
    tmp = np.linalg.solve(chol_dual.T, tmp)
    var = model.sigma2[0] * (kappa_xx - np.sum(k_star_train * tmp, axis=0))
    return np.clip(var, 0.0, None)


class TestRMTBUQ(SMTestCase):
    def _fit(self, spacing="random", **kwargs):
        x, y = _make_dataset(spacing)
        kw = dict(RMTB_KW)
        kw.update(kwargs)
        sm = RMTBUQ(**kw)
        sm.set_training_values(x, y)
        with Silence():
            sm.train()
        return sm, x, y

    def test_mean_prediction_matches_plain_rmtb(self):
        """RMTBUQ is a strictly additive subclass -- its mean must be
        bit-for-bit what plain RMTB would have predicted on the same data."""
        x, y = _make_dataset("random")

        sm_uq, _, _ = self._fit("random")
        sm_plain = RMTB(**RMTB_KW)
        sm_plain.set_training_values(x, y)
        with Silence():
            sm_plain.train()

        xq = np.random.default_rng(2).uniform(0, 1, size=(30, 2))
        self.assert_error(
            sm_uq.predict_values(xq), sm_plain.predict_values(xq), atol=1e-12
        )

    def test_predict_variances_shape_and_nonnegative(self):
        sm, _, _ = self._fit("regular")
        xq = np.random.default_rng(3).uniform(0, 1, size=(25, 2))
        s2 = sm.predict_variances(xq)
        self.assertEqual(s2.shape, (25, 1))
        self.assertTrue(np.all(s2 >= 0.0))

    def test_predict_variances_grow_away_from_training_data(self):
        """Posterior variance should be much higher in a data-sparse region
        (near the domain's corners) than inside the data-dense region the
        training points were actually drawn from (near the domain center)."""
        rng = np.random.default_rng(1)
        x = rng.uniform(0.35, 0.65, size=(64, 2))
        y = np.sin(3 * x[:, 0]) * np.cos(3 * x[:, 1]) + 0.05 * rng.standard_normal(
            len(x)
        )
        sm = RMTBUQ(**RMTB_KW)
        sm.set_training_values(x, y)
        with Silence():
            sm.train()

        near = np.array([[0.5, 0.5]])
        far = np.array([[0.02, 0.02], [0.98, 0.02], [0.02, 0.98], [0.98, 0.98]])
        s2_near = sm.predict_variances(near)
        s2_far = sm.predict_variances(far)
        self.assertGreater(s2_far.min(), 10 * s2_near.max())

    def test_sigma2_accepts_known_float(self):
        sm, _, _ = self._fit("random", sigma2=0.25)
        self.assert_error(sm.sigma2, np.array([0.25]))

    def test_weight_space_matches_dual_route(self):
        """Cross-check the production weight-space route against the
        independent dual/kernel-space derivation (see module docstring)."""
        for spacing in ("regular", "random"):
            with self.subTest(spacing=spacing):
                sm, _, _ = self._fit(spacing)
                gq = np.linspace(0.05, 0.95, 12)
                gq1, gq2 = np.meshgrid(gq, gq, indexing="ij")
                xq = np.column_stack([gq1.ravel(), gq2.ravel()])

                var_weight_space = sm.predict_variances(xq).ravel()
                var_dual = _dual_route_variances(sm, xq)

                gap = np.max(np.abs(var_weight_space - var_dual))
                self.assertLess(gap / np.ptp(var_weight_space), 1e-3)

    def test_multioutput_variance_shape(self):
        rng = np.random.default_rng(4)
        x = rng.uniform(0, 1, size=(64, 2))
        y = np.column_stack(
            [
                np.sin(3 * x[:, 0]) * np.cos(3 * x[:, 1]),
                x[:, 0] ** 2 - x[:, 1],
            ]
        )
        sm = RMTBUQ(**RMTB_KW)
        sm.set_training_values(x, y)
        with Silence():
            sm.train()

        xq = rng.uniform(0, 1, size=(10, 2))
        s2 = sm.predict_variances(xq)
        self.assertEqual(s2.shape, (10, 2))
        self.assertTrue(np.all(s2 >= 0.0))

    def test_wahba_warns_near_exact_interpolation(self):
        """An energy_weight too small relative to n pushes edf to within a
        hair of n, where the Wahba RSS/(n - edf) estimator is no longer
        meaningful; RMTBUQ should warn rather than silently return a
        degenerate sigma2."""
        x, y = _make_dataset("regular")
        sm = RMTBUQ(**NEAR_INTERPOLATION_KW)
        sm.set_training_values(x, y)
        with self.assertWarns(UserWarning):
            with Silence():
                sm.train()
        self.assertGreater(sm.edf, 0.9 * len(y))

    def test_wahba_silent_when_dof_is_healthy(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._fit("random")

    def test_known_sigma2_never_warns_regardless_of_edf(self):
        """`sigma2` is a direct passthrough when not 'wahba' -- edf/dof are
        still computed (for `.edf` introspection) but don't gate anything."""
        x, y = _make_dataset("regular")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sm = RMTBUQ(**dict(NEAR_INTERPOLATION_KW, sigma2=0.1))
            sm.set_training_values(x, y)
            with Silence():
                sm.train()
        self.assertGreater(sm.edf, 0.9 * len(y))

    def test_approx_order_other_than_2_raises(self):
        """The Kimeldorf-Wahba correspondence RMTBUQ's variance relies on
        only holds for ordinary least-squares (approx_order=2); any other
        exponent must be rejected rather than silently trained."""
        for approx_order in (0, 1, 3, 4):
            with self.subTest(approx_order=approx_order):
                x, y = _make_dataset("random")
                kw = dict(RMTB_KW, approx_order=approx_order)
                sm = RMTBUQ(**kw)
                sm.set_training_values(x, y)
                with self.assertRaises(ValueError):
                    with Silence():
                        sm.train()

    def test_approx_order_defaults_to_2(self):
        """RMTBUQ overrides RMTB/RMTS's own approx_order default (4) with
        the only value its posterior variance formula is valid for."""
        sm = RMTBUQ(print_global=False)
        self.assertEqual(sm.options["approx_order"], 2)

    def test_order_sweep_matches_dual_route(self):
        """B-spline order is a basis/conditioning choice, not a requirement
        of the underlying math (unlike approx_order) -- the weight-space
        and dual-route variances should keep agreeing across a range of
        orders, at a fixed, already-sane num_ctrl_pts/energy_weight."""
        x, y = _make_dataset("random")
        gq = np.linspace(0.05, 0.95, 12)
        gq1, gq2 = np.meshgrid(gq, gq, indexing="ij")
        xq = np.column_stack([gq1.ravel(), gq2.ravel()])

        for order in (3, 4, 5, 6):
            with self.subTest(order=order):
                kw = dict(RMTB_KW, order=order)
                sm = RMTBUQ(**kw)
                sm.set_training_values(x, y)
                with Silence():
                    sm.train()

                var_weight_space = sm.predict_variances(xq).ravel()
                var_dual = _dual_route_variances(sm, xq)
                gap = np.max(np.abs(var_weight_space - var_dual))
                self.assertLess(gap / np.ptp(var_weight_space), 1e-2)


if __name__ == "__main__":
    unittest.main()
