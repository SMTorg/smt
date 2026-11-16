"""
Tests for differentiable JAX predictor exports across SMT surrogate models.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from smt.applications.mfck import MFCK  # noqa: E402
from smt.applications.mfk import MFK  # noqa: E402
from smt.applications.mfkpls import MFKPLS  # noqa: E402
from smt.applications.mfkplsk import MFKPLSK  # noqa: E402
from smt.applications.smfck import SMFCK  # noqa: E402
from smt.applications.smfk import SMFK  # noqa: E402
from smt.design_space import (  # noqa: E402
    CategoricalVariable,
    DesignSpace,
    FloatVariable,
)
from smt.kernels import PowExp  # noqa: E402
from smt.surrogate_models import (  # noqa: E402
    GEKPLS,
    IDW,
    KPLS,
    KPLSK,
    KRG,
    LS,
    MGP,
    QP,
    RBF,
    RMTS,
    SGP,
)
from smt.surrogate_models.cckrg import CoopCompKRG  # noqa: E402
from smt.surrogate_models.krg_based.krg_jax import (  # noqa: E402
    _get_matern_kernels,
    eval_corr_jax,
    eval_poly_jax,
)
from smt.surrogate_models.surrogate_model import SurrogateModel  # noqa: E402
from smt.utils.checks import _enable_jax_x64  # noqa: E402


def _closed_form_krg_variance(sm, x_test):
    """Exact Kriging variance sigma2 * (1 - r^T R^-1 r + u^T (F^T R^-1 F)^-1 u)."""
    from scipy import linalg

    par = sm.optimal_par
    R = par["C"] @ par["C"].T
    xn = (x_test - sm.X_offset) / sm.X_scale
    dx = xn[:, None, :] - sm.X_norma[None, :, :]
    r = sm.corr(sm._componentwise_distance(dx.reshape(-1, sm.nx))).reshape(
        x_test.shape[0], sm.nt
    )
    f = sm._regression_types[sm.options["poly"]](xn)
    Rinv_r = linalg.solve(R, r.T)
    a = sm.F.T @ Rinv_r - f.T
    mse = (
        1.0
        - np.sum(r.T * Rinv_r, axis=0)
        + np.sum(a * linalg.solve(sm.F.T @ linalg.solve(R, sm.F), a), axis=0)
    )
    return np.maximum(0.0, np.outer(mse, par["sigma2"]))


class TestJAXPredictors:
    @pytest.fixture(autouse=True)
    def setup_data(self):
        np.random.seed(42)
        self.xt = np.random.rand(20, 2)
        self.yt = (
            np.sin(self.xt[:, 0:1])
            + np.cos(self.xt[:, 1:2])
            + 0.5 * self.xt[:, 0:1] * self.xt[:, 1:2]
        )
        self.x_eval = np.random.rand(10, 2)
        self.pt = np.array([0.45, 0.55])
        with _enable_jax_x64(jax):
            yield

    @pytest.mark.parametrize(
        "corr", ["squar_exp", "abs_exp", "pow_exp", "matern32", "matern52"]
    )
    @pytest.mark.parametrize("poly", ["constant", "linear", "quadratic"])
    def test_krg_jax_prediction(self, corr, poly):
        kwargs = {"corr": corr, "poly": poly, "print_global": False}
        if corr == "pow_exp":
            kwargs["pow_exp_power"] = 1.5
        sm = KRG(**kwargs)
        sm.set_training_values(self.xt, self.yt)
        sm.train()

        y_np = sm.predict_values(self.x_eval)

        pred_jax = sm.get_jax_predictor()
        y_jax = pred_jax(jnp.asarray(self.x_eval))
        np.testing.assert_allclose(np.asarray(y_jax), y_np, atol=1e-10)

        # Autodiff derivative check for differentiable kernels
        if corr in ("squar_exp", "matern32", "matern52"):
            grad_fn = jax.grad(pred_jax)
            g = grad_fn(jnp.asarray(self.pt))
            assert not np.isnan(np.asarray(g)).any()
            if poly in ["constant", "linear"]:
                dy0 = sm.predict_derivatives(self.pt.reshape(1, -1), 0)[0, 0]
                dy1 = sm.predict_derivatives(self.pt.reshape(1, -1), 1)[0, 0]
                np.testing.assert_allclose(np.asarray(g), [dy0, dy1], atol=1e-9)

        # Variance prediction in JAX vs closed form and (for constant poly) NumPy
        pred_var_jax = sm.get_jax_predictor(return_variance=True)
        _, s2_jax = pred_var_jax(jnp.asarray(self.x_eval))
        s2_exact = _closed_form_krg_variance(sm, self.x_eval)
        np.testing.assert_allclose(np.asarray(s2_jax), s2_exact, rtol=1e-6, atol=1e-10)
        if poly == "constant":
            s2_np = sm.predict_variances(self.x_eval)
            np.testing.assert_allclose(np.asarray(s2_jax), s2_np, atol=1e-10)

        y_1d, s2_1d = pred_var_jax(jnp.asarray(self.pt))
        assert isinstance(y_1d, (float, jax.Array))
        assert isinstance(s2_1d, (float, jax.Array))

    @pytest.mark.parametrize("corr", ["squar_exp", "pow_exp", "abs_exp"])
    @pytest.mark.parametrize("poly", ["constant", "linear", "quadratic"])
    def test_kpls_jax_prediction(self, corr, poly):
        kwargs = {"n_comp": 1, "corr": corr, "poly": poly, "print_global": False}
        if corr == "pow_exp":
            kwargs["pow_exp_power"] = 1.5
        sm = KPLS(**kwargs)
        sm.set_training_values(self.xt, self.yt)
        sm.train()

        y_np = sm.predict_values(self.x_eval)
        pred_jax = sm.get_jax_predictor()
        y_jax = pred_jax(jnp.asarray(self.x_eval))
        np.testing.assert_allclose(np.asarray(y_jax), y_np, atol=1e-10)

        pred_var_jax = sm.get_jax_predictor(return_variance=True)
        _, s2_jax = pred_var_jax(jnp.asarray(self.x_eval))
        if poly == "constant":
            np.testing.assert_allclose(
                np.asarray(s2_jax), sm.predict_variances(self.x_eval), atol=1e-10
            )
        assert not np.isnan(np.asarray(s2_jax)).any()

    def test_gekpls_jax_prediction(self):
        x_gek = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        y_gek = np.array([[0.0], [0.5], [2.0], [1.0], [1.0]])
        dydx0 = np.array([[0.0], [1.0], [2.0], [0.0], [1.0]])
        dydx1 = np.array([[0.0], [1.0], [2.0], [1.0], [0.0]])
        gekpls = GEKPLS(theta0=[1e-2, 1e-2], n_comp=2, print_global=False)
        gekpls.set_training_values(x_gek, y_gek)
        gekpls.set_training_derivatives(x_gek, dydx0, 0)
        gekpls.set_training_derivatives(x_gek, dydx1, 1)
        gekpls.train()

        y_np = gekpls.predict_values(self.x_eval)
        pred_jax = gekpls.get_jax_predictor()
        y_jax = pred_jax(jnp.asarray(self.x_eval))
        np.testing.assert_allclose(np.asarray(y_jax), y_np, atol=1e-10)

    @pytest.mark.parametrize("model_cls", [MFK, MFKPLS, MFKPLSK])
    @pytest.mark.parametrize("poly", ["constant", "linear"])
    @pytest.mark.parametrize("rho_regr", ["constant", "linear"])
    def test_mfk_jax_prediction(self, model_cls, poly, rho_regr):
        xt_low = np.random.rand(25, 2)
        yt_low = np.sin(xt_low[:, 0:1]) + np.cos(xt_low[:, 1:2])
        xt_high = np.random.rand(10, 2)
        yt_high = 2.0 * (np.sin(xt_high[:, 0:1]) + np.cos(xt_high[:, 1:2])) + 0.1

        kwargs = {
            "corr": "squar_exp",
            "poly": poly,
            "rho_regr": rho_regr,
            "print_global": False,
        }
        if model_cls in (MFKPLS, MFKPLSK):
            kwargs["n_comp"] = 1
        sm = model_cls(**kwargs)
        sm.set_training_values(xt_low, yt_low, name=0)
        sm.set_training_values(xt_high, yt_high)
        sm.train()

        # When MFKPLS with n_comp=1 drives theta toward 1e-6, ||gamma|| can reach ~1e11,
        # giving ~1e-4 IEEE-754 BLAS-vs-XLA contraction rounding despite bitwise-identical kernels.
        tol = 1e-3 if model_cls is MFKPLS else 1e-7
        y_np = sm.predict_values(self.x_eval)
        pred_jax = sm.get_jax_predictor()
        y_jax = pred_jax(jnp.asarray(self.x_eval))
        np.testing.assert_allclose(np.asarray(y_jax), y_np, rtol=tol, atol=tol)

        if model_cls is MFKPLS:
            sm_exact = MFKPLS(
                n_comp=1,
                corr="squar_exp",
                poly=poly,
                rho_regr=rho_regr,
                hyper_opt="NoOp",
                theta0=[0.5],
                print_global=False,
            )
            sm_exact.set_training_values(xt_low, yt_low, name=0)
            sm_exact.set_training_values(xt_high, yt_high)
            sm_exact.train()
            np.testing.assert_allclose(
                np.asarray(sm_exact.get_jax_predictor()(jnp.asarray(self.x_eval))),
                sm_exact.predict_values(self.x_eval),
                atol=1e-10,
            )

        grad_fn = jax.grad(pred_jax)
        g = grad_fn(jnp.asarray(self.pt))
        assert not np.isnan(np.asarray(g)).any()

    @pytest.mark.parametrize("eval_noise", [False, True])
    def test_krg_noisy_reinterpolation_variance(self, eval_noise):
        rng = np.random.default_rng(7)
        yt_noisy = (
            self.yt + rng.normal(0.0, 0.1, self.yt.shape) if eval_noise else self.yt
        )
        sm = KRG(
            noise0=[0.05] if eval_noise else [0.01],
            eval_noise=eval_noise,
            is_ri=True,
            poly="constant",
            print_global=False,
        )
        sm.set_training_values(self.xt, yt_noisy)
        sm.train()

        s2_results = {}
        for is_ri in (False, True):
            s2_np = sm.predict_variances(self.x_eval, is_ri=is_ri)
            pred_var_jax = sm.get_jax_predictor(return_variance=True, is_ri=is_ri)
            _, s2_jax = pred_var_jax(jnp.asarray(self.x_eval))
            np.testing.assert_allclose(np.asarray(s2_jax), s2_np, atol=1e-10)
            s2_results[is_ri] = np.asarray(s2_jax)

        assert not np.allclose(s2_results[False], s2_results[True])

        # Model trained with is_ri=False rejects is_ri=True export cleanly
        sm_no_ri = KRG(noise0=[0.01], eval_noise=False, is_ri=False, print_global=False)
        sm_no_ri.set_training_values(self.xt, self.yt)
        sm_no_ri.train()
        with pytest.raises(RuntimeError, match="is_ri"):
            sm_no_ri.get_jax_predictor(return_variance=True, is_ri=True)

    def test_multi_output_jax(self):
        yt_multi = np.hstack([self.yt, 2.0 * self.yt])
        for model_cls, kwargs in [
            (LS, {}),
            (QP, {}),
            (IDW, {}),
            (RBF, {"d0": [1.0, 1.0]}),
            (KRG, {}),
        ]:
            sm = model_cls(print_global=False, **kwargs)
            sm.set_training_values(self.xt, yt_multi)
            sm.train()

            pred_jax = sm.get_jax_predictor()
            val_1d = pred_jax(jnp.asarray(self.pt))
            assert val_1d.shape == (2,)
            np.testing.assert_allclose(
                np.asarray(pred_jax(jnp.asarray(self.x_eval))),
                sm.predict_values(self.x_eval),
                atol=1e-10,
            )

            if model_cls is KRG:
                pred_var = sm.get_jax_predictor(return_variance=True)
                y_1d, s2_1d = pred_var(jnp.asarray(self.pt))
                assert y_1d.shape == (2,)
                assert s2_1d.shape == (2,)

    def test_rmtb_jax_prediction(self):
        sm = RMTS(
            spline_type="B-Spline",
            xlimits=np.array([[0.0, 1.0], [0.0, 1.0]]),
            order=3,
            num_ctrl_pts=6,
            print_global=False,
        )
        sm.set_training_values(self.xt, self.yt)
        sm.train()

        y_np = sm.predict_values(self.x_eval)
        pred_jax = sm.get_jax_predictor()
        y_jax = pred_jax(jnp.asarray(self.x_eval))
        assert y_jax.shape == (self.x_eval.shape[0], 1)
        assert pred_jax(jnp.asarray(self.pt)).ndim == 0
        np.testing.assert_allclose(np.asarray(y_jax), y_np, atol=1e-12)

        grad_fn = jax.grad(pred_jax)
        g = grad_fn(jnp.asarray(self.pt))
        assert not np.isnan(np.asarray(g)).any()

        bs = sm.get_bspline()
        assert "knots" in bs and "coefficients" in bs
        scipy_spline = sm.to_scipy()
        assert callable(scipy_spline)

    def test_rmtc_jax_prediction(self):
        sm = RMTS(
            spline_type="Cubic_Hermite_Spline",
            xlimits=np.array([[0.0, 1.0], [0.0, 1.0]]),
            num_elements=4,
            print_global=False,
        )
        sm.set_training_values(self.xt, self.yt)
        sm.train()

        y_np = sm.predict_values(self.x_eval)
        pred_jax = sm.get_jax_predictor()
        y_jax = pred_jax(jnp.asarray(self.x_eval))
        assert y_jax.shape == (self.x_eval.shape[0], 1)
        assert pred_jax(jnp.asarray(self.pt)).ndim == 0
        np.testing.assert_allclose(np.asarray(y_jax), y_np, atol=1e-12)

        grad_fn = jax.grad(pred_jax)
        g = grad_fn(jnp.asarray(self.pt))
        assert not np.isnan(np.asarray(g)).any()

    def test_ls_jax_prediction(self):
        sm = LS(print_global=False)
        sm.set_training_values(self.xt, self.yt)
        sm.train()

        y_np = sm.predict_values(self.x_eval)
        pred_jax = sm.get_jax_predictor()
        y_jax = pred_jax(jnp.asarray(self.x_eval))
        np.testing.assert_allclose(np.asarray(y_jax), y_np, atol=1e-12)

        grad_fn = jax.grad(pred_jax)
        g = grad_fn(jnp.asarray(self.pt))
        dy0 = sm.predict_derivatives(self.pt.reshape(1, -1), 0)[0, 0]
        dy1 = sm.predict_derivatives(self.pt.reshape(1, -1), 1)[0, 0]
        np.testing.assert_allclose(np.asarray(g), [dy0, dy1], atol=1e-12)

    def test_qp_jax_prediction(self):
        sm = QP(print_global=False)
        sm.set_training_values(self.xt, self.yt)
        sm.train()

        y_np = sm.predict_values(self.x_eval)
        pred_jax = sm.get_jax_predictor()
        y_jax = pred_jax(jnp.asarray(self.x_eval))
        np.testing.assert_allclose(np.asarray(y_jax), y_np, atol=1e-12)

        grad_fn = jax.grad(pred_jax)
        g = grad_fn(jnp.asarray(self.pt))
        dy0 = sm.predict_derivatives(self.pt.reshape(1, -1), 0)[0, 0]
        dy1 = sm.predict_derivatives(self.pt.reshape(1, -1), 1)[0, 0]
        np.testing.assert_allclose(np.asarray(g), [dy0, dy1], atol=1e-10)

    def test_idw_and_rbf_jax_predictors(self):
        sm_idw = IDW(print_global=False)
        sm_idw.set_training_values(self.xt, self.yt)
        sm_idw.train()
        pred_jax_idw = sm_idw.get_jax_predictor()
        np.testing.assert_allclose(
            np.asarray(pred_jax_idw(jnp.asarray(self.x_eval))),
            sm_idw.predict_values(self.x_eval),
            atol=1e-12,
        )
        assert isinstance(pred_jax_idw(jnp.asarray(self.pt)), (float, jax.Array))

        sm_rbf = RBF(d0=[1.0, 1.0], poly_degree=1, print_global=False)
        sm_rbf.set_training_values(self.xt, self.yt)
        sm_rbf.train()
        pred_jax_rbf = sm_rbf.get_jax_predictor()
        np.testing.assert_allclose(
            np.asarray(pred_jax_rbf(jnp.asarray(self.x_eval))),
            sm_rbf.predict_values(self.x_eval),
            atol=1e-10,
        )
        assert isinstance(pred_jax_rbf(jnp.asarray(self.pt)), (float, jax.Array))

    def test_supports_jax_opt_in_and_opt_out(self):
        for opt_in_cls, kwargs in [
            (KRG, {}),
            (KPLS, {"n_comp": 1}),
            (KPLSK, {"n_comp": 1}),
            (GEKPLS, {"n_comp": 2}),
            (MFK, {}),
            (MFKPLS, {"n_comp": 1}),
            (MFKPLSK, {"n_comp": 1}),
            (LS, {}),
            (QP, {}),
            (IDW, {}),
            (RBF, {"d0": [1.0, 1.0]}),
            (RMTS, {"xlimits": np.array([[0.0, 1.0], [0.0, 1.0]])}),
        ]:
            sm = opt_in_cls(print_global=False, **kwargs)
            assert sm.supports["jax"] is True

        for opt_out_cls, kwargs in [
            (SGP, {}),
            (MGP, {}),
            (SMFK, {}),
            (MFCK, {}),
            (SMFCK, {}),
            (CoopCompKRG, {}),
        ]:
            sm = opt_out_cls(print_global=False, **kwargs)
            assert sm.supports["jax"] is False
            with pytest.raises(NotImplementedError):
                sm.get_jax_predictor()

        # Trained SMFK (level 0 is SGP) and MFCK must still reject get_jax_predictor
        xt_low = np.random.rand(15, 2)
        yt_low = np.sin(xt_low[:, 0:1]) + np.cos(xt_low[:, 1:2])
        xt_high = np.random.rand(8, 2)
        yt_high = 2.0 * (np.sin(xt_high[:, 0:1]) + np.cos(xt_high[:, 1:2]))

        smfk = SMFK(n_inducing=5, print_global=False)
        smfk.set_training_values(xt_low, yt_low, name=0)
        smfk.set_training_values(xt_high, yt_high)
        smfk.train()
        with pytest.raises(NotImplementedError):
            smfk.get_jax_predictor()

        mfck = MFCK(print_global=False)
        mfck.set_training_values(xt_low, yt_low, name=0)
        mfck.set_training_values(xt_high, yt_high)
        mfck.train()
        with pytest.raises(NotImplementedError):
            mfck.get_jax_predictor()

    def test_exceptions_and_edge_cases(self):
        # Untrained models raise RuntimeError
        for model_cls, kwargs in [
            (KRG, {}),
            (LS, {}),
            (QP, {}),
            (IDW, {}),
            (RBF, {"d0": [1.0, 1.0]}),
            (RMTS, {"xlimits": np.array([[0.0, 1.0], [0.0, 1.0]])}),
            (MFK, {}),
        ]:
            sm = model_cls(print_global=False, **kwargs)
            with pytest.raises(RuntimeError):
                sm.get_jax_predictor()

        # Non-continuous design space raises NotImplementedError
        ds = DesignSpace([CategoricalVariable(values=[0, 1]), FloatVariable(0.0, 1.0)])
        sm_cat = KRG(design_space=ds, print_global=False)
        xt_cat = np.array([[0, 0.2], [1, 0.4], [0, 0.6], [1, 0.8]])
        yt_cat = np.array([[1.0], [2.0], [3.0], [4.0]])
        sm_cat.set_training_values(xt_cat, yt_cat)
        sm_cat.train()
        with pytest.raises(NotImplementedError):
            sm_cat.get_jax_predictor()

        # Unsupported correlation kernel (squar_sin_exp and custom Kernel)
        sm_sin = KRG(corr="squar_sin_exp", print_global=False)
        sm_sin.set_training_values(self.xt, self.yt)
        sm_sin.train()
        with pytest.raises(NotImplementedError):
            sm_sin.get_jax_predictor()

        sm_custom = KRG(corr=PowExp(np.ones(2)), print_global=False)
        sm_custom.set_training_values(self.xt, self.yt)
        sm_custom.train()
        with pytest.raises(NotImplementedError):
            sm_custom.get_jax_predictor()

        from unittest.mock import patch

        sm_mfk = MFK(print_global=False)
        sm_mfk.set_training_values(self.xt, self.yt, name=0)
        sm_mfk.set_training_values(self.xt, self.yt)
        sm_mfk.train()
        with patch.object(MFK, "is_continuous", False):
            with pytest.raises(NotImplementedError):
                sm_mfk.get_jax_predictor()

        with pytest.raises(NotImplementedError):
            eval_poly_jax("unsupported_poly", jnp.ones((2, 2)), 2)

        with pytest.raises(NotImplementedError):
            eval_corr_jax("unsupported_corr", jnp.ones((2, 2, 2)), jnp.ones(2))

        class DummyJaxModel(SurrogateModel):
            name = "DummyJax"

            def _initialize(self):
                self.supports["jax"] = True

            def _train(self):
                pass

            def _predict_values(self, x):
                return x

        with pytest.raises(NotImplementedError):
            DummyJaxModel().get_jax_predictor()

    def test_snapshot_and_update_model_semantics(self):
        sm = KRG(poly="constant", print_global=False)
        sm.set_training_values(self.xt[:12], self.yt[:12])
        sm.train()

        pred_before = sm.get_jax_predictor(return_variance=True)
        mu1, s2_1 = pred_before(jnp.asarray(self.x_eval))
        np.testing.assert_allclose(
            np.asarray(mu1), sm.predict_values(self.x_eval), atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(s2_1), sm.predict_variances(self.x_eval), atol=1e-10
        )

        # In-place update_model changes predictions; pred_before remains a snapshot,
        # while re-exporting get_jax_predictor() captures the updated model.
        sm.update_model(self.xt[12:], self.yt[12:])
        y_updated_np = sm.predict_values(self.x_eval)
        s2_updated_np = sm.predict_variances(self.x_eval)
        assert not np.allclose(np.asarray(mu1), y_updated_np)

        pred_after = sm.get_jax_predictor(return_variance=True)
        mu2, s2_2 = pred_after(jnp.asarray(self.x_eval))
        np.testing.assert_allclose(np.asarray(mu2), y_updated_np, atol=1e-10)
        np.testing.assert_allclose(np.asarray(s2_2), s2_updated_np, atol=1e-10)

    def test_kplsk_jax_prediction(self):
        sm_kplsk = KPLSK(n_comp=1, poly="linear", print_global=False)
        sm_kplsk.set_training_values(self.xt, self.yt)
        sm_kplsk.train()
        np.testing.assert_allclose(
            np.asarray(sm_kplsk.get_jax_predictor()(jnp.asarray(self.x_eval))),
            sm_kplsk.predict_values(self.x_eval),
            atol=1e-10,
        )

    def test_multioutput_qp_and_rmts_extrapolation(self):
        yt_multi = np.hstack([self.yt, 3.0 * self.yt + 1.0])

        sm_qp = QP(print_global=False)
        sm_qp.set_training_values(self.xt, yt_multi)
        sm_qp.train()
        pred_qp = sm_qp.get_jax_predictor()
        np.testing.assert_allclose(
            np.asarray(pred_qp(jnp.asarray(self.x_eval))),
            sm_qp.predict_values(self.x_eval),
            atol=1e-10,
        )
        eps = 1e-6
        for kx in [0, 1]:
            dx = np.zeros_like(self.x_eval)
            dx[:, kx] = eps
            fd = (
                sm_qp.predict_values(self.x_eval + dx)
                - sm_qp.predict_values(self.x_eval - dx)
            ) / (2 * eps)
            np.testing.assert_allclose(
                sm_qp.predict_derivatives(self.x_eval, kx), fd, atol=1e-6
            )

        sm_krg_m = KRG(poly="linear", print_global=False)
        sm_krg_m.set_training_values(self.xt, yt_multi)
        sm_krg_m.train()
        pred_krg_m = sm_krg_m.get_jax_predictor()
        assert pred_krg_m(jnp.asarray(self.pt)).shape == (2,)
        np.testing.assert_allclose(
            np.asarray(pred_krg_m(jnp.asarray(self.x_eval))),
            sm_krg_m.predict_values(self.x_eval),
            atol=1e-10,
        )

        # RMTS extrapolate=True/False boundary derivatives & explicit float32 dtype
        x_mixed = np.array([[0.5, 0.5], [-0.2, 0.5], [1.2, 0.8], [0.3, 1.4]])
        x_out = np.array([[-0.3, 0.5], [1.4, 0.2]])

        for spline_kw in [
            {"spline_type": "B-Spline", "order": 3, "num_ctrl_pts": 5},
            {"spline_type": "Cubic_Hermite_Spline", "num_elements": 4},
        ]:
            sm_ext = RMTS(
                xlimits=np.array([[0.0, 1.0], [0.0, 1.0]]),
                extrapolate=True,
                print_global=False,
                **spline_kw,
            )
            sm_ext.set_training_values(self.xt, yt_multi)
            sm_ext.train()
            y_np_ext = sm_ext.predict_values(x_mixed)
            np.testing.assert_allclose(
                np.asarray(sm_ext.get_jax_predictor()(x_mixed)), y_np_ext, atol=1e-10
            )
            # Zero-gradient / interior lax.cond parity
            jac_ext = np.asarray(
                jax.vmap(jax.jacfwd(sm_ext.get_jax_predictor()))(x_mixed)
            )
            for kx in [0, 1]:
                np.testing.assert_allclose(
                    jac_ext[:, :, kx],
                    sm_ext.predict_derivatives(x_mixed, kx),
                    atol=1e-8,
                )

            pred_fp32 = sm_ext.get_jax_predictor(dtype=jnp.float32)
            y_fp32 = pred_fp32(x_mixed)
            assert y_fp32.dtype == jnp.float32
            np.testing.assert_allclose(np.asarray(y_fp32), y_np_ext, atol=1e-4)

            sm_noext = RMTS(
                xlimits=np.array([[0.0, 1.0], [0.0, 1.0]]),
                extrapolate=False,
                print_global=False,
                **spline_kw,
            )
            sm_noext.set_training_values(self.xt, yt_multi)
            sm_noext.train()
            jac_out = np.asarray(
                jax.vmap(jax.jacfwd(sm_noext.get_jax_predictor()))(x_out)
            )
            np.testing.assert_allclose(jac_out, 0.0, atol=1e-12)

    @pytest.mark.parametrize("corr", ["matern32", "matern52"])
    def test_matern_second_order_ad_at_training_points(self, corr):
        matern32_1d, matern52_1d = _get_matern_kernels()
        kernel_1d, expected_d2_at_zero = (
            (matern32_1d, -3.0) if corr == "matern32" else (matern52_1d, -5.0 / 3.0)
        )
        d2_at_zero = jax.grad(jax.grad(kernel_1d))(jnp.float64(0.0))
        np.testing.assert_allclose(float(d2_at_zero), expected_d2_at_zero, atol=1e-12)

        sm = KRG(corr=corr, theta0=[0.8, 1.2], print_global=False)
        sm.set_training_values(self.xt, self.yt)
        sm.train()

        pred_fn = sm.get_jax_predictor()
        hess_fn = jax.jit(jax.hessian(pred_fn))
        grad_fn = jax.jit(jax.grad(pred_fn))

        x0 = jnp.asarray(self.xt[0], dtype=jnp.float64)
        h_ad = np.asarray(hess_fn(x0))
        assert np.all(np.isfinite(h_ad))
        np.testing.assert_allclose(h_ad, h_ad.T, atol=1e-12)

        eps = 1e-6 if corr == "matern32" else 1e-4
        h_fd = np.zeros((2, 2))
        for j in range(2):
            step = np.zeros(2)
            step[j] = eps
            g_plus = np.asarray(grad_fn(jnp.asarray(self.xt[0] + step)))
            g_minus = np.asarray(grad_fn(jnp.asarray(self.xt[0] - step)))
            h_fd[:, j] = (g_plus - g_minus) / (2.0 * eps)

        assert np.all(np.abs(np.diag(h_ad)) > 1e-3)
        np.testing.assert_allclose(h_ad, h_fd, rtol=1e-5, atol=1e-5)

    def test_1d_scalar_and_shape_validation(self):
        """JAX predictors accept 0D scalars when nx==1 and validate feature count/rank."""
        xt1 = np.linspace(0.0, 1.0, 9)[:, None]
        yt1 = np.sin(np.pi * xt1)
        x_pt = np.array([[0.4]])
        x_batch = jnp.array([0.2, 0.4, 0.7])

        models = [
            RMTS(
                spline_type="B-Spline",
                xlimits=np.array([[0.0, 1.0]]),
                order=3,
                num_ctrl_pts=5,
                print_global=False,
            ),
            RMTS(
                spline_type="Cubic_Hermite_Spline",
                xlimits=np.array([[0.0, 1.0]]),
                num_elements=4,
                print_global=False,
            ),
            KRG(corr="squar_exp", print_global=False),
            LS(print_global=False),
            QP(print_global=False),
            MFK(print_global=False),
        ]

        for sm in models:
            if isinstance(sm, MFK):
                sm.set_training_values(xt1, 0.8 * yt1, name=0)
            sm.set_training_values(xt1, yt1)
            sm.train()

            pred = sm.get_jax_predictor()
            y_0d = pred(0.4)
            assert y_0d.ndim == 0
            y_ref = sm.predict_values(x_pt)[0, 0]
            np.testing.assert_allclose(float(y_0d), y_ref, atol=1e-10)

            dy_0d = float(jax.grad(pred)(0.4))
            dy_ref = sm.predict_derivatives(x_pt, 0)[0, 0]
            np.testing.assert_allclose(dy_0d, dy_ref, atol=1e-8)

            y_vmap = np.asarray(jax.vmap(pred)(x_batch))
            assert y_vmap.shape == (3,)
            np.testing.assert_allclose(
                y_vmap,
                sm.predict_values(np.asarray(x_batch)[:, None])[:, 0],
                atol=1e-10,
            )

            with pytest.raises(ValueError):
                pred(np.array([0.2, 0.4]))
            with pytest.raises(ValueError):
                pred(np.zeros((1, 2, 1)))

        pred_var = models[2].get_jax_predictor(return_variance=True)
        mu_0d, s2_0d = pred_var(0.4)
        assert mu_0d.ndim == 0 and s2_0d.ndim == 0
        np.testing.assert_allclose(
            float(mu_0d), models[2].predict_values(x_pt)[0, 0], atol=1e-10
        )
        np.testing.assert_allclose(
            float(s2_0d), models[2].predict_variances(x_pt)[0, 0], atol=1e-10
        )

        sm_2d = RMTS(
            spline_type="B-Spline",
            xlimits=np.array([[0.0, 1.0], [0.0, 1.0]]),
            order=3,
            num_ctrl_pts=5,
            print_global=False,
        )
        sm_2d.set_training_values(self.xt, self.yt)
        sm_2d.train()
        with pytest.raises(ValueError):
            sm_2d.get_jax_predictor()(0.5)

    def test_krg_analytic_derivatives_and_trend_variance_match_jax(self):
        """Verify JAX trend variances against closed form and gradients against NumPy."""
        rng = np.random.default_rng(0)
        X = rng.random((50, 4))
        y = (
            np.sin(3 * X[:, 0])
            + (X[:, 1:] ** 2).sum(axis=1)
            + 0.3 * np.cos(5 * X[:, -1])
        )
        x_test = np.random.default_rng(1).random((6, 4))

        for poly in ["constant", "linear", "quadratic"]:
            sm = KRG(
                corr="squar_exp",
                poly=poly,
                hyper_opt="NoOp",
                theta0=[0.5] * 4,
                print_global=False,
            )
            sm.set_training_values(X, y)
            sm.train()

            _, s2_jax = sm.get_jax_predictor(return_variance=True)(jnp.asarray(x_test))
            s2_exact = _closed_form_krg_variance(sm, x_test)
            np.testing.assert_allclose(np.asarray(s2_jax), s2_exact, rtol=1e-6)
            if poly == "constant":
                np.testing.assert_allclose(
                    np.asarray(s2_jax),
                    sm.predict_variances(x_test),
                    rtol=1e-8,
                    atol=1e-12,
                )

        for corr in ["squar_exp", "abs_exp", "matern32", "matern52"]:
            for poly in ["constant", "linear"]:
                sm = KRG(
                    corr=corr,
                    poly=poly,
                    hyper_opt="NoOp",
                    theta0=[0.5] * 4,
                    print_global=False,
                )
                sm.set_training_values(X, y)
                sm.train()

                pred_mean = sm.get_jax_predictor()
                jac_mean = np.asarray(
                    jax.vmap(jax.grad(pred_mean))(jnp.asarray(x_test))
                )
                analytic_mean = np.hstack(
                    [sm.predict_derivatives(x_test, kx) for kx in range(4)]
                )
                np.testing.assert_allclose(
                    analytic_mean, jac_mean, rtol=1e-8, atol=1e-10
                )

                pred_var = sm.get_jax_predictor(return_variance=True)
                jac_var = np.asarray(
                    jax.vmap(jax.grad(lambda x: pred_var(x)[1]))(jnp.asarray(x_test))
                )
                analytic_var = np.hstack(
                    [sm.predict_variance_derivatives(x_test, kx) for kx in range(4)]
                )
                np.testing.assert_allclose(analytic_var, jac_var, rtol=1e-6, atol=1e-9)


def test_jax_predictor_respects_and_preserves_x64_config():
    """get_jax_predictor() follows caller's jax_enable_x64 without mutating global config."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from smt.applications.mfk import MFK
        from smt.surrogate_models import IDW, KRG, LS, QP, RBF, RMTS
        from smt.utils.checks import _enable_jax_x64

        assert not jax.config.jax_enable_x64
        rng = np.random.default_rng(0)
        xt = rng.random((20, 2))
        yt = np.sin(xt[:, :1]) + np.cos(xt[:, 1:2])
        x_eval = rng.random((6, 2))

        models = []
        for cls, kw in [
            (KRG, {}),
            (LS, {}),
            (QP, {}),
            (IDW, {}),
            (RBF, {"d0": [1.0, 1.0]}),
            (RMTS, {"xlimits": np.array([[0.0, 1.0], [0.0, 1.0]]), "spline_type": "B-Spline"}),
            (RMTS, {"xlimits": np.array([[0.0, 1.0], [0.0, 1.0]]), "spline_type": "Cubic_Hermite_Spline"}),
            (MFK, {}),
        ]:
            m = cls(print_global=False, **kw)
            if cls is MFK:
                m.set_training_values(xt, 0.9 * yt, name=0)
            m.set_training_values(xt, yt)
            m.train()
            models.append(m)

            # Default x64=False -> float32 predictor without mutating global flag
            pred32 = jax.jit(m.get_jax_predictor())
            y32 = pred32(jnp.asarray(x_eval, dtype=jnp.float32))
            assert y32.dtype == jnp.float32
            assert not jax.config.jax_enable_x64

        with _enable_jax_x64(jax):
            for m in models:
                pred64 = jax.jit(m.get_jax_predictor())
                y64 = pred64(jnp.asarray(x_eval, dtype=jnp.float64))
                assert y64.dtype == jnp.float64
                np.testing.assert_allclose(
                    np.asarray(y64), m.predict_values(x_eval), rtol=1e-7, atol=1e-7
                )

        assert not jax.config.jax_enable_x64
        """
    )
    subprocess.check_call([sys.executable, "-c", code])
