"""
Tests for JAX predictors and accelerated inference across SMT surrogate models.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from smt.applications.mfk import MFK  # noqa: E402
from smt.applications.mfkpls import MFKPLS  # noqa: E402
from smt.applications.mfkplsk import MFKPLSK  # noqa: E402
from smt.design_space import CategoricalVariable, DesignSpace, FloatVariable  # noqa: E402
from smt.surrogate_models import (  # noqa: E402
    GEKPLS,
    IDW,
    KPLS,
    KPLSK,
    KRG,
    LS,
    MGP,
    PCE,
    QP,
    RBF,
    RMTS,
    SGP,
    SparsePCE,
)
from smt.surrogate_models.krg_based.krg_jax import eval_corr_jax, eval_poly_jax  # noqa: E402
from smt.surrogate_models.surrogate_model import SurrogateModel  # noqa: E402
from smt.utils.checks import _enable_jax_x64  # noqa: E402


class TestJAXAcceleration:
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

    @pytest.mark.parametrize("corr", ["squar_exp", "abs_exp", "matern32", "matern52"])
    @pytest.mark.parametrize("poly", ["constant", "linear", "quadratic"])
    def test_krg_jax_prediction(self, corr, poly):
        sm = KRG(corr=corr, poly=poly, print_global=False)
        sm.set_training_values(self.xt, self.yt)
        sm.train()

        # Passive / NumPy prediction
        y_np = sm.predict_values(self.x_eval)

        # JAX predictor
        pred_jax = sm.get_jax_predictor()
        y_jax = pred_jax(jnp.asarray(self.x_eval))
        np.testing.assert_allclose(np.asarray(y_jax), y_np, atol=1e-10)

        # Accelerated predict_values with use_jax=True
        sm.options["use_jax"] = True
        y_accel = sm.predict_values(self.x_eval)
        np.testing.assert_allclose(y_accel, y_np, atol=1e-10)

        # Autodiff derivative test (squar_exp has smooth derivatives)
        if corr == "squar_exp":
            grad_fn = jax.grad(pred_jax)
            g = grad_fn(jnp.asarray(self.pt))
            assert not np.isnan(np.asarray(g)).any()
            if poly in ["constant", "linear"]:
                dy0 = sm.predict_derivatives(self.pt.reshape(1, -1), 0)[0, 0]
                dy1 = sm.predict_derivatives(self.pt.reshape(1, -1), 1)[0, 0]
                np.testing.assert_allclose(np.asarray(g), [dy0, dy1], atol=1e-10)

        # Variance prediction in JAX (both batch and 1D)
        pred_var_jax = sm.get_jax_predictor(return_variance=True)
        _, s2_jax = pred_var_jax(jnp.asarray(self.x_eval))
        s2_np = sm.predict_variances(self.x_eval)
        np.testing.assert_allclose(np.asarray(s2_jax), s2_np, atol=1e-10)

        y_1d, s2_1d = pred_var_jax(jnp.asarray(self.pt))
        assert isinstance(y_1d, (float, jax.Array))
        assert isinstance(s2_1d, (float, jax.Array))

    @pytest.mark.parametrize(
        "corr", ["squar_exp", "pow_exp", "abs_exp", "matern32", "matern52"]
    )
    def test_kpls_jax_prediction(self, corr):
        kwargs = {"n_comp": 1, "corr": corr, "poly": "linear", "print_global": False}
        if corr == "pow_exp":
            kwargs["pow_exp_power"] = 1.5
        sm = KPLS(**kwargs)
        sm.set_training_values(self.xt, self.yt)
        sm.train()

        y_np = sm.predict_values(self.x_eval)
        pred_jax = sm.get_jax_predictor()
        y_jax = pred_jax(jnp.asarray(self.x_eval))
        np.testing.assert_allclose(np.asarray(y_jax), y_np, atol=1e-10)

        sm.options["use_jax"] = True
        y_accel = sm.predict_values(self.x_eval)
        np.testing.assert_allclose(y_accel, y_np, atol=1e-10)

    def test_krg_pow_exp(self):
        sm = KRG(corr="pow_exp", pow_exp_power=1.5, poly="constant", print_global=False)
        sm.set_training_values(self.xt, self.yt)
        sm.train()

        y_np = sm.predict_values(self.x_eval)
        pred_jax = sm.get_jax_predictor()
        y_jax = pred_jax(jnp.asarray(self.x_eval))
        np.testing.assert_allclose(np.asarray(y_jax), y_np, atol=1e-10)

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

    def test_mfk_jax_prediction(self):
        xt_low = np.random.rand(25, 2)
        yt_low = np.sin(xt_low[:, 0:1]) + np.cos(xt_low[:, 1:2])
        xt_high = np.random.rand(10, 2)
        yt_high = 2.0 * (np.sin(xt_high[:, 0:1]) + np.cos(xt_high[:, 1:2])) + 0.1

        for model_cls in [MFK, MFKPLS]:
            kwargs = {
                "corr": "squar_exp",
                "poly": "constant",
                "rho_regr": "constant",
                "print_global": False,
            }
            if model_cls is MFKPLS:
                kwargs["n_comp"] = 1
            sm = model_cls(**kwargs)
            sm.set_training_values(xt_low, yt_low, name=0)
            sm.set_training_values(xt_high, yt_high)
            sm.train()

            tol = 1e-3 if model_cls is MFKPLS else 1e-10
            y_np = sm.predict_values(self.x_eval)
            pred_jax = sm.get_jax_predictor()
            y_jax = pred_jax(jnp.asarray(self.x_eval))
            np.testing.assert_allclose(np.asarray(y_jax), y_np, atol=tol)

            sm.options["use_jax"] = True
            y_accel = sm.predict_values(self.x_eval)
            np.testing.assert_allclose(y_accel, y_np, atol=tol)

            # Autodiff on MFK
            grad_fn = jax.grad(pred_jax)
            g = grad_fn(jnp.asarray(self.pt))
            assert not np.isnan(np.asarray(g)).any()

    def test_krg_noisy_reinterpolation_variance(self):
        sm = KRG(noise0=[0.01], eval_noise=False, is_ri=True, print_global=False)
        sm.set_training_values(self.xt, self.yt)
        sm.train()
        pred_var_jax = sm.get_jax_predictor(return_variance=True)
        _, s2_jax = pred_var_jax(jnp.asarray(self.x_eval))
        assert s2_jax.shape == (self.x_eval.shape[0], 1)

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

        sm.options["use_jax"] = True
        y_accel = sm.predict_values(self.x_eval)
        np.testing.assert_allclose(y_accel, y_np, atol=1e-12)

        grad_fn = jax.grad(pred_jax)
        g = grad_fn(jnp.asarray(self.pt))
        assert not np.isnan(np.asarray(g)).any()

        # Test SciPy export and B-spline dict
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

        sm.options["use_jax"] = True
        y_accel = sm.predict_values(self.x_eval)
        np.testing.assert_allclose(y_accel, y_np, atol=1e-12)

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

        sm.options["use_jax"] = True
        y_accel = sm.predict_values(self.x_eval)
        np.testing.assert_allclose(y_accel, y_np, atol=1e-12)

        # Autodiff gradient vs analytical derivatives
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

        sm.options["use_jax"] = True
        y_accel = sm.predict_values(self.x_eval)
        np.testing.assert_allclose(y_accel, y_np, atol=1e-12)

        # Autodiff gradient vs analytical derivatives
        grad_fn = jax.grad(pred_jax)
        g = grad_fn(jnp.asarray(self.pt))
        dy0 = sm.predict_derivatives(self.pt.reshape(1, -1), 0)[0, 0]
        dy1 = sm.predict_derivatives(self.pt.reshape(1, -1), 1)[0, 0]
        np.testing.assert_allclose(np.asarray(g), [dy0, dy1], atol=1e-10)

    def test_pce_jax_prediction(self):
        for model_cls, kwargs in [
            (PCE, {"degree": 3, "method": "ols"}),
            (SparsePCE, {"degree": [1, 2, 3, 4], "q_norm": 1.0}),
        ]:
            sm = model_cls(print_global=False, **kwargs)
            sm.set_training_values(self.xt, self.yt)
            sm.train()

            y_np = sm.predict_values(self.x_eval)
            pred_jax = sm.get_jax_predictor()
            y_jax = pred_jax(jnp.asarray(self.x_eval))
            np.testing.assert_allclose(np.asarray(y_jax), y_np, atol=1e-11)

            pred_fp32 = sm.get_jax_predictor(dtype="float32")
            y_fp32 = pred_fp32(jnp.asarray(self.x_eval, dtype=jnp.float32))
            assert y_fp32.dtype == jnp.float32
            np.testing.assert_allclose(np.asarray(y_fp32), y_np, atol=1e-4)

            grad_fn = jax.grad(pred_jax)
            g = np.asarray(grad_fn(jnp.asarray(self.pt)))
            dy0 = sm.predict_derivatives(self.pt.reshape(1, -1), 0)[0, 0]
            dy1 = sm.predict_derivatives(self.pt.reshape(1, -1), 1)[0, 0]
            np.testing.assert_allclose(g, [dy0, dy1], atol=1e-10)

        # Multi-output JAX evaluation (1D and batch)
        yt_multi = np.hstack([self.yt, 2.0 * self.yt])
        sm_m = PCE(degree=3, print_global=False)
        sm_m.set_training_values(self.xt, yt_multi)
        sm_m.train()
        pred_m = sm_m.get_jax_predictor()
        assert pred_m(jnp.asarray(self.pt)).shape == (2,)
        np.testing.assert_allclose(
            np.asarray(pred_m(jnp.asarray(self.x_eval))),
            sm_m.predict_values(self.x_eval),
            atol=1e-11,
        )

    def test_idw_and_rbf_jax_acceleration(self):
        # IDW
        sm_idw = IDW(print_global=False, use_jax=True)
        sm_idw.set_training_values(self.xt, self.yt)
        sm_idw.train()
        pred_jax_idw = sm_idw.get_jax_predictor()
        val_1d = pred_jax_idw(jnp.asarray(self.pt))
        assert isinstance(val_1d, (float, jax.Array))

        # RBF
        sm_rbf = RBF(d0=[1.0, 1.0], poly_degree=1, print_global=False, use_jax=True)
        sm_rbf.set_training_values(self.xt, self.yt)
        sm_rbf.train()
        pred_jax_rbf = sm_rbf.get_jax_predictor()
        val_rbf_1d = pred_jax_rbf(jnp.asarray(self.pt))
        assert isinstance(val_rbf_1d, (float, jax.Array))

    def test_exceptions_and_edge_cases(self):
        # Untrained models raise RuntimeError
        for model_cls in [KRG, LS, QP, RMTS, MFK, PCE, SparsePCE]:
            sm = model_cls(print_global=False)
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

        from unittest.mock import patch

        sm_mfk = MFK(print_global=False)
        sm_mfk.set_training_values(self.xt, self.yt, name=0)
        sm_mfk.set_training_values(self.xt, self.yt)
        sm_mfk.train()
        with patch.object(MFK, "is_continuous", False):
            with pytest.raises(NotImplementedError):
                sm_mfk.get_jax_predictor()

        # Helper functions invalid parameters raise NotImplementedError
        with pytest.raises(NotImplementedError):
            eval_poly_jax("unsupported_poly", jnp.ones((2, 2)), 2)

        with pytest.raises(NotImplementedError):
            eval_corr_jax("unsupported_corr", jnp.ones((2, 2, 2)), jnp.ones(2))

        # Base SurrogateModel with supports['jax'] = True raises NotImplementedError
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

        # predict_variance_gradient with array-like input
        sm_krg = KRG(print_global=False)
        sm_krg.set_training_values(self.xt, self.yt)
        sm_krg.train()
        grad = sm_krg.predict_variance_gradient(jnp.array([0.5, 0.5]))
        assert grad.shape in [(1, 2), (2, 1), (2, 1, 1)]

    def test_rmts_jax_derivatives_acceleration(self):
        """Test Axe 3: predict_derivatives accelerated via JAX on GPU/CPU."""
        for kwargs in [
            {"spline_type": "B-Spline", "order": 3, "num_ctrl_pts": 6},
            {"spline_type": "Cubic_Hermite_Spline", "num_elements": 4},
        ]:
            sm = RMTS(
                xlimits=np.array([[0.0, 1.0], [0.0, 1.0]]),
                print_global=False,
                **kwargs,
            )
            sm.set_training_values(self.xt, self.yt)
            sm.train()

            for kx in [0, 1]:
                # Standard CPU analytic derivative
                sm.options["use_jax"] = False
                d_np = sm.predict_derivatives(self.x_eval, kx)

                # Accelerated JAX forward AD (JVP) on GPU/CPU
                sm.options["use_jax"] = True
                d_jax = sm.predict_derivatives(self.x_eval, kx)

                np.testing.assert_allclose(d_jax, d_np, atol=1e-7, rtol=1e-5)

    def test_rmts_jax_solver_training(self):
        """Test Axe 4: Training RMTS with GPU/CPU sparse solver='jax-cg'."""
        for kwargs in [
            {"spline_type": "B-Spline", "order": 3, "num_ctrl_pts": 5},
            {"spline_type": "Cubic_Hermite_Spline", "num_elements": 4},
        ]:
            sm = RMTS(
                xlimits=np.array([[0.0, 1.0], [0.0, 1.0]]),
                solver="jax-cg",
                print_global=False,
                **kwargs,
            )
            sm.set_training_values(self.xt, self.yt)
            sm.train()

            y_pred = sm.predict_values(self.x_eval)
            assert not np.isnan(y_pred).any()
            assert y_pred.shape == (self.x_eval.shape[0], self.yt.shape[1])

    def test_rmts_jax_extrapolation_and_fp32(self):
        """Test Axes 1, 2, 5: lax.cond extrapolation, VRAM caching, and FP32."""
        sm = RMTS(
            spline_type="B-Spline",
            xlimits=np.array([[0.0, 1.0], [0.0, 1.0]]),
            order=3,
            num_ctrl_pts=5,
            extrapolate=True,
            print_global=False,
        )
        sm.set_training_values(self.xt, self.yt)
        sm.train()

        # Points containing both interior and exterior points
        x_mixed = np.array(
            [
                [0.5, 0.5],
                [-0.2, 0.5],
                [1.2, 0.8],
                [0.3, 1.4],
            ]
        )

        # Axe 2: Verify lax.cond extrapolation matches standard evaluation
        y_np = sm.predict_values(x_mixed)
        pred_jax = sm.get_jax_predictor()
        y_jax = pred_jax(x_mixed)
        np.testing.assert_allclose(np.asarray(y_jax), y_np, atol=1e-10)

        # Axe 5: FP32 evaluation on consumer Tensor Cores
        pred_fp32 = sm.get_jax_predictor(dtype=jnp.float32)
        y_fp32 = pred_fp32(x_mixed)
        assert y_fp32.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(y_fp32), y_np, atol=1e-4)

        # Axe 5 via option jax_dtype="float32"
        sm.options["jax_dtype"] = "float32"
        pred_opt = sm.get_jax_predictor()
        y_opt = pred_opt(x_mixed)
        assert y_opt.dtype == jnp.float32

    def test_use_jax_dispatch_and_pickle_after_inference(self):
        import pickle

        for model_cls, kwargs in [
            (KRG, {"corr": "squar_exp", "poly": "linear"}),
            (IDW, {}),
            (RBF, {"d0": [1.0, 1.0]}),
            (PCE, {"degree": 3, "method": "ols"}),
            (SparsePCE, {"degree": [1, 2, 3]}),
            (
                RMTS,
                {"spline_type": "B-Spline", "order": 3, "num_ctrl_pts": 6},
            ),
        ]:
            sm = model_cls(print_global=False, **kwargs)
            if issubclass(model_cls, RMTS):
                sm.options["xlimits"] = np.array([[0.0, 1.0], [0.0, 1.0]])
            sm.set_training_values(self.xt, self.yt)
            sm.train()

            y_cpu = sm.predict_values(self.x_eval)
            dy_cpu = sm.predict_derivatives(self.x_eval, 0)

            sm.options["use_jax"] = True
            y_jax = sm.predict_values(self.x_eval)
            dy_jax = sm.predict_derivatives(self.x_eval, 0)

            assert getattr(sm, "_jax_jit_predict", None) is not None
            assert getattr(sm, "_jax_jit_deriv", None)
            np.testing.assert_allclose(y_jax, y_cpu, atol=1e-10)
            np.testing.assert_allclose(dy_jax, dy_cpu, atol=1e-7)

            # Pickle roundtrip must strip unpicklable JIT closures and still work
            sm_loaded = pickle.loads(pickle.dumps(sm))
            np.testing.assert_allclose(
                sm_loaded.predict_values(self.x_eval), y_cpu, atol=1e-10
            )
            np.testing.assert_allclose(
                sm_loaded.predict_derivatives(self.x_eval, 0), dy_cpu, atol=1e-7
            )

        # Unsupported KRG kernel with use_jax=True must fall back to CPU cleanly
        sm_sin = KRG(corr="squar_sin_exp", use_jax=True, print_global=False)
        sm_sin.set_training_values(self.xt, self.yt)
        sm_sin.train()
        assert sm_sin.predict_values(self.x_eval).shape == (self.x_eval.shape[0], 1)

    def test_missing_jax_fallback_and_kx_validation(self):
        import builtins
        from unittest.mock import patch

        sm = RBF(d0=[1.0, 1.0], print_global=False)
        sm.set_training_values(self.xt, self.yt)
        sm.train()
        y_cpu = sm.predict_values(self.x_eval)
        dy_cpu = sm.predict_derivatives(self.x_eval, 0)

        for bad_kx in (-1, 2, 0.5, True):
            with pytest.raises(ValueError):
                sm.predict_derivatives(self.x_eval, bad_kx)

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "jax" or name.startswith("jax."):
                raise ImportError("mocked missing jax")
            return real_import(name, globals, locals, fromlist, level)

        sm.options["use_jax"] = True
        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.warns(UserWarning, match="JAX is not installed"):
                y_fb = sm.predict_values(self.x_eval)
            dy_fb = sm.predict_derivatives(self.x_eval, 0)
        np.testing.assert_allclose(y_fb, y_cpu, atol=1e-12)
        np.testing.assert_allclose(dy_fb, dy_cpu, atol=1e-12)

    def test_kplsk_mfkplsk_and_sgp_mgp_support(self):
        sm_kplsk = KPLSK(n_comp=1, poly="linear", print_global=False)
        sm_kplsk.set_training_values(self.xt, self.yt)
        sm_kplsk.train()
        np.testing.assert_allclose(
            np.asarray(sm_kplsk.get_jax_predictor()(jnp.asarray(self.x_eval))),
            sm_kplsk.predict_values(self.x_eval),
            atol=1e-10,
        )

        xt_low = np.random.rand(25, 2)
        yt_low = np.sin(xt_low[:, 0:1]) + np.cos(xt_low[:, 1:2])
        xt_high = np.random.rand(10, 2)
        yt_high = 2.0 * (np.sin(xt_high[:, 0:1]) + np.cos(xt_high[:, 1:2])) + 0.1
        sm_mfkplsk = MFKPLSK(n_comp=1, print_global=False)
        sm_mfkplsk.set_training_values(xt_low, yt_low, name=0)
        sm_mfkplsk.set_training_values(xt_high, yt_high)
        sm_mfkplsk.train()
        np.testing.assert_allclose(
            np.asarray(sm_mfkplsk.get_jax_predictor()(jnp.asarray(self.x_eval))),
            sm_mfkplsk.predict_values(self.x_eval),
            atol=1e-10,
        )

        for unsupported_cls in (SGP, MGP):
            sm_u = unsupported_cls(print_global=False)
            with pytest.raises(NotImplementedError):
                sm_u.get_jax_predictor()

    def test_multioutput_qp_mfk_rmts_and_solver_accuracy(self):
        yt_multi = np.hstack([self.yt, 3.0 * self.yt + 1.0])

        # QP multi-output analytic derivatives vs FD
        sm_qp = QP(print_global=False)
        sm_qp.set_training_values(self.xt, yt_multi)
        sm_qp.train()
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

        # KRG multi-output & MFK 1D/2D JAX predictor shapes
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

        # RMTS multi-output, extrapolate=False outside zero derivs,
        # fp32 use_jax, and jax-cg vs dense-lu
        sm_lu = RMTS(
            xlimits=np.array([[0.0, 1.0], [0.0, 1.0]]),
            spline_type="B-Spline",
            order=3,
            num_ctrl_pts=5,
            solver="dense-lu",
            extrapolate=False,
            print_global=False,
        )
        sm_lu.set_training_values(self.xt, yt_multi)
        sm_lu.train()

        sm_cg = RMTS(
            xlimits=np.array([[0.0, 1.0], [0.0, 1.0]]),
            spline_type="B-Spline",
            order=3,
            num_ctrl_pts=5,
            solver="jax-cg",
            extrapolate=False,
            print_global=False,
        )
        sm_cg.set_training_values(self.xt, yt_multi)
        sm_cg.train()
        np.testing.assert_allclose(
            sm_cg.predict_values(self.x_eval),
            sm_lu.predict_values(self.x_eval),
            atol=1e-8,
        )

        # Outside derivatives when extrapolate=False must be 0.0 (CPU & JAX, RMTS B-Spline & Cubic_Hermite_Spline)
        x_out = np.array([[-0.3, 0.5], [1.4, 0.2]])
        sm_rmtc_noext = RMTS(
            xlimits=np.array([[0.0, 1.0], [0.0, 1.0]]),
            spline_type="Cubic_Hermite_Spline",
            num_elements=4,
            extrapolate=False,
            print_global=False,
        )
        sm_rmtc_noext.set_training_values(self.xt, yt_multi)
        sm_rmtc_noext.train()
        for model in [sm_lu, sm_rmtc_noext]:
            for kx in [0, 1]:
                np.testing.assert_allclose(
                    model.predict_derivatives(x_out, kx), 0.0, atol=1e-12
                )
                model.options["use_jax"] = True
                np.testing.assert_allclose(
                    model.predict_derivatives(x_out, kx), 0.0, atol=1e-12
                )
                model.options["use_jax"] = False

        # RMTS use_jax=True + jax_dtype="float32" + cache key switch from float64
        sm_lu.options["use_jax"] = True
        y64 = sm_lu.predict_values(self.x_eval)
        dy64 = sm_lu.predict_derivatives(self.x_eval, 0)
        sm_lu.options["jax_dtype"] = "float32"
        y32 = sm_lu.predict_values(self.x_eval)
        dy32 = sm_lu.predict_derivatives(self.x_eval, 0)
        np.testing.assert_allclose(y32, y64, atol=1e-4)
        np.testing.assert_allclose(dy32, dy64, atol=1e-4)

    @pytest.mark.parametrize("corr", ["matern32", "matern52"])
    def test_matern_second_order_ad_at_training_points(self, corr):
        from smt.surrogate_models.krg_based.krg_jax import _get_matern_kernels

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

        # matern32 has a C2 third-derivative jump (O(eps) FD error), while matern52
        # is C4 with ill-conditioned dual weights here so eps=1e-4 avoids cancellation.
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


def test_jax_operations_do_not_mutate_global_x64_config():
    """SMT JAX predictors and internal JAX solvers/inference must not enable x64 globally."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import warnings
        import jax
        import jax.numpy as jnp
        import numpy as np
        from smt.surrogate_models import IDW, KRG, LS, QP, RBF, RMTS

        assert not jax.config.jax_enable_x64
        rng = np.random.default_rng(0)
        xt = rng.random((20, 2))
        yt = np.sin(xt[:, :1]) + np.cos(xt[:, 1:2])
        x_eval = rng.random((6, 2))

        sm_cg = RMTS(
            xlimits=np.array([[0.0, 1.0], [0.0, 1.0]]),
            solver="jax-cg",
            print_global=False,
        )
        sm_cg.set_training_values(xt, yt)
        sm_cg.train()
        assert not jax.config.jax_enable_x64

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for cls, kw in [
                (KRG, {"use_jax": True}),
                (LS, {"use_jax": True}),
                (QP, {"use_jax": True}),
                (IDW, {"use_jax": True}),
                (RBF, {"d0": [1.0, 1.0], "use_jax": True}),
                (RMTS, {"xlimits": np.array([[0.0, 1.0], [0.0, 1.0]]), "use_jax": True}),
            ]:
                m = cls(print_global=False, **kw)
                m.set_training_values(xt, yt)
                m.train()
                y_j = m.predict_values(x_eval)
                dy_j = m.predict_derivatives(x_eval, 0)
                m.options["use_jax"] = False
                np.testing.assert_allclose(y_j, m.predict_values(x_eval), atol=1e-10)
                np.testing.assert_allclose(
                    dy_j, m.predict_derivatives(x_eval, 0), atol=1e-7
                )
                pred = m.get_jax_predictor()
                _ = jax.jit(pred)(jnp.asarray(x_eval))
            assert not any("float64" in str(item.message) for item in w)
        assert not jax.config.jax_enable_x64
        """
    )
    subprocess.check_call([sys.executable, "-c", code])
