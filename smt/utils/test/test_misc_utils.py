"""
Author: P. Saves

This package is distributed under New BSD license.
"""

import unittest

import numpy as np

from smt.problems import Sphere
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG
from smt.utils.misc import (
    compute_pva,
    compute_q2,
    compute_rmse,
    standardization,
)

try:
    import jax  # noqa: F401

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class TestMisc(unittest.TestCase):
    def test_standardization(self):
        X = np.array([[0], [1], [2]])
        y = np.array([[1], [3], [5]])
        X2, y2, X_offset, y_mean, X_scale, y_std = standardization(
            np.copy(X), np.copy(y)
        )

        self.assertTrue(np.array_equal(X2.T, np.array([[-1, 0, 1]])))
        self.assertTrue(np.array_equal(y2.T, np.array([[-1, 0, 1]])))

        self.assertTrue(np.array_equal(X_offset, np.array([1])))
        self.assertTrue(np.array_equal(y_mean, np.array([3])))

        self.assertTrue(np.array_equal(X_scale, np.array([1])))
        self.assertTrue(np.array_equal(y_std, np.array([2])))

    def prepare_tests_errors(self):
        ndim = 2
        fun = Sphere(ndim=ndim)

        sampling = LHS(xlimits=fun.xlimits, criterion="ese", seed=42)
        xt = sampling(20)
        yt = fun(xt)
        xe = sampling(100)
        ye = fun(xe)
        return xt, yt, xe, ye

    def test_pva_error(self):
        xt, yt, xe, ye = self.prepare_tests_errors()
        sm = KRG(print_global=False, n_start=25, seed=42)
        sm.set_training_values(xt, yt)
        sm.train()

        pva = compute_pva(sm, xe, ye)
        self.assertAlmostEqual(pva, 0.05, delta=1e-1)

    def test_rmse_error(self):
        xt, yt, xe, ye = self.prepare_tests_errors()
        sm = KRG(print_global=False, seed=42)
        sm.set_training_values(xt, yt)
        sm.train()

        rmse = compute_rmse(sm, xe, ye)
        self.assertAlmostEqual(rmse, 0.0, delta=1e-2)

    def test_q2_error(self):
        xt, yt, xe, ye = self.prepare_tests_errors()
        sm = KRG(print_global=False, seed=42)
        sm.set_training_values(xt, yt)
        sm.train()

        q2 = compute_q2(sm, xe, ye)
        self.assertAlmostEqual(q2, 1.0, delta=1e-2)

    def test_silence_utilities(self):
        """Covers smt/utils/silence.py Silence and Silence2 classes."""
        from smt.utils.silence import Silence, Silence2

        # Test Silence (basic sys.stdout override)
        with Silence():
            print("Silenced")

        # Test Silence2 (low-level FD suppression)
        # Using default devnull avoids Windows file-locking issues.
        with Silence2():
            import sys

            print("FD Stdout Silenced")
            sys.stderr.write("FD Stderr Silenced\n")

    def test_checks(self):
        """Covers smt/utils/checks.py."""
        from smt.utils.checks import ensure_2d_array, check_support, check_nx

        # 1. ensure_2d_array accepts lists, 1D arrays, complex arrays, and numeric objects
        arr_list = ensure_2d_array([1, 2], "list")
        self.assertEqual(arr_list.shape, (2, 1))

        arr_c = ensure_2d_array(np.array([[1.0 + 2.0j]]), "c")
        self.assertEqual(arr_c.shape, (1, 1))
        self.assertEqual(arr_c.dtype, np.complex128)

        arr_obj = ensure_2d_array(np.array([[1.0], [2.0]], dtype=object), "obj")
        self.assertEqual(arr_obj.shape, (2, 1))
        self.assertEqual(arr_obj.dtype, np.float64)

        with self.assertRaises(ValueError):
            ensure_2d_array("invalid", "str")

        with self.assertRaises(ValueError):
            ensure_2d_array(np.array(["a", "b"]), "str_arr")

        arr1d = np.array([1, 2])
        arr2d = ensure_2d_array(arr1d, "arr1d")
        self.assertEqual(arr2d.shape, (2, 1))

        from smt.utils.checks import ensure_float64_array, is_arraylike

        self.assertTrue(is_arraylike(np.array([1, 2])))
        self.assertFalse(is_arraylike("string"))
        self.assertFalse(is_arraylike(1.0))
        self.assertFalse(is_arraylike(np.float64(1.0)))

        f32 = np.array([1.0, 2.0], dtype=np.float32)
        f64 = ensure_float64_array(f32)
        self.assertEqual(f64.dtype, np.float64)
        self.assertTrue(f64.flags.writeable)

        # Rank > 2 (though hard with NumPy atleast_2d logic, we check the error branch)
        # atleast_2d(rank3.T).T usually results in rank 3 or reshaped rank 2
        # But we hit the ValueError if shape len != 2

        # 2. check_support
        from smt.surrogate_models import IDW

        sm = IDW()  # IDW supports 'derivatives' but not 'variances'
        check_support(sm, "derivatives")  # Should pass
        with self.assertRaises(NotImplementedError):
            check_support(sm, "variances")
        with self.assertRaises(NotImplementedError):
            check_support(sm, "derivatives", fail=True)

        # 3. check_nx
        sm.nx = 2
        with self.assertRaises(ValueError):
            check_nx(sm, np.zeros((10, 3)))  # expects 2

        sm.nx = 1
        with self.assertRaisesRegex(ValueError, "x should have shape"):
            check_nx(sm, np.zeros((10, 2)))  # expects 1

    def test_arraylike_inputs(self):
        """Issue #827: generic array-likes (__array__ / __array_interface__)."""
        from smt.utils.checks import ensure_2d_array
        from smt.utils.options_dictionary import OptionsDictionary

        class ArrayProto:
            def __init__(self, data):
                self._data = np.asarray(data)
                self.ndim = self._data.ndim

            def __array__(self, dtype=None, copy=None):
                arr = self._data if dtype is None else self._data.astype(dtype)
                return np.array(arr, copy=copy) if copy is not None else arr

        class InterfaceProto:
            def __init__(self, data):
                self._data = np.asarray(data)
                self.__array_interface__ = self._data.__array_interface__
                self.ndim = self._data.ndim

        class MinimalArrayProto:
            """Exposes only __array__ with no .ndim, .shape or .item attributes."""

            def __init__(self, data):
                self._data = np.asarray(data)

            def __array__(self, dtype=None, copy=None):
                arr = self._data if dtype is None else self._data.astype(dtype)
                return np.array(arr, copy=copy) if copy is not None else arr

        a1 = ensure_2d_array(ArrayProto([1.0, 2.0, 3.0]), "a1")
        self.assertIsInstance(a1, np.ndarray)
        self.assertEqual(a1.shape, (3, 1))

        a2 = ensure_2d_array(InterfaceProto([[1.0, 2.0], [3.0, 4.0]]), "a2")
        self.assertIsInstance(a2, np.ndarray)
        self.assertEqual(a2.shape, (2, 2))

        a3 = ensure_2d_array(MinimalArrayProto([1.0, 2.0, 3.0]), "a3")
        self.assertIsInstance(a3, np.ndarray)
        self.assertEqual(a3.shape, (3, 1))

        # Scalars and 1D sequences are accepted and converted to 2D
        s1 = ensure_2d_array(np.float64(1.0), "scalar")
        self.assertEqual(s1.shape, (1, 1))

        with self.assertRaises(ValueError):
            ensure_2d_array(ArrayProto(np.zeros((2, 2, 2))), "rank3")

        opts = OptionsDictionary()
        opts.declare("xlimits", types=np.ndarray)
        opts["xlimits"] = ArrayProto([[0.0, 1.0]])
        self.assertIsInstance(opts["xlimits"], np.ndarray)
        self.assertEqual(opts["xlimits"].shape, (1, 2))

        opts["xlimits"] = MinimalArrayProto([[0.0, 1.0]])
        self.assertIsInstance(opts["xlimits"], np.ndarray)
        self.assertEqual(opts["xlimits"].shape, (1, 2))

        # Test MinimalArrayProto on KRG set_training_values & predict_values
        krg_proto = KRG(theta0=[1e-2], print_global=False)
        krg_proto.set_training_values(
            MinimalArrayProto([[0.0], [0.5], [1.0]]),
            MinimalArrayProto([[0.0], [0.25], [1.0]]),
        )
        krg_proto.train()
        self.assertEqual(
            krg_proto.predict_values(MinimalArrayProto([[0.2], [0.8]])).shape, (2, 1)
        )

        # Prediction must never alias (and thus risk mutating) the caller's array
        xq_np = np.array([[0.2], [0.8]])
        x_pre = krg_proto._pre_predict(xq_np)
        self.assertFalse(np.shares_memory(x_pre, xq_np))

        # Non-numerical objects are rejected
        with self.assertRaisesRegex(ValueError, "array-like"):
            ensure_2d_array({"a": 1}, "xt")
        with self.assertRaisesRegex(ValueError, "array-like"):
            ensure_2d_array("hello", "xt")

    @unittest.skipUnless(HAS_JAX, "JAX not installed")
    def test_jax_inputs(self):
        """Issue #827: JAX arrays are accepted everywhere a NumPy array is."""
        import jax
        import jax.numpy as jnp

        from smt.applications import MFK
        from smt.applications.mixed_integer import MixedIntegerSurrogateModel
        from smt.design_space import DesignSpace, FloatVariable, IntegerVariable
        from smt.surrogate_models import LS, QP

        x = jnp.linspace(0.0, 1.0, 5)
        y = jnp.power(x, 2.0)

        model = KRG(theta0=[1e-2], print_global=False)
        model.set_training_values(x, y)
        model.train()
        xq = jnp.linspace(0.0, 1.0, 6)
        yq = model.predict_values(xq)
        self.assertIsInstance(yq, np.ndarray)
        self.assertEqual(yq.shape, (6, 1))
        self.assertLess(np.max(np.abs(yq[:, 0] - np.asarray(xq) ** 2)), 5e-2)
        self.assertEqual(model.predict_variances(xq).shape, (6, 1))
        self.assertEqual(model.predict_derivatives(xq, 0).shape, (6, 1))
        self.assertEqual(model.predict_variance_derivatives(xq, 0).shape, (6, 1))

        # Lists and scalars across all prediction methods return identical results
        y_list = model.predict_values([[0.5]])
        y_scalar = model.predict_values(0.5)
        y_np_scalar = model.predict_values(np.float64(0.5))
        self.assertEqual(y_list.shape, (1, 1))
        np.testing.assert_allclose(y_list, y_scalar)
        np.testing.assert_allclose(y_list, y_np_scalar)

        v_list = model.predict_variances([[0.5]])
        v_scalar = model.predict_variances(0.5)
        v_np_scalar = model.predict_variances(np.float64(0.5))
        np.testing.assert_allclose(v_list, v_scalar)
        np.testing.assert_allclose(v_list, v_np_scalar)

        d_list = model.predict_derivatives([[0.5]], 0)
        d_scalar = model.predict_derivatives(0.5, 0)
        d_np_scalar = model.predict_derivatives(np.float64(0.5), 0)
        np.testing.assert_allclose(d_list, d_scalar)
        np.testing.assert_allclose(d_list, d_np_scalar)

        # 1D row-vector convenience path of predict_variance_gradient
        self.assertEqual(
            model.predict_variance_gradient(jnp.array([0.3])).shape,
            model.predict_variance_gradient(np.array([0.3])).shape,
        )
        model.update_training_values(y + 1.0)

        qp = QP(print_global=False)
        qp.set_training_values(x, y)
        qp.train()
        self.assertEqual(qp.predict_values(xq).shape, (6, 1))

        ls = LS(print_global=False)
        ls.set_training_values(x, y)
        ls.train()
        self.assertEqual(ls.predict_values(xq).shape, (6, 1))

        mfk = MFK(theta0=[1e-2], print_global=False)
        mfk.set_training_values(x, y)
        mfk.set_training_values(x, 0.9 * y, name=0)
        mfk.train()
        self.assertEqual(mfk.predict_values(xq).shape, (6, 1))

        sampling = LHS(xlimits=jnp.array([[0.0, 1.0], [-1.0, 1.0]]), seed=0)
        self.assertEqual(sampling(4).shape, (4, 2))

        # GEKPLS with training derivatives passed as JAX arrays
        from smt.surrogate_models import GEKPLS

        x_gek = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        y_gek = jnp.array([[0.0], [0.5], [2.0], [1.0], [1.0]])
        dydx0 = jnp.array([[0.0], [1.0], [2.0], [0.0], [1.0]])
        dydx1 = jnp.array([[0.0], [1.0], [2.0], [1.0], [0.0]])
        gekpls = GEKPLS(theta0=[1e-2, 1e-2], n_comp=2, print_global=False)
        gekpls.set_training_values(x_gek, y_gek)
        gekpls.set_training_derivatives(x_gek, dydx0, 0)
        gekpls.set_training_derivatives(x_gek, dydx1, 1)
        gekpls.train()
        self.assertEqual(gekpls.predict_values(jnp.array([[0.2, 0.3]])).shape, (1, 1))

        # JAX Tracers inside jit or grad raise a clean TypeError
        from smt.utils.checks import _is_jax_tracer

        self.assertFalse(_is_jax_tracer(jnp.array([1.0])))
        self.assertFalse(_is_jax_tracer(np.array([1.0])))
        self.assertFalse(_is_jax_tracer(1.0))

        @jax.jit
        def tracer_test(x_tr):
            return model.predict_values(x_tr)

        with self.assertRaisesRegex(TypeError, "Cannot pass a JAX Tracer"):
            tracer_test(jnp.array([[0.2]]))

        # JAX arrays in OptionsDictionary
        from smt.utils.options_dictionary import OptionsDictionary

        opts_jax = OptionsDictionary()
        opts_jax.declare("theta0", types=(list, np.ndarray))
        opts_jax["theta0"] = jnp.array([1e-2, 2e-2])
        self.assertIsInstance(opts_jax["theta0"], np.ndarray)
        self.assertEqual(opts_jax["theta0"].dtype, np.float64)
        self.assertTrue(opts_jax["theta0"].flags.writeable)

        # MixedIntegerSurrogateModel with float64 JAX input and no in-place mutation
        jax.config.update("jax_enable_x64", True)
        ds = DesignSpace([FloatVariable(0.0, 1.0), IntegerVariable(0, 2)])
        mi_sm = MixedIntegerSurrogateModel(
            design_space=ds, surrogate=LS(print_global=False)
        )
        mi_sm.set_training_values(
            np.array([[0.0, 0.0], [1.0, 2.0]]),
            np.array([[0.0], [1.0]]),
        )
        mi_sm.train()
        x_jax = jnp.array([[0.4, 1.6]], dtype=jnp.float64)
        y_mi = mi_sm.predict_values(x_jax)
        self.assertEqual(y_mi.shape, (1, 1))

        x_np = np.array([[0.4, 1.6]])
        mi_sm.predict_values(x_np)
        np.testing.assert_array_equal(x_np, np.array([[0.4, 1.6]]))

        from smt.surrogate_models import IDW, RBF, RMTB, RMTC

        rmtb = RMTB(xlimits=jnp.array([[0.0, 1.0]]), num_ctrl_pts=8, print_global=False)
        rmtb.set_training_values(x, y)
        rmtb.train()
        self.assertEqual(rmtb.predict_values(xq).shape, (6, 1))

        rmtc = RMTC(xlimits=jnp.array([[0.0, 1.0]]), num_elements=4, print_global=False)
        rmtc.set_training_values(x, y)
        rmtc.train()
        self.assertEqual(rmtc.predict_values(xq).shape, (6, 1))

        idw = IDW(p=2, print_global=False)
        idw.set_training_values(x, y)
        idw.train()
        self.assertEqual(idw.predict_values(xq).shape, (6, 1))

        rbf = RBF(d0=[1.0], print_global=False)
        rbf.set_training_values(x, y)
        rbf.train()
        self.assertEqual(rbf.predict_values(xq).shape, (6, 1))


if __name__ == "__main__":
    unittest.main()
