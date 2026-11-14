"""
Tests of the pure-Python RBF / IDW kernels (NumPy and optional Numba engines).

This package is distributed under New BSD license.
"""

import os
import pickle
import unittest
from unittest import mock

import numpy as np

from smt.surrogate_models import IDW, RBF
from smt.surrogate_models import rbf_idw_backend as bk


def _problem(nt=15, nx=3, ne=8, seed=0):
    rng = np.random.RandomState(seed)
    xt = rng.rand(nt, nx)
    x = rng.rand(ne, nx)
    x[1] = xt[4]  # exact hit with a training point
    return xt, x


class TestBackendResolution(unittest.TestCase):
    def test_default_is_numpy(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("USE_NUMBA_JIT", None)
            self.assertEqual(bk._resolve_backend(None), "numpy")

    def test_fallback_without_numba(self):
        with mock.patch.object(bk, "has_numba", return_value=False):
            self.assertEqual(bk._resolve_backend("numba"), "numpy")
            with mock.patch.dict(os.environ, {"USE_NUMBA_JIT": "1"}):
                self.assertEqual(bk._resolve_backend(None), "numpy")

    def test_invalid_backend(self):
        with self.assertRaises(ValueError):
            bk._resolve_backend("cuda")


class TestKernels(unittest.TestCase):
    def _engines(self):
        return ["numpy"] + (["numba"] if bk.has_numba() else [])

    def test_engines_agree(self):
        xt, x = _problem()
        d0 = np.array([0.5, 1.0, 2.0])
        for eng in self._engines():
            for deg in (-1, 0, 1):
                for kx in (None, 0, 2):
                    np.testing.assert_allclose(
                        bk.rbf_jac(x, xt, d0, deg, kx, _backend=eng),
                        bk.rbf_jac(x, xt, d0, deg, kx, _backend="numpy"),
                        rtol=1e-13,
                        atol=1e-13,
                    )
            for kx in (None, 0, 2):
                np.testing.assert_allclose(
                    bk.idw_jac(x, xt, 2.5, kx, _backend=eng),
                    bk.idw_jac(x, xt, 2.5, kx, _backend="numpy"),
                    rtol=1e-13,
                    atol=1e-13,
                )

    def test_derivatives_finite_difference(self):
        xt, x = _problem()
        d0 = np.array([0.5, 1.0, 2.0])
        h = 1e-6
        for kx in range(3):
            xp, xm = x.copy(), x.copy()
            xp[:, kx] += h
            xm[:, kx] -= h
            for deg in (-1, 0, 1):
                fd = (bk.rbf_jac(xp, xt, d0, deg) - bk.rbf_jac(xm, xt, d0, deg)) / (
                    2 * h
                )
                np.testing.assert_allclose(
                    bk.rbf_jac(x, xt, d0, deg, kx), fd, rtol=1e-5, atol=1e-6
                )
            # skip the exact-hit row: IDW weights are not differentiable there
            rows = np.arange(x.shape[0]) != 1
            fd = (bk.idw_jac(xp, xt, 2.5) - bk.idw_jac(xm, xt, 2.5)) / (2 * h)
            np.testing.assert_allclose(
                bk.idw_jac(x, xt, 2.5, kx)[rows], fd[rows], rtol=1e-5, atol=1e-6
            )

    def test_idw_exact_hit(self):
        xt, x = _problem()
        w = bk.idw_jac(x, xt, 2.5)
        np.testing.assert_array_equal(w[1], np.eye(xt.shape[0])[4])
        np.testing.assert_array_equal(bk.idw_jac(x, xt, 2.5, 0)[1], 0.0)
        np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-14)

    def test_idw_duplicate_training_points_first_wins(self):
        xt = np.array([[0.0], [0.5], [0.5], [1.0]])
        w = bk.idw_jac(np.array([[0.5]]), xt, 2.0)
        np.testing.assert_array_equal(w, [[0.0, 1.0, 0.0, 0.0]])

    def test_idw_underflow_nearest_neighbor(self):
        xt = np.array([[0.0, 0.0], [10.0, 10.0]])
        x = np.array([[1e4, 1e4]])
        for eng in self._engines():
            w = bk.idw_jac(x, xt, 100.0, _backend=eng)
            dw = bk.idw_jac(x, xt, 100.0, kx=0, _backend=eng)
            np.testing.assert_array_equal(w, [[0.0, 1.0]])
            np.testing.assert_array_equal(dw, [[0.0, 0.0]])


class TestModels(unittest.TestCase):
    def test_pickle_round_trip(self):
        xt, x = _problem()
        yt = np.sin(xt.sum(axis=1, keepdims=True))
        for sm in (RBF(print_global=False, poly_degree=1), IDW(print_global=False)):
            sm.set_training_values(xt, yt)
            sm.train()
            sm2 = pickle.loads(pickle.dumps(sm))
            np.testing.assert_array_equal(sm.predict_values(x), sm2.predict_values(x))
            np.testing.assert_array_equal(
                sm.predict_derivatives(x, 1), sm2.predict_derivatives(x, 1)
            )

    @unittest.skipUnless(bk.has_numba(), "numba not installed")
    def test_numba_env_switch(self):
        xt, x = _problem()
        yt = np.sin(xt.sum(axis=1, keepdims=True))
        out = {}
        for flag in ("0", "1"):
            with mock.patch.dict(os.environ, {"USE_NUMBA_JIT": flag}):
                sm = RBF(print_global=False, d0=0.8, poly_degree=1)
                sm.set_training_values(xt, yt)
                sm.train()
                out[flag] = sm.predict_values(x)
        np.testing.assert_allclose(out["0"], out["1"], rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
