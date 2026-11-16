"""
Unit tests for pure-Python RMTS/RMTB/RMTC backends (NumPy, optional Numba JIT,
optional JAX AD) and Issues #573, #827, #846.
"""

import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager

import numpy as np
import smt

from smt.surrogate_models import RMTS
from smt.surrogate_models.rmts_backend import (
    PureRMTB,
    PureRMTC,
    _enable_x64,
    _opt_func_jax,
    _opt_grad_jax,
    _opt_hess_jax,
    _prepare_jax_opt_cache,
    _resolve_backend,
    compute_ext_dist,
    compute_knot_vector_uniform,
    compute_quadrature_points,
    has_jax,
    has_numba,
    predict_rmtb_jax,
    predict_rmtc_jax,
)
from smt.utils.silence import Silence


def _available_engines():
    engines = ["numpy"]
    if has_numba():
        engines.append("numba")
    if has_jax():
        engines.append("jax")
    return engines


@contextmanager
def _temp_env(**updates):
    old = {k: os.environ.get(k) for k in updates}
    try:
        for k, v in updates.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _smooth_fn(x):
    if x.shape[1] == 1:
        return np.sin(np.pi * x[:, :1]) + 0.25 * x[:, :1] ** 2
    if x.shape[1] == 2:
        return np.sin(np.pi * x[:, 0:1]) * np.cos(0.5 * np.pi * x[:, 1:2]) + 0.1 * (
            x[:, 0:1] ** 2 - x[:, 1:2] ** 2
        )
    return (
        np.sin(np.pi * x[:, 0:1]) * np.cos(0.5 * np.pi * x[:, 1:2])
        + 0.2 * x[:, 2:3] ** 2
    )


class TestRMTSBackend(unittest.TestCase):
    def test_lazy_imports_and_no_public_backend_option(self):
        # RMTS must not expose a public 'backend' option
        sm_b = RMTS(
            spline_type="B-Spline",
            xlimits=np.array([[0.0, 1.0]]),
            print_global=False,
        )
        sm_c = RMTS(
            spline_type="Cubic_Hermite_Spline",
            xlimits=np.array([[0.0, 1.0]]),
            print_global=False,
        )
        self.assertFalse(sm_b.options.is_declared("backend"))
        self.assertFalse(sm_c.options.is_declared("backend"))

        # Importing SMT and training default RMTS must not import jax or numba
        repo_root = os.path.dirname(os.path.dirname(smt.__file__))
        env = os.environ.copy()
        env["USE_JAX"] = "0"
        env["USE_NUMBA_JIT"] = "0"
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else repo_root
        )
        code = (
            "import sys, numpy as np\n"
            "from smt.surrogate_models import RMTS\n"
            "assert 'jax' not in sys.modules, 'jax imported eagerly'\n"
            "assert 'numba' not in sys.modules, 'numba imported eagerly'\n"
            "x = np.linspace(0.0, 1.0, 15).reshape(-1, 1)\n"
            "y = np.sin(np.pi * x)\n"
            "for st in ('B-Spline', 'Cubic_Hermite_Spline'):\n"
            "    m = RMTS(spline_type=st, xlimits=np.array([[0.0, 1.0]]), print_global=False)\n"
            "    m.set_training_values(x, y)\n"
            "    m.train()\n"
            "    m.predict_values(x)\n"
            "assert 'jax' not in sys.modules, 'jax imported during default numpy run'\n"
            "assert 'numba' not in sys.modules, 'numba imported during default numpy run'\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )

    def test_geometry_helpers(self):
        knots = compute_knot_vector_uniform(4, 10)
        self.assertEqual(knots.shape, (14,))
        self.assertEqual(knots.dtype, np.float64)
        self.assertAlmostEqual(knots[3], 0.0)
        self.assertAlmostEqual(knots[-4], 1.0)

        lower = np.array([-1.0, 2.0])
        upper = np.array([1.0, 6.0])
        x = np.array([[-1.5, 4.0], [0.0, 1.0], [2.0, 7.5], [0.5, 3.0]])
        dx = compute_ext_dist(x, lower, upper, nterm=3)
        expected_1 = np.array([[-0.5, 0.0], [0.0, -1.0], [1.0, 1.5], [0.0, 0.0]])
        np.testing.assert_allclose(dx, np.repeat(expected_1, 3, axis=0), atol=1e-15)
        self.assertEqual(compute_ext_dist(np.empty((0, 2)), lower, upper).shape, (0, 2))

        qp = compute_quadrature_points(np.array([2, 3]), lower, upper)
        self.assertEqual(qp.shape, (6, 2))
        np.testing.assert_allclose(qp[:, 0], [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5])

    def test_pure_rmtb_cross_engine_orders_and_edges(self):
        engines = _available_engines()
        rng = np.random.default_rng(7)

        # Orders 1..6 across 1D and 2D + heterogeneous orders + edge cases
        cases = [
            ([1], [5], 12),
            ([2], [8], 15),
            ([3], [10], 15),
            ([4], [12], 15),
            ([5], [14], 15),
            ([6], [16], 15),
            ([2, 4], [7, 10], 20),
            ([4, 4, 3], [8, 7, 6], 18),
        ]
        for orders, ncps, npts in cases:
            nx = len(orders)
            lower = np.zeros(nx)
            upper = np.ones(nx)
            pts = rng.uniform(0.0, 1.0, size=(npts, nx))
            # Inject exact boundaries and out-of-range points to verify clamping
            pts[0, :] = 0.0
            pts[1, :] = 1.0
            pts[2, :] = -0.15
            pts[3, :] = 1.15

            deriv_pairs = [(-1, -1), (0, -1), (0, 0)]
            if nx >= 2:
                deriv_pairs += [(1, -1), (0, 1), (1, 1)]

            for ix1, ix2 in deriv_pairs:
                ref_bk = PureRMTB(nx, lower, upper, orders, ncps, _backend="numpy")
                d_ref, r_ref, c_ref = ref_bk.compute_jac(ix1, ix2, pts)
                self.assertEqual(d_ref.dtype, np.float64)
                self.assertEqual(r_ref.dtype, np.int32)
                self.assertEqual(c_ref.dtype, np.int32)
                self.assertTrue(np.all(np.isfinite(d_ref)))

                for eng in engines[1:]:
                    bk = PureRMTB(nx, lower, upper, orders, ncps, _backend=eng)
                    d_eng, r_eng, c_eng = bk.compute_jac(ix1, ix2, pts)
                    np.testing.assert_array_equal(r_eng, r_ref)
                    np.testing.assert_array_equal(c_eng, c_ref)
                    np.testing.assert_allclose(
                        d_eng,
                        d_ref,
                        rtol=0.0,
                        atol=1e-12,
                        err_msg=f"RMTB {eng} orders={orders} deriv=({ix1},{ix2})",
                    )

            # Empty and 1D single-point shapes
            for eng in engines:
                bk = PureRMTB(nx, lower, upper, orders, ncps, _backend=eng)
                d0, r0, c0 = bk.compute_jac(-1, -1, np.empty((0, nx)))
                self.assertEqual((len(d0), len(r0), len(c0)), (0, 0, 0))
                d1, r1, c1 = bk.compute_jac(-1, -1, pts[0])
                self.assertEqual(d1.shape[0], int(np.prod(orders)))

    def test_pure_rmtc_cross_engine(self):
        engines = _available_engines()
        rng = np.random.default_rng(11)

        for nx, nelems in [(1, [6]), (2, [4, 5]), (3, [3, 3, 2])]:
            lower = np.full(nx, -2.0)
            upper = np.full(nx, 3.0)
            nterms = [4] * nx
            bk_np = PureRMTC(nx, lower, upper, nelems, nterms, _backend="numpy")

            c2n = bk_np.compute_coeff2nodal()
            self.assertEqual(c2n.shape, (4**nx, 4**nx))
            u2e_d, u2e_r, u2e_c = bk_np.compute_uniq2elem()
            self.assertEqual(u2e_d.shape[0], bk_np.nelem * bk_np.nterm)
            fb_d, _, _ = bk_np.compute_full_from_block(np.eye(bk_np.nterm))
            self.assertEqual(fb_d.shape[0], bk_np.nelem * bk_np.nterm**2)

            pts = rng.uniform(-2.0, 3.0, size=(16, nx))
            pts[0, :] = -2.0
            pts[1, :] = 3.0
            pts[2, :] = -2.5
            pts[3, :] = 3.4

            deriv_pairs = [(-1, -1), (0, -1), (0, 0)]
            if nx >= 2:
                deriv_pairs += [(1, -1), (0, 1)]

            for ix1, ix2 in deriv_pairs:
                d_ref, r_ref, c_ref = bk_np.compute_jac(ix1, ix2, pts)
                self.assertEqual(d_ref.dtype, np.float64)
                self.assertEqual(r_ref.dtype, np.int32)
                self.assertEqual(c_ref.dtype, np.int32)
                for eng in engines[1:]:
                    bk = PureRMTC(nx, lower, upper, nelems, nterms, _backend=eng)
                    d_eng, r_eng, c_eng = bk.compute_jac(ix1, ix2, pts)
                    np.testing.assert_array_equal(r_eng, r_ref)
                    np.testing.assert_array_equal(c_eng, c_ref)
                    np.testing.assert_allclose(
                        d_eng,
                        d_ref,
                        rtol=0.0,
                        atol=1e-12,
                        err_msg=f"RMTC {eng} nx={nx} deriv=({ix1},{ix2})",
                    )

    def test_issue_573_boundaries_and_low_orders_all_engines(self):
        rng = np.random.default_rng(12345)
        xt = rng.uniform(-1.0, 1.0, size=(25, 2))
        xt[0, :] = [-1.0, -1.0]
        xt[1, :] = [1.0, 1.0]
        yt = np.sin(np.pi * xt[:, 0:1]) * np.sin(np.pi * xt[:, 1:2])
        xlimits = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        xq = np.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.25, -0.5]])

        engines = _available_engines()
        for eng in engines:
            nb = "1" if eng == "numba" else "0"
            jx = "1" if eng == "jax" else "0"
            with _temp_env(USE_NUMBA_JIT=nb, USE_JAX=jx):
                for order in (1, 2, 4):
                    sm = RMTS(
                        spline_type="B-Spline",
                        xlimits=xlimits,
                        order=order,
                        num_ctrl_pts=8,
                        solver="lu",
                        approx_order=2,
                        print_global=False,
                    )
                    sm.set_training_values(xt, yt)
                    with Silence():
                        sm.train()
                    yq = sm.predict_values(xq)
                    self.assertTrue(np.all(np.isfinite(yq)))

    def test_end_to_end_cross_engine_equivalence(self):
        engines = _available_engines()
        rng = np.random.default_rng(2026)

        for ndim, rmtb_opts, rmtc_opts in [
            (1, {"order": 4, "num_ctrl_pts": 15}, {"num_elements": 8}),
            (2, {"order": 4, "num_ctrl_pts": 10}, {"num_elements": 5}),
        ]:
            xlimits = np.tile([-1.5, 2.0], (ndim, 1))
            xt = rng.uniform(-1.5, 2.0, size=(35 * ndim, ndim))
            yt = _smooth_fn(xt)
            xq = rng.uniform(-1.5, 2.0, size=(12, ndim))
            xq_ext = np.vstack([xq, xq + 2.5, xq - 2.5])

            for st, opts in [
                ("B-Spline", rmtb_opts),
                ("Cubic_Hermite_Spline", rmtc_opts),
            ]:
                ref = {}
                for approx_order in (2, 4):
                    for eng in engines:
                        nb = "1" if eng == "numba" else "0"
                        jx = "1" if eng == "jax" else "0"
                        with _temp_env(USE_NUMBA_JIT=nb, USE_JAX=jx):
                            sm = RMTS(
                                spline_type=st,
                                xlimits=xlimits,
                                approx_order=approx_order,
                                solver="lu",
                                derivative_solver="lu",
                                extrapolate=True,
                                print_global=False,
                                **opts,
                            )
                            sm.set_training_values(xt, yt)
                            with Silence():
                                sm.train()
                            yq = sm.predict_values(xq_ext)
                            dy0 = sm.predict_derivatives(xq_ext, 0)
                            dy_dyt = sm.predict_output_derivatives(xq)[None]
                            if eng == "numpy":
                                ref[approx_order] = (yq, dy0, dy_dyt)
                            else:
                                y_r, dy_r, dyt_r = ref[approx_order]
                                np.testing.assert_allclose(yq, y_r, atol=1e-11)
                                np.testing.assert_allclose(dy0, dy_r, atol=1e-11)
                                np.testing.assert_allclose(dy_dyt, dyt_r, atol=1e-9)

    @unittest.skipIf(not has_jax(), "JAX is not installed")
    def test_jax_ad_solver_and_differentiable_predictors(self):
        import jax
        import jax.numpy as jnp

        rng = np.random.default_rng(99)
        xlimits = np.array([[-1.0, 2.0], [0.0, 3.0]])
        xt = np.column_stack([rng.uniform(-1.0, 2.0, 60), rng.uniform(0.0, 3.0, 60)])
        yt = np.column_stack([_smooth_fn(xt)[:, 0], xt[:, 0] ** 2 - xt[:, 1]])
        xq = np.array([[-0.3, 1.2], [0.7, 2.4], [-1.4, -0.5], [2.4, 3.6]])

        # 0. Verify internal JAX usage leaves process-wide JAX config untouched
        repo_root = os.path.dirname(os.path.dirname(smt.__file__))
        env = os.environ.copy()
        env["USE_JAX"] = "1"
        env["USE_NUMBA_JIT"] = "0"
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else repo_root
        )
        probe_code = (
            "import jax, numpy as np\n"
            "x64_before = bool(jax.config.jax_enable_x64)\n"
            "cache_before = jax.config.jax_compilation_cache_dir\n"
            "min_t_before = jax.config.jax_persistent_cache_min_compile_time_secs\n"
            "min_s_before = jax.config.jax_persistent_cache_min_entry_size_bytes\n"
            "from smt.surrogate_models import RMTS\n"
            "from smt.surrogate_models.rmts_backend import has_jax\n"
            "assert has_jax()\n"
            "x = np.linspace(0.0, 1.0, 15).reshape(-1, 1)\n"
            "y = np.sin(np.pi * x)\n"
            "for st in ('B-Spline', 'Cubic_Hermite_Spline'):\n"
            "    m = RMTS(spline_type=st, xlimits=np.array([[0.0, 1.0]]), solver='lu', print_global=False)\n"
            "    m.set_training_values(x, y)\n"
            "    m.train()\n"
            "    assert m.predict_values(x).shape == y.shape\n"
            "assert bool(jax.config.jax_enable_x64) == x64_before\n"
            "assert jax.config.jax_compilation_cache_dir == cache_before\n"
            "assert jax.config.jax_persistent_cache_min_compile_time_secs == min_t_before\n"
            "assert jax.config.jax_persistent_cache_min_entry_size_bytes == min_s_before\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe_code],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )

        # 1. Verify _opt_func/grad/hess JAX AD match NumPy for p in (0, 1, 2, 3, 4)
        with _temp_env(USE_JAX="0", USE_NUMBA_JIT="0"):
            sm_opt = RMTS(
                spline_type="B-Spline",
                xlimits=xlimits,
                order=4,
                num_ctrl_pts=8,
                solver="lu",
                print_global=False,
            )
            sm_opt.set_training_values(xt, yt[:, :1])
            with Silence():
                sm_opt.train()

        cache = _prepare_jax_opt_cache(sm_opt.full_hess, sm_opt.full_jac_dict)
        yt_d = sm_opt._get_yt_dict(0)
        sol_probe = sm_opt.sol[:, 0] + 0.05 * rng.normal(size=sm_opt.num["dof"])
        for p in (0, 1, 2, 3, 4):
            f_np = sm_opt._opt_func(sol_probe, p, yt_d)
            g_np = sm_opt._opt_grad(sol_probe, p, yt_d)
            h_np = sm_opt._opt_hess(sol_probe, p, yt_d).toarray()
            f_jx = _opt_func_jax(sol_probe, p, yt_d, cache)
            g_jx = _opt_grad_jax(sol_probe, p, yt_d, cache)
            h_jx = _opt_hess_jax(
                sol_probe, p, yt_d, cache, sm_opt.full_hess, sm_opt.full_jac_dict
            ).toarray()
            self.assertAlmostEqual(f_jx, f_np, places=11)
            np.testing.assert_allclose(g_jx, g_np, atol=1e-11)
            np.testing.assert_allclose(h_jx, h_np, atol=1e-11)

        # 2. Verify predict_rmtb_jax & predict_rmtc_jax values, extrapolate & AD
        with _temp_env(USE_JAX="1"):
            for st, opts in [
                ("B-Spline", {"order": 4, "num_ctrl_pts": 9}),
                ("Cubic_Hermite_Spline", {"num_elements": 5}),
            ]:
                for extrap in (False, True):
                    sm = RMTS(
                        spline_type=st,
                        xlimits=xlimits,
                        solver="lu",
                        extrapolate=extrap,
                        print_global=False,
                        **opts,
                    )
                    sm.set_training_values(xt, yt)
                    with Silence():
                        sm.train()

                    def pred_fn(x_arr, c_arr=sm.sol_coeff, model=sm, is_extrap=extrap):
                        if model.options["spline_type"] == "B-Spline":
                            return predict_rmtb_jax(
                                x_arr,
                                c_arr,
                                xlimits[:, 0],
                                xlimits[:, 1],
                                model.num["order_list"],
                                model.num["ctrl_list"],
                                knots_list=model.knots_list,
                                extrapolate=is_extrap,
                            )
                        return predict_rmtc_jax(
                            x_arr,
                            c_arr,
                            xlimits[:, 0],
                            xlimits[:, 1],
                            model.num["elem_list"],
                            model.num["term_list"],
                            extrapolate=is_extrap,
                        )

                    # Works cleanly under caller's default JAX precision (e.g. float32 + outer jit/jacfwd)
                    y_default = jax.jit(pred_fn)(jnp.asarray(xq))
                    j_default = jax.jit(jax.jacfwd(pred_fn))(jnp.asarray(xq[0]))
                    self.assertEqual(y_default.shape, (xq.shape[0], 2))
                    self.assertEqual(j_default.shape, (2, 2))

                    # High-precision verification inside caller-scoped x64 context
                    with _enable_x64(jax):
                        y_jax = np.asarray(pred_fn(jnp.asarray(xq)))
                        np.testing.assert_allclose(
                            y_jax, sm.predict_values(xq), atol=1e-11
                        )

                        # Single-point (nx,) input & AD wrt x for output 0 and 1
                        pts_check = xq if extrap else xq[:2]
                        jac_fn = jax.jacfwd(pred_fn)
                        for i in range(pts_check.shape[0]):
                            xi = jnp.asarray(pts_check[i])
                            yi = np.asarray(pred_fn(xi))
                            np.testing.assert_allclose(
                                yi,
                                sm.predict_values(pts_check[i : i + 1])[0],
                                atol=1e-11,
                            )
                            ji = np.asarray(jac_fn(xi))
                            for kx in range(2):
                                np.testing.assert_allclose(
                                    ji[:, kx],
                                    sm.predict_derivatives(pts_check[i : i + 1], kx)[0],
                                    atol=1e-10,
                                )

                # Differentiability wrt sol_coeff
                with _enable_x64(jax):
                    g_c = np.asarray(
                        jax.grad(lambda c: pred_fn(jnp.asarray(xq[0]), c)[0])(
                            jnp.asarray(sm.sol_coeff)
                        )
                    )
                    eps = 1e-6
                    e0 = np.zeros_like(sm.sol_coeff)
                    idx = sm.sol_coeff.shape[0] // 2
                    e0[idx, 0] = eps
                    fd = (
                        float(pred_fn(jnp.asarray(xq[0]), sm.sol_coeff + e0)[0])
                        - float(pred_fn(jnp.asarray(xq[0]), sm.sol_coeff - e0)[0])
                    ) / (2.0 * eps)
                    self.assertAlmostEqual(float(g_c[idx, 0]), fd, places=8)

    def test_issue_846_get_bspline_and_to_scipy(self):
        rng = np.random.default_rng(846)
        sm_untrained = RMTS(
            spline_type="B-Spline", xlimits=np.array([[0.0, 1.0]]), print_global=False
        )
        with self.assertRaises(RuntimeError):
            sm_untrained.get_bspline()
        with self.assertRaises(RuntimeError):
            sm_untrained.to_scipy()

        for ndim in (1, 2, 3):
            xlimits = np.column_stack(
                [np.full(ndim, -1.5), np.linspace(1.0, 3.0, ndim)]
            )
            xt = np.column_stack(
                [
                    rng.uniform(xlimits[d, 0], xlimits[d, 1], 40 * ndim)
                    for d in range(ndim)
                ]
            )
            yt = np.column_stack([_smooth_fn(xt)[:, 0], np.sum(xt**2, axis=1) - 0.5])
            orders = [min(4, 3 + d) for d in range(ndim)]
            ncps = [10 - d for d in range(ndim)]

            sm = RMTS(
                spline_type="B-Spline",
                xlimits=xlimits,
                order=orders,
                num_ctrl_pts=ncps,
                solver="lu",
                print_global=False,
            )
            sm.set_training_values(xt, yt)
            with Silence():
                sm.train()

            for bad_out in (True, -1, 2, "0"):
                with self.assertRaises(ValueError):
                    sm.get_bspline(output=bad_out)

            xq = np.column_stack(
                [rng.uniform(xlimits[d, 0], xlimits[d, 1], 15) for d in range(ndim)]
            )
            xq_norm = (xq - xlimits[:, 0]) / (xlimits[:, 1] - xlimits[:, 0])

            for out_idx in (0, 1):
                bs_phys = sm.get_bspline(output=out_idx, physical=True)
                bs_norm = sm.get_bspline(output=out_idx, physical=False)
                self.assertEqual(
                    set(bs_phys.keys()), {"knots", "degree", "coefficients", "domain"}
                )
                self.assertEqual(bs_phys["degree"], tuple(o - 1 for o in orders))
                self.assertEqual(bs_phys["coefficients"].shape, tuple(ncps))
                np.testing.assert_allclose(bs_phys["domain"], xlimits)
                np.testing.assert_allclose(
                    bs_norm["domain"], np.tile([0.0, 1.0], (ndim, 1))
                )

                sp_phys = sm.to_scipy(output=out_idx, physical=True)
                sp_norm = sm.to_scipy(output=out_idx, physical=False)
                y_rmtb = sm.predict_values(xq)[:, out_idx]
                np.testing.assert_allclose(sp_phys(xq), y_rmtb, atol=1e-12)
                np.testing.assert_allclose(sp_norm(xq_norm), y_rmtb, atol=1e-12)

                if ndim == 1:
                    sp_1d_phys = sm.to_scipy(output=out_idx, physical=True, as_1d=True)
                    sp_1d_norm = sm.to_scipy(output=out_idx, physical=False, as_1d=True)
                    np.testing.assert_allclose(sp_1d_phys(xq[:, 0]), y_rmtb, atol=1e-12)
                    np.testing.assert_allclose(
                        sp_1d_norm(xq_norm[:, 0]), y_rmtb, atol=1e-12
                    )
                    dy_rmtb_1d = sm.predict_derivatives(xq, 0)[:, out_idx]
                    np.testing.assert_allclose(
                        sp_1d_phys.derivative(1)(xq[:, 0]), dy_rmtb_1d, atol=1e-11
                    )
                else:
                    with self.assertRaises(ValueError):
                        sm.to_scipy(output=out_idx, as_1d=True)

                for kx in range(ndim):
                    nu = tuple(1 if d == kx else 0 for d in range(ndim))
                    dy_rmtb = sm.predict_derivatives(xq, kx)[:, out_idx]
                    np.testing.assert_allclose(sp_phys(xq, nu=nu), dy_rmtb, atol=1e-11)

    def test_energy_cache_and_pickle_roundtrip(self):
        rng = np.random.default_rng(55)
        xlimits = np.array([[-1.0, 1.0], [-2.0, 2.0]])
        xt = np.column_stack([rng.uniform(-1.0, 1.0, 40), rng.uniform(-2.0, 2.0, 40)])
        yt = _smooth_fn(xt)
        xq = np.column_stack([rng.uniform(-1.0, 1.0, 10), rng.uniform(-2.0, 2.0, 10)])

        with tempfile.TemporaryDirectory() as tmpdir:
            for st, opts in [
                ("B-Spline", {"order": 4, "num_ctrl_pts": 8}),
                ("Cubic_Hermite_Spline", {"num_elements": 4}),
            ]:
                # Untrained pickle
                sm0 = RMTS(spline_type=st, xlimits=xlimits, print_global=False, **opts)
                sm0_unpickled = pickle.loads(pickle.dumps(sm0))
                sm0_unpickled.set_training_values(xt, yt)
                with Silence():
                    sm0_unpickled.train()

                # Energy cache miss -> hit + trained pickle
                sm1 = RMTS(
                    spline_type=st,
                    xlimits=xlimits,
                    solver="lu",
                    save_energy_terms=True,
                    data_dir=tmpdir,
                    print_global=False,
                    **opts,
                )
                sm1.set_training_values(xt, yt)
                with Silence():
                    sm1.train()

                sm2 = RMTS(
                    spline_type=st,
                    xlimits=xlimits,
                    solver="lu",
                    save_energy_terms=True,
                    data_dir=tmpdir,
                    print_global=False,
                    **opts,
                )
                sm2.set_training_values(xt, yt)
                with Silence():
                    sm2.train()

                sm1_unpickled = pickle.loads(pickle.dumps(sm1))
                np.testing.assert_allclose(
                    sm1_unpickled.predict_values(xq),
                    sm1.predict_values(xq),
                    atol=1e-14,
                )
                np.testing.assert_allclose(
                    sm2.predict_values(xq),
                    sm1.predict_values(xq),
                    atol=1e-12,
                )

    def test_missing_optional_dependency_fallback(self):
        # Mask jax and numba in sys.modules even when installed locally
        saved = {}
        to_mask = list(
            dict.fromkeys(
                [
                    name
                    for name in list(sys.modules.keys())
                    if name == "jax"
                    or name.startswith("jax.")
                    or name == "numba"
                    or name.startswith("numba.")
                ]
                + ["jax", "numba"]
            )
        )
        try:
            for name in to_mask:
                saved[name] = sys.modules.get(name, "__MISSING__")
                sys.modules[name] = None

            with _temp_env(USE_JAX="1", USE_NUMBA_JIT="1"):
                self.assertEqual(_resolve_backend("auto"), "numpy")
                self.assertEqual(_resolve_backend("jax"), "numpy")
                self.assertEqual(_resolve_backend("numba"), "numpy")

                xlimits = np.array([[0.0, 1.0]])
                xt = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
                yt = np.sin(np.pi * xt)
                for st in ("B-Spline", "Cubic_Hermite_Spline"):
                    sm = RMTS(spline_type=st, xlimits=xlimits, print_global=False)
                    sm.set_training_values(xt, yt)
                    with Silence():
                        sm.train()
                    y = sm.predict_values(xt)
                    self.assertEqual(y.shape, yt.shape)
                    self.assertTrue(np.all(np.isfinite(y)))
        finally:
            for name, orig in saved.items():
                if orig == "__MISSING__":
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = orig

    def test_tolerant_env_parsing_and_rmtc_nterm_guard(self):
        from smt.surrogate_models.rmts_backend import (
            _is_jax_enabled,
            _is_numba_enabled,
        )

        for raw, expected in [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("0", False),
            ("", False),
            ("invalid", False),
        ]:
            with _temp_env(USE_JAX=raw, USE_NUMBA_JIT=raw):
                self.assertEqual(_is_jax_enabled(), expected)
                self.assertEqual(_is_numba_enabled(), expected)

        bk_bad = PureRMTC(
            1,
            np.array([0.0]),
            np.array([1.0]),
            np.array([3], dtype=np.int32),
            np.array([3], dtype=np.int32),
        )
        with self.assertRaises(NotImplementedError):
            bk_bad.compute_coeff2nodal()
        with self.assertRaises(NotImplementedError):
            bk_bad.compute_uniq2elem()

    def test_legacy_rmtb_rmtc_compatibility_for_subclasses(self):
        """Downstream code (e.g. RMTBUQ, PR #845) subclassing RMTB must keep working."""
        import warnings

        from smt.surrogate_models import RMTB, RMTC
        from smt.surrogate_models.rmtb import RMTB as RMTB_shim
        from smt.surrogate_models.rmtc import RMTC as RMTC_shim

        # Legacy module paths still resolve to the unified classes
        self.assertIs(RMTB_shim, RMTB)
        self.assertIs(RMTC_shim, RMTC)

        # Direct use of the deprecated aliases warns...
        xlimits = np.array([[0.0, 1.0]])
        with self.assertWarns(FutureWarning):
            RMTB(xlimits=xlimits, print_global=False)
        with self.assertWarns(FutureWarning):
            RMTC(xlimits=xlimits, print_global=False)

        # ...but subclasses do not
        class MyRMTB(RMTB):
            pass

        class MyRMTC(RMTC):
            pass

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MyRMTB(xlimits=xlimits, print_global=False)
            MyRMTC(xlimits=xlimits, print_global=False)

        # Invalid approx_order is rejected at train(), not at construction,
        # so subclasses can apply their own checks in _train.
        xt = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
        for bad in (0, 1, 3):
            sm = RMTS(
                spline_type="B-Spline",
                xlimits=xlimits,
                approx_order=bad,
                print_global=False,
            )
            sm.set_training_values(xt, xt**2)
            with self.assertRaises(ValueError):
                sm.train()

    def test_numba_pyfunc_and_backend_edge_cases(self):
        from unittest import mock

        from smt.surrogate_models import rmts_backend as rb

        # 1. Pure-Python execution of Numba kernels (matching NumPy & covering all branches)
        kernels = rb._get_numba_kernels()
        if kernels is not None:
            rmtb_fn, rmtc_fn = kernels[0].py_func, kernels[1].py_func
        else:
            rmtb_fn, rmtc_fn = rb._rmtb_jac_numba_raw, rb._rmtc_jac_numba_raw

        # RMTB: order=1 & order=4 with clamping (< 0 and > 1) across deriv=0,1,2
        for orders, ncps in [([1, 4], [4, 7]), ([4, 3], [7, 6])]:
            bk = PureRMTB(2, np.zeros(2), np.ones(2), orders, ncps, _backend="numpy")
            pts = np.array([[-0.2, -0.1], [0.0, 0.5], [0.37, 0.82], [1.2, 1.3]])
            n = pts.shape[0]
            for ix1, ix2 in [(-1, -1), (0, -1), (0, 0), (1, -1), (1, 1)]:
                b0 = np.empty((n, bk.max_order), dtype=np.float64)
                b1 = np.empty((n, bk.max_order), dtype=np.float64)
                b2 = np.empty((n, bk.max_order), dtype=np.float64)
                data = np.empty(n * bk.nnz_row, dtype=np.float64)
                rows = np.empty(n * bk.nnz_row, dtype=np.int32)
                cols = np.empty(n * bk.nnz_row, dtype=np.int32)
                rmtb_fn(
                    ix1,
                    ix2,
                    bk.nx,
                    n,
                    pts,
                    bk.order_list,
                    bk.ncp_list,
                    bk.ncp_strides,
                    bk.knots_flat,
                    bk.knots_offsets,
                    bk.col_offsets,
                    bk.nnz_row,
                    b0,
                    b1,
                    b2,
                    data,
                    rows,
                    cols,
                )
                d_np, r_np, c_np = bk.compute_jac(ix1, ix2, pts)
                np.testing.assert_allclose(data, d_np, atol=1e-13)
                np.testing.assert_array_equal(rows, r_np)
                np.testing.assert_array_equal(cols, c_np)

        # Clamped knots with repeated endpoints (exercises den1 == 0 and den2 == 0)
        bk_clamped = PureRMTB(1, [0.0], [1.0], [4], [5], _backend="numpy")
        clamped_knots = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
        pts_c = np.array([[0.0], [0.25], [0.75], [1.0]])
        for deriv_pair in [(-1, -1), (0, -1), (0, 0)]:
            n_c = pts_c.shape[0]
            d_c = np.empty(n_c * bk_clamped.nnz_row, dtype=np.float64)
            r_c = np.empty(n_c * bk_clamped.nnz_row, dtype=np.int32)
            c_c = np.empty(n_c * bk_clamped.nnz_row, dtype=np.int32)
            rmtb_fn(
                deriv_pair[0],
                deriv_pair[1],
                1,
                n_c,
                pts_c,
                bk_clamped.order_list,
                bk_clamped.ncp_list,
                bk_clamped.ncp_strides,
                clamped_knots,
                np.array([0], dtype=np.int64),
                bk_clamped.col_offsets,
                bk_clamped.nnz_row,
                np.empty((n_c, 4), dtype=np.float64),
                np.empty((n_c, 4), dtype=np.float64),
                np.empty((n_c, 4), dtype=np.float64),
                d_c,
                r_c,
                c_c,
            )
            self.assertTrue(np.all(np.isfinite(d_c)))

        # RMTC: nterm=4 and nterm=1 (exercises nt_ix == 1 branch) across deriv=0,1,2
        for nterm_list in ([4, 4], [1]):
            nx_c = len(nterm_list)
            bk_c = PureRMTC(
                nx_c,
                np.zeros(nx_c),
                np.ones(nx_c),
                np.full(nx_c, 3, dtype=np.int64),
                np.array(nterm_list, dtype=np.int64),
                _backend="numpy",
            )
            pts_m = np.full((3, nx_c), 0.4)
            pts_m[0, :] = -0.2
            pts_m[2, :] = 1.2
            pairs = [(-1, -1), (0, -1), (0, 0)]
            for ix1, ix2 in pairs:
                n_m = pts_m.shape[0]
                data_m = np.empty(n_m * bk_c.nterm, dtype=np.float64)
                rows_m = np.empty(n_m * bk_c.nterm, dtype=np.int32)
                cols_m = np.empty(n_m * bk_c.nterm, dtype=np.int32)
                rmtc_fn(
                    ix1,
                    ix2,
                    bk_c.nx,
                    n_m,
                    pts_m,
                    bk_c.lower,
                    bk_c.upper,
                    bk_c.dxb_dx,
                    bk_c.nelem_list,
                    bk_c.nterm_list,
                    bk_c.nelem_strides,
                    bk_c.nterm,
                    np.empty((n_m, bk_c.max_nterm), dtype=np.float64),
                    data_m,
                    rows_m,
                    cols_m,
                )
                d_np, r_np, c_np = bk_c.compute_jac(ix1, ix2, pts_m)
                np.testing.assert_allclose(data_m, d_np, atol=1e-13)
                np.testing.assert_array_equal(rows_m, r_np)
                np.testing.assert_array_equal(cols_m, c_np)

        # 2. Defensive backend branches & helpers
        with self.assertRaises(ValueError):
            rb._resolve_backend("invalid")

        if rb.has_numba():
            with (
                mock.patch.object(rb, "has_jax", return_value=False),
                _temp_env(USE_NUMBA_JIT="1"),
            ):
                self.assertEqual(rb._resolve_backend("jax"), "numba")

        lower, upper = np.array([0.0, 0.0]), np.array([1.0, 1.0])
        self.assertEqual(compute_ext_dist(np.empty(0), lower, upper).shape, (0, 2))
        ext1 = compute_ext_dist(np.array([[-0.5, 0.5]]), lower, upper, nterm=1)
        np.testing.assert_allclose(ext1, [[-0.5, 0.0]])

        with self.assertRaises(ValueError):
            PureRMTB(2, [0.0], [1.0, 2.0], [4, 4], [5, 5])
        with self.assertRaises(ValueError):
            PureRMTC(2, [0.0], [1.0, 2.0], [3, 3], [4, 4])

        bk_b = PureRMTB(1, [0.0], [2.0], [3], [5], _backend="numpy")
        self.assertEqual(bk_b.compute_ext_dist(np.array([[-1.0]])).shape, (3, 1))
        self.assertEqual(bk_b.compute_quadrature_points([2]).shape, (2, 1))
        self.assertEqual(bk_b.compute_jac(-1, -1, np.empty(0))[0].size, 0)
        self.assertEqual(bk_b.compute_jac(-1, -1, np.empty((0, 1)))[0].size, 0)

        bk_h = PureRMTC(1, [0.0], [2.0], [2], [4], _backend="numpy")
        self.assertEqual(bk_h.compute_ext_dist(np.array([[-1.0]])).shape, (4, 1))
        self.assertEqual(bk_h.compute_quadrature_points().shape, (2, 1))
        self.assertEqual(bk_h.compute_jac(-1, -1, np.empty(0))[0].size, 0)
        self.assertEqual(bk_h.compute_jac(-1, -1, np.empty((0, 1)))[0].size, 0)
        np.testing.assert_allclose(
            PureRMTC(0, [], [], [], []).compute_coeff2nodal(), np.ones((1, 1))
        )

        # Fallback to NumPy inside _compute_jac_numba / _compute_jac_jax + missing JAX errors
        pt1 = np.array([[0.5]])
        with mock.patch.object(rb, "_get_numba_kernels", return_value=None):
            np.testing.assert_allclose(
                bk_b._compute_jac_numba(-1, -1, pt1)[0],
                bk_b._compute_jac_numpy(-1, -1, pt1)[0],
            )
            np.testing.assert_allclose(
                bk_h._compute_jac_numba(-1, -1, pt1)[0],
                bk_h._compute_jac_numpy(-1, -1, pt1)[0],
            )
        with mock.patch.object(rb, "_get_jax", return_value=(None, None)):
            np.testing.assert_allclose(
                bk_b._compute_jac_jax(-1, -1, pt1)[0],
                bk_b._compute_jac_numpy(-1, -1, pt1)[0],
            )
            np.testing.assert_allclose(
                bk_h._compute_jac_jax(-1, -1, pt1)[0],
                bk_h._compute_jac_numpy(-1, -1, pt1)[0],
            )
            with self.assertRaises(ImportError):
                rb.predict_rmtb_jax(pt1, np.ones((5, 1)), [0.0], [2.0], [3], [5])
            with self.assertRaises(ImportError):
                rb.predict_rmtc_jax(pt1, np.ones((8, 1)), [0.0], [2.0], [2])
            with self.assertRaises(ImportError):
                rb._prepare_jax_opt_cache(None, {})

        if rb.has_jax():
            import jax

            with _enable_x64(jax):
                y_b = rb.predict_rmtb_jax(
                    np.array([0.5]), np.ones(5), [0.0], [2.0], [3], [5]
                )
                np.testing.assert_allclose(np.asarray(y_b), 1.0, atol=1e-12)
                # 1D x, 1D sol_coeff, nterm=3 in RMTC JAX (non-cubic polynomial branch)
                y_c = rb.predict_rmtc_jax(
                    np.array([1.0]), np.ones(6), [0.0], [2.0], [2], nterm_list=[3]
                )
                self.assertEqual(np.asarray(y_c).shape, ())
                # 0D scalar inputs work for both predict_rmtb_jax and predict_rmtc_jax
                self.assertEqual(
                    np.asarray(
                        rb.predict_rmtb_jax(0.5, np.ones(5), [0.0], [2.0], [3], [5])
                    ).shape,
                    (),
                )
                self.assertEqual(
                    np.asarray(
                        rb.predict_rmtc_jax(
                            1.0, np.ones(6), [0.0], [2.0], [2], nterm_list=[3]
                        )
                    ).shape,
                    (),
                )


if __name__ == "__main__":
    unittest.main()
